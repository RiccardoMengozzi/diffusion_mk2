import argparse
import numpy as np
import genesis as gs
import os
from genesis.engine.entities import RigidEntity, MPMEntity
from genesis.engine.entities.rigid_entity import RigidLink
from numpy.typing import NDArray
from typing import Tuple
from scipy.spatial.transform import Rotation as R
import time
from tqdm import tqdm
import diffusion_mk2.utils.dlo_computations as dlo_utils


np.set_printoptions(
    precision=4,  # number of digits after the decimal
    suppress=True,  # disable scientific notation for small numbers
    linewidth=120,  # max characters per line before wrapping
    threshold=1000,  # max total elements to print before summarizing
    edgeitems=3,  # how many items at array edges to show
    formatter={  # custom formatting per dtype
        float: "{: .4f}".format,
        int: "{: d}".format,
    },
)
PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NUMBER_OF_EPISODES = 3
NUMBER_OF_ACTIONS_PER_EPISODE = 8
ACTION_TIME = 2.0  # seconds (for 256 batch size)
VELOCITY = 0.05  # m/s
DT = 1e-2  # simulation time step
MPM_GRID_DENSITY = 256
SUBSTEPS = 40
TABLE_HEIGHT = 0.7005
HEIGHT_OFFSET = TABLE_HEIGHT
EE_OFFSET = 0.119
EE_QUAT_ROTATION = np.array([0, 0, -1, 0])
ROPE_LENGTH = 0.2
ROPE_RADIUS = 0.003
ROPE_BASE_POSITION = np.array([0.5, 0.0, HEIGHT_OFFSET + ROPE_RADIUS])
NUMBER_OF_PARTICLES = 15
PARTICLES_NUMBER_FOR_POS_SMOOTHING = 10


class PushingDatasetGenerator:
    def __init__(
        self,
        cpu: bool = False,
        gui: bool = True,
        vis: bool = True,
        show_fps: bool = False,
        n_episodes: int = 3,
        n_actions: int = 10,
        action_length: int = 10,
    ):

        self.cpu = cpu
        self.gui = gui
        self.vis = vis
        self.show_fps = show_fps
        self.n_episodes = n_episodes
        self.n_actions = n_actions
        self.action_length = action_length

        self.npz_save_path = os.path.join(
            PROJECT_FOLDER, "npz_data", "test.npz")

        gs.init(
            backend=gs.cpu if self.cpu else gs.gpu,
            logging_level="error",
        )

        ########################## create a scene ##########################
        self.scene: gs.Scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=DT,
                substeps=SUBSTEPS,
            ),
            viewer_options=gs.options.ViewerOptions(
                res=(1080, 720),
                camera_pos=(0.5, 0.0, 1.4),
                camera_lookat=(0.5, 0.0, 0.0),
                camera_fov=80,
                refresh_rate=30,
                max_FPS=240,
            ),
            vis_options=gs.options.VisOptions(
                visualize_mpm_boundary=True,
                show_world_frame=True,
            ),
            mpm_options=gs.options.MPMOptions(
                lower_bound=(0.2, -0.3, HEIGHT_OFFSET - 0.05),
                upper_bound=(0.8, 0.3, HEIGHT_OFFSET + 0.1),
                grid_density=MPM_GRID_DENSITY,
            ),
            show_FPS=self.show_fps,
            show_viewer=self.vis,
        )

        self.cam = self.scene.add_camera(
            res=(1080, 720),
            pos=(0.5, -0.5, TABLE_HEIGHT + 0.6),
            lookat=(0.5, 0.0, TABLE_HEIGHT),
            fov=50,
            GUI=self.gui,
        )

        ########################## entities ##########################
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )

        self.table = self.scene.add_entity(
            morph=gs.morphs.URDF(
                file=os.path.join(PROJECT_FOLDER, "models/SimpleTable/SimpleTable.urdf"),
                pos=(0.0, 0.0, 0.0),
                euler=(0, 0, 90),
                scale=1,
                fixed=True,
            ),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Default(),
        )

        self.rope: MPMEntity = self.scene.add_entity(
            material=gs.materials.MPM.Elastic(
                E=5e4,  # Determines the squishiness of the rope (very low values act as a sponge)
                nu=0.45,
                rho=2000,
                sampler="pbs",
            ),
            morph=gs.morphs.Cylinder(
                height=ROPE_LENGTH,
                radius=ROPE_RADIUS,
                pos=ROPE_BASE_POSITION,
                euler=(90, 0, 0),
            ),
            surface=gs.surfaces.Default(roughness=2, vis_mode="particle"),
        )
        self.franka: RigidEntity = self.scene.add_entity(
            gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda.xml",
                pos=(0.0, 0.0, HEIGHT_OFFSET),
            ),
            material=gs.materials.Rigid(
                friction=2.0,
                needs_coup=True,
                coup_friction=2.0,
                sdf_cell_size=0.005,
            ),
        )

        ########################## build ##########################
        self.scene.build()

        self.motors_dof = np.arange(7)
        self.fingers_dof = np.arange(7, 9)
        self.end_effector : RigidLink = self.franka.get_link("hand")

        # Optional: set control gains
        self.franka.set_dofs_kp(
            np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
        )
        self.franka.set_dofs_kv(
            np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
        )
        self.franka.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        )

    def _reset(self, clear_debug: bool = True) -> None:
        """Reset the environment for a new episode."""
        if clear_debug:
            self.scene.clear_debug_objects()
        # Reset robot to initial position
        qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])
        self.franka.set_qpos(qpos)
        
        # Reset rope with random position
        rope_pos = ROPE_BASE_POSITION + np.array([
            np.random.uniform(low=-0.05, high=0.05),
            np.random.uniform(low=-0.05, high=0.05),
            0.0
        ])
        self.rope.set_position(rope_pos)
        

        # Step to initialize positions
        self._step()



    def _step(
        self,
        track: bool = False,
        link=None,
        track_offset=np.array([0.0, 0.0, 0.0]),
        gui: bool = False,
        render_interval: int = 1,
        current_step: int = 1,
        draw_skeleton_frames: bool = False,
        show_real_time_factor: bool = False,
    ) -> None:
        """
        Step the scene and update the camera.
        """
        start_time = time.time()

        render = False
        if current_step % render_interval == 0:
            render = True

        self.scene.step(update_visualizer=render)
        if draw_skeleton_frames:
            assert (
                self.rope is not None
            ), "Rope entity must be provided to draw skeleton frames."
            particles = self.sample_skeleton_particles(
                self.rope.get_particles(),
                NUMBER_OF_PARTICLES,
                PARTICLES_NUMBER_FOR_POS_SMOOTHING,
            )
            self.draw_skeleton(particles, self.scene)
        if gui:
            if render:
                self.cam.render()
        if track:
            assert link is not None, "Link must be provided to track the camera."
            ee_pos = link.get_pos().cpu().numpy() + track_offset
            self.cam.set_pose(pos=ee_pos, lookat=[ee_pos[0], ee_pos[1], 0.0])

        end_time = time.time()
        if show_real_time_factor:
            real_time_factor = DT / (end_time - start_time)
            print(f"Real-time factor: {real_time_factor:.4f}x")


    def _get_random_ee_position(self,
                                particles: NDArray[np.float32],
                                idx: int,   
                                max_distance: float = 0.1,
                                min_distance: float = 0.01,
                                max_attempts: int = 1000
                                ) -> NDArray:
        """
        Compute the initial end-effector pose based on a particle's position and orientation.
        Args:
            particles (NDArray[np.float32]): Array of rope particle positions
            particle_pos (NDArray[np.float32]): Position of the selected particle
            particle_quat (NDArray[np.float32]): Quaternion orientation of the selected particle
        Returns:    
            Tuple[NDArray, NDArray]: Initial end-effector position and quaternion
        """
        particle_pos = particles[idx]
        for _ in range(max_attempts):
            # 1. Random angle and distance for XY
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(min_distance, max_distance)

            # Calculate proposed XY relative to particle_pos
            delta_x = distance * np.cos(angle)
            delta_y = distance * np.sin(angle)

            # Proposed 3D position
            proposed_ee_pos = np.array([
                particle_pos[0] + delta_x,
                particle_pos[1] + delta_y,
                HEIGHT_OFFSET + EE_OFFSET # Fixed Z
            ])

            # 2. Check clearance from all rope particles
            is_too_close = False
            for p in particles:
                # Only consider XY distance for clearance check
                dist_to_particle = np.linalg.norm(proposed_ee_pos[:2] - p[:2])
                if dist_to_particle < min_distance:
                    is_too_close = True
                    break
            
            if not is_too_close:
                return proposed_ee_pos
            
        # If loop finishes without finding a valid pose
        raise RuntimeError(
            f"Could not find a suitable initial EE pose after {max_attempts} attempts."
        )
        


    def _move_to_target_pose(
        self,
        target_pos: NDArray,
        target_quat: NDArray,
        path_period: float = 0.5,
        distance_tolerance: float = 0.005,
    ):
        qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=target_pos,
            quat=target_quat,
        )
        qpos[-2:] = 0.0  # Close fingers

        num_waypoints = int(path_period // DT)
        print(f"Number of waypoints: {num_waypoints}")
        path = self.franka.plan_path(
            qpos_goal=qpos,
            num_waypoints=num_waypoints,
        )
        obss = []
        acts = []
        for p in path:
            self.franka.control_dofs_position(p, [*self.motors_dof, *self.fingers_dof])

            ####### save ee_init and dlo state obs #######
            obs_ee = self.end_effector.get_pos()[:2].cpu().numpy()
            obs_dlo = dlo_utils.get_skeleton(self.rope.get_particles(),
                                             downsample_number=NUMBER_OF_PARTICLES,
                                             average_number=PARTICLES_NUMBER_FOR_POS_SMOOTHING)[:, :2]

            self._step()

            ####### save ee_final observation and ee_final action #######
            obs_dlo_f = dlo_utils.get_skeleton(self.rope.get_particles(),
                                               downsample_number=NUMBER_OF_PARTICLES,
                                               average_number=PARTICLES_NUMBER_FOR_POS_SMOOTHING)[:, :2]
            act = self.end_effector.get_pos()[:2].cpu().numpy()

            ####### save observations and actions #######
            obss.append(np.vstack([obs_ee, obs_dlo, obs_dlo_f]))
            acts.append(act) # so it's (1, 2) instead of (2,)

            ### exit if already reached ###
            if (np.linalg.norm(self.end_effector.get_pos().cpu().numpy()[:2] - target_pos[:2]) < distance_tolerance):
                break

        return np.array(obss), np.array(acts)

    def _get_random_action_through_dlo(
        self,
        particles: NDArray[np.float32],
        idx: int,
        ee_position: NDArray,
        min_strike_distance: float = 0.01,
        max_strike_distance: float = 0.1,
        debug: bool = False,
    ) -> NDArray:
        """
        Get the target position for the action based on a particle index.

        Args:
            particles (NDArray[np.float32]): Array of rope particle positions
            particle_index (int): Index of the particle to use for target position
            initial_ee_pos (NDArray): Initial end-effector position

        Returns:
            NDArray: Target position for the action
        """
        particle_position = particles[idx]
        # Get the 2D position of the target particle
        target_particle_xy = particle_position[:2]

        # Calculate the vector from initial_ee_pos (XY) to the target particle (XY)
        vector_to_target_particle = target_particle_xy - ee_position[:2]

        # Calculate the scalar distance from initial_ee_pos to the target particle
        distance_to_target_particle = np.linalg.norm(vector_to_target_particle)

        # Normalize the direction vector
        # Handle case where initial_ee_pos is exactly on the particle (unlikely but possible)
        if distance_to_target_particle < 1e-6:  # Epsilon to avoid division by zero
            action_direction_xy = np.array([0.0, 1.0])  # Default to +Y if at target
        else:
            action_direction_xy = (
                vector_to_target_particle / distance_to_target_particle
            )

        # Determine the total scalar movement distance
        # You want to strike through the particle, so the movement should be
        # (distance to particle) + (some additional strike depth beyond particle)
        strike_depth_beyond_particle = np.random.uniform(
            min_strike_distance, max_strike_distance
        )
        total_movement_distance = (
            distance_to_target_particle + strike_depth_beyond_particle
        )

        # Calculate the final target position
        # Start from initial_ee_pos and move along the normalized direction
        action_target_pos_xy = (
            ee_position[:2] + action_direction_xy * total_movement_distance
        )

        action_target_pos = np.array(
            [
                action_target_pos_xy[0],
                action_target_pos_xy[1],
                HEIGHT_OFFSET + EE_OFFSET + ROPE_RADIUS / 2,  # Fixed Z position
            ]
        )

        if debug:
            self.scene.clear_debug_objects()
            # Draw particle position
            self.scene.draw_debug_sphere(
                pos=particle_position,
                radius=0.005,
                color=(1, 0, 0, 1),
            )
            self.scene.draw_debug_line(
                start=np.array(
                    [
                        ee_position[0],
                        ee_position[1],
                        HEIGHT_OFFSET + 0.01,
                    ]
                ),
                end=action_target_pos - np.array([0.0, 0.0, EE_OFFSET]),
                radius=0.001,
                color=(0, 0, 1, 1),  # Blue color for action direction
            )
            # Draw target position
            self.scene.draw_debug_sphere(
                pos=action_target_pos - np.array([0.0, 0.0, EE_OFFSET]),
                radius=0.005,
                color=(0, 1, 0, 1),
            )

        return action_target_pos, total_movement_distance

    def _get_random_action_inside_bounding_box(
        self,
        ee_position: NDArray,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        debug: bool = True,
    ) -> tuple[NDArray, float]:
        """
        Get a random target position for the action that remains within specified bounds.

        Args:
            particles (NDArray[np.float32]): Array of rope particle positions
            ee_position (NDArray): Current end-effector position
            x_bounds (tuple[float, float]): Min and max X coordinates (x_min, x_max)
            y_bounds (tuple[float, float]): Min and max Y coordinates (y_min, y_max)
            debug (bool): Whether to draw debug visualizations

        Returns:
            tuple[NDArray, float]: Target position for the action and movement distance
        """
        x_min, x_max = x_bounds
        y_min, y_max = y_bounds
        
        # Generate random target position within the bounding box
        target_x = np.random.uniform(x_min, x_max)
        target_y = np.random.uniform(y_min, y_max)
        
        # Create the target position in XY plane
        target_pos_xy = np.array([target_x, target_y])
        
        # Calculate movement vector and distance
        movement_vector_xy = target_pos_xy - ee_position[:2]
        movement_distance = np.linalg.norm(movement_vector_xy)
        
        # Create the full 3D target position
        action_target_pos = np.array([
            target_x,
            target_y,
            HEIGHT_OFFSET + EE_OFFSET + ROPE_RADIUS / 2,  # Fixed Z position
        ])
        
        if debug:
            self.scene.clear_debug_objects()
            
            # Draw current end-effector position
            self.scene.draw_debug_sphere(
                pos=np.array([
                    ee_position[0],
                    ee_position[1],
                    HEIGHT_OFFSET + EE_OFFSET + ROPE_RADIUS / 2,
                ]),
                radius=0.005,
                color=(1, 1, 0, 1),  # Yellow for current EE position
            )
            
            # Draw movement vector
            if movement_distance > 1e-6:  # Only draw if there's actual movement
                self.scene.draw_debug_arrow(
                    pos=np.array([
                        ee_position[0],
                        ee_position[1],
                        HEIGHT_OFFSET + 0.01,
                    ]),
                    vec=np.array([
                        movement_vector_xy[0],
                        movement_vector_xy[1],
                        0.0,  # Keep Z constant
                    ]) * 0.8,  # Scale to show most of the movement
                    radius=0.005,
                    color=(0, 0, 1, 1),  # Blue for movement direction
                )
            
            # Draw target position
            self.scene.draw_debug_sphere(
                pos=action_target_pos - np.array([0.0, 0.0, EE_OFFSET]),
                radius=0.005,
                color=(0, 1, 0, 1),  # Green for target position
            )
            
            # Draw bounding box outline (optional - creates a wireframe box)
            # Corner points of the bounding box at the working height
            box_height = HEIGHT_OFFSET + 0.005
            box_corners = [
                [x_min, y_min, box_height],
                [x_max, y_min, box_height],
                [x_max, y_max, box_height],
                [x_min, y_max, box_height],
            ]
            
            # Draw bounding box edges
            for i in range(4):
                start_corner = np.array(box_corners[i])
                end_corner = np.array(box_corners[(i + 1) % 4])
                
                self.scene.draw_debug_line(
                    start=start_corner,
                    end=end_corner,
                    radius=0.002,
                    color=(0.5, 0.5, 0.5, 0.8),  # Gray for bounding box
                )
        
        return action_target_pos, movement_distance


    def _clean_data(self, obss, acts, number_of_data) -> Tuple[NDArray, NDArray]:
        obss_indices = dlo_utils.downsample_array(obss, self.action_length)
        acts_indices = dlo_utils.downsample_array(acts, self.action_length)
        cleaned_obss = obss[obss_indices]
        cleaned_acts = acts[acts_indices]

        return cleaned_obss, cleaned_acts

    def run(self):
        observations = []
        actions = []
        episode_ends = []
        steps_counter = 0
        for episode in tqdm(range(self.n_episodes), desc="Episodes"):
            self._reset()
            for step in tqdm(range(self.n_actions), desc="Steps"):
                # 1. Get the skeleton
                rope_skeleton = dlo_utils.get_skeleton(self.rope.get_particles(),
                                                       downsample_number=NUMBER_OF_PARTICLES,
                                                       average_number=PARTICLES_NUMBER_FOR_POS_SMOOTHING)

                # 2. Choose a random index
                idx = np.random.randint(0, len(rope_skeleton) - 1)  

                # 3. given the idx, choose a rondom starting ee poisition, and get the proper ee orientation
                _, ee_quat = dlo_utils.compute_pose_from_paticle_index(
                    rope_skeleton, idx, EE_QUAT_ROTATION, EE_OFFSET
                )
                ee_position = self._get_random_ee_position(particles=rope_skeleton, 
                                                           idx=idx,
                                                           max_distance=0.1)

                # 4. set the ee pose
                qpos = self.franka.inverse_kinematics(
                    link=self.end_effector,
                    pos=ee_position,
                    quat=ee_quat,
                    # rot_mask=[True, True, False],
                )
                qpos[-2:] = 0.0  # set fingers to 0
                self.franka.set_qpos(qpos=qpos)

                # 5. given the idx and ee_position, choose a random action that strikes the rope through the idx
                action_pos, distance = self._get_random_action_through_dlo(
                                                               particles=rope_skeleton, 
                                                               idx=idx,
                                                               ee_position=ee_position,
                                                               debug=True)
                # ee_position = self.end_effector.get_pos().cpu().numpy()
                # ee_quat = self.end_effector.get_quat().cpu().numpy()
                # action_pos, distance =self._get_random_action_inside_bounding_box(
                #     ee_position=ee_position,
                #     x_bounds=(0.3, 0.6),
                #     y_bounds=(-0.2, 0.2),
                # )
                
                path_period = distance / VELOCITY
                obss, acts = self._move_to_target_pose(target_pos=action_pos,
                                                       target_quat=ee_quat,
                                                       path_period=path_period,
                                                       distance_tolerance=0.005,
                                                       )                             

                # 6. clean the observations and actions:
                #   - remove data where ee_init and ee_final are very close
                #   - sample them depending on the number of data per action
                clean_obss, clean_acts = self._clean_data(obss, acts, self.action_length)

                # 7. save 
                observations.append(clean_obss)
                actions.append(clean_acts)
                steps_counter += len(clean_obss)

                print("observations shape:", np.array(observations).shape)
                print("actions shape:", np.array(actions).shape)

                # 8. go back to 1.
            episode_ends.append(steps_counter)

        # Save observations and actions
        observations = np.array(observations)
        actions = np.array(actions)
        print("obss shape:", clean_obss.shape)
        print("acts shape:", clean_acts.shape)
        observations = observations.reshape(-1, clean_obss.shape[-2], clean_obss.shape[-1]) # from (episodes, steps, obs_dim, 2) to (N,  obs_dim, 2)
        actions = actions.reshape(-1, clean_acts.shape[-1]) # from (episodes, steps, act_dim, 2) to (N, 2)
        print(f"Final observations shape: {observations.shape}")
        print(f"Final actions shape: {actions.shape}")
        np.savez(self.npz_save_path, observations=observations, actions=actions, episode_ends=episode_ends)
        print(f"Saved dataset with {len(observations)} observations and actions.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-g", "--gui", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    parser.add_argument("-f", "--show_fps", action="store_true", default=False)
    parser.add_argument(
        "-e",
        "--n_episodes",
        type=int,
        default=NUMBER_OF_EPISODES,
        help="Number of episodes to run. Default is 3.",
    )
    parser.add_argument("-a", "--n_actions", type=int, default=10)
    parser.add_argument("-l", "--action_length", type=int, default=10,)
    args = parser.parse_args()

    ########################## init ##########################
    pushing_dataset_generator = PushingDatasetGenerator(
        cpu=args.cpu,
        gui=args.gui,
        vis=args.vis,
        show_fps=args.show_fps,
        n_episodes=args.n_episodes,
        n_actions=args.n_actions,
        action_length=args.action_length,
    )
    pushing_dataset_generator.run()
