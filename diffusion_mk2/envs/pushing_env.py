import argparse
import numpy as np
import genesis as gs
import collections
import os
from genesis.engine.entities import RigidEntity, MPMEntity
from genesis.engine.entities.rigid_entity import RigidLink
from numpy.typing import NDArray
from typing import Tuple
from scipy.spatial.transform import Rotation as R
import time
from tqdm import tqdm
import diffusion_mk2.utils.dlo_computations as dlo_utils
from diffusion_mk2.inference.inference_state import InferenceState

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



DLO_TARGET = [
    [0.5227354,   0.05167644,  0.70309025],
    [0.53375965,  0.04684567,  0.7036251 ],
    [0.54692805,  0.04094509,  0.70339483],
    [0.5597578,   0.03540486,  0.7035428 ],
    [0.57288355,  0.02992657,  0.70367193],
    [0.58578736,  0.02363537,  0.7034466 ],
    [0.5968107,   0.01501581,  0.70353395],
    [0.601522,    0.00081029,  0.7034673 ],
    [0.599685,   -0.01317035,  0.7034919 ],
    [0.5898543,  -0.02381201,  0.70341164],
    [0.5776969,  -0.03069391,  0.7035469 ],
    [0.56477267, -0.03661821,  0.70367104],
    [0.551634,   -0.04232463,  0.7034438 ],
    [0.53882974, -0.04774107,  0.70359427],
    [0.5272029,  -0.05254408,  0.70393485],
]



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
MODEL_PATH = os.path.join(PROJECT_FOLDER, "weights/dummy-j6akzdd4_model.pt")


class PushingEnv:
    def __init__(
        self,
        cpu: bool = False,
        gui: bool = True,
        vis: bool = True,
        show_fps: bool = False,
        n_episodes: int = NUMBER_OF_EPISODES,
        n_actions: int = NUMBER_OF_ACTIONS_PER_EPISODE,
    ):

        self.cpu = cpu
        self.gui = gui
        self.vis = vis
        self.show_fps = show_fps
        self.obs_deque = None
        self.n_episodes = n_episodes
        self.n_actions = n_actions

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

        #### Initialize modela and observation deque ####
        self.model = InferenceState(MODEL_PATH, device=gs.device)
        obs_horizon = self.model.obs_horizon
        
        # Initialize observation buffer
        observation_ee = self.end_effector.get_pos()[:2].cpu().numpy()
        observation_dlo = dlo_utils.get_skeleton(
            self.rope.get_particles(), 
            NUMBER_OF_PARTICLES, 
            PARTICLES_NUMBER_FOR_POS_SMOOTHING
        )[:, :2]
        dlo_target = np.array(DLO_TARGET)[:, :2]
        obs = np.vstack([observation_ee, observation_dlo, dlo_target])
        self.obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)



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
        path = self.franka.plan_path(
            qpos_goal=qpos,
            num_waypoints=num_waypoints,
        )
        obss = []
        acts = []
        for p in path:
            self.franka.control_dofs_position(p, [*self.motors_dof, *self.fingers_dof])
            self._step()
            ### exit if already reached ###
            if (np.linalg.norm(self.end_effector.get_pos().cpu().numpy()[:2] - target_pos[:2]) < distance_tolerance):
                break

        return np.array(obss), np.array(acts)

    
    def draw_action_trajectory(self, action: NDArray) -> None:
        """
        Visualize the predicted action trajectory.
        
        Args:
            action (NDArray): Array of action positions to visualize
        """
        for i, pos in enumerate(action):
            if len(action) > 1:
                t = i / (len(action) - 1)
            else:
                t = 0.0
            
            # Red to blue gradient
            color = [1.0 - t, 0.0, t, 1.0]

            pos_3d = [pos[0], pos[1], HEIGHT_OFFSET]
            
            self.scene.draw_debug_sphere(
                pos=pos_3d,
                radius=0.005,
                color=color
            )


    def run(self):
        for episode in tqdm(range(self.n_episodes), desc="Episodes"):
            self._reset()
            for step in tqdm(range(self.n_actions), desc="Steps"):
                self.scene.clear_debug_objects()
                dlo_utils.draw_skeleton(
                    particles=DLO_TARGET,
                    scene=self.scene,
                    rope_radius=ROPE_RADIUS,
                )

                ### get_observation ###
                obs_ee = self.end_effector.get_pos().cpu().numpy()[:2]
                obs_dlo = dlo_utils.get_skeleton(self.rope.get_particles(), 
                                                 NUMBER_OF_PARTICLES,
                                                 PARTICLES_NUMBER_FOR_POS_SMOOTHING)[:, :2]
                dlo_target = np.array(DLO_TARGET)[:, :2]
                obs = np.vstack([obs_ee, obs_dlo, dlo_target])
                self.obs_deque.append(obs)
                obs = np.stack(self.obs_deque)
                obs = obs.reshape(self.model.obs_horizon, -1)

                pred_action, pred_actions = self.model.run_inference(
                    observation=obs,
                )
                # Visualize denoising
                # for a in pred_actions:
                #     self.scene.clear_debug_objects()
                #     self.draw_action(a)  # Fixed method call

                # Loop through each waypoint in the predicted action
                self.draw_action_trajectory(pred_action)
                for action in pred_action:
                    qpos = self.franka.inverse_kinematics(
                        link=self.end_effector,
                        pos=action,
                        quat=EE_QUAT_ROTATION,
                        rot_mask=[False, False, True],
                    )
                    qpos[-2:] = 0.0
                    self.franka.control_dofs_position(qpos, [*self.motors_dof, *self.fingers_dof])
                    action_steps = 0
                    # Wait until the end effector reaches the target position or until a maximum number of steps is reached
                    while np.linalg.norm(self.end_effector.get_pos().cpu().numpy()[:2] - action) > 0.0075 and action_steps < int(0.2 // DT):
                        self.scene.step()  # Fixed method call
                        action_steps += 1


             
if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("-v", "--vis", action="store_true", default=False)
        parser.add_argument("-g", "--gui", action="store_true", default=False)
        parser.add_argument("-c", "--cpu", action="store_true", default=False)
        parser.add_argument("-f", "--show_fps", action="store_true", default=False)
        parser.add_argument("-n", "--n_episodes", type=int, default=NUMBER_OF_EPISODES,)
        parser.add_argument("-a", "--n_actions", type=int, default=NUMBER_OF_ACTIONS_PER_EPISODE,)
        args = parser.parse_args()

        ########################## init ##########################
        pushing_dataset_generator = PushingEnv(
            cpu=args.cpu,
            gui=args.gui,
            vis=args.vis,
            show_fps=args.show_fps,
            n_episodes=NUMBER_OF_EPISODES,
            n_actions=NUMBER_OF_ACTIONS_PER_EPISODE,
        )
        pushing_dataset_generator.run()
