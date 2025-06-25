import os
import argparse
import numpy as np
import genesis as gs
import time
from tqdm import tqdm
from genesis.engine.entities import RigidEntity, MPMEntity
from genesis.engine.entities.rigid_entity import RigidLink
import diffusion_mk2.utils.dlo_computations as dlo_utils
from scipy.spatial.transform import Rotation as R
from diffusion_mk2.utils.dlo_shapes import U_SHAPE, S_SHAPE

SHAPES = {
    "U_SHAPE": U_SHAPE,
    "S_SHAPE": S_SHAPE
    }


PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NUMBER_OF_EPISODES = 3
NUMBER_OF_ACTIONS_PER_EPISODE = 8 # This is not directly used in the teleop script, but kept for consistency
ACTION_TIME = 2.0  # seconds (for 256 batch size)
VELOCITY = 0.05  # m/s
DT = 1e-2  # simulation time step
MPM_GRID_DENSITY = 256
SUBSTEPS = 40
TABLE_HEIGHT = 0.7005
HEIGHT_OFFSET = TABLE_HEIGHT
EE_OFFSET = 0.105
EE_Z = 0.07
EE_QUAT_ROTATION = np.array([0, 0, -1, 0])
ROPE_LENGTH = 0.2
ROPE_RADIUS = 0.003
ROPE_BASE_POSITION = np.array([0.5, 0.0, HEIGHT_OFFSET + ROPE_RADIUS])
NUMBER_OF_PARTICLES = 15
PARTICLES_NUMBER_FOR_POS_SMOOTHING = 10


EE_VELOCITY = 0.02
EE_ANG_VELOCITY = 0.2
SAVE_DATA_INTERVAL = 3 # every 3 steps
CLOSE_GRIPPER_POSITION = 0.00   
OPEN_GRIPPER_POSITION = 0.01  




class GripperEnv():
    def __init__(self, 
                 vis=False, 
                 gui=False, 
                 cpu=False,
                 n_episodes=NUMBER_OF_EPISODES,
                 n_actions=NUMBER_OF_ACTIONS_PER_EPISODE,
                 show_fps=False, 
                 save_name="dummy"):
        
        self.vis = vis
        self.gui = gui
        self.cpu = cpu
        self.n_episodes = n_episodes
        self.n_actions = n_actions
        self.show_fps = show_fps
        self.real_time_factor = 0.0


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
                friction=5.0,
                needs_coup=True,
                coup_friction=5.0,
                sdf_cell_size=0.005,
                gravity_compensation=1.0,
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

        self.initial_pose = np.array([0.45, 0.0, HEIGHT_OFFSET + EE_OFFSET + EE_Z, 0.0, 0.707, 0.707, 0.0])



    def _step(self):
        start_time = time.time()

        self.scene.step()
        end_time = time.time()
        self.real_time_factor = DT / (end_time - start_time)


    def reset(self):
        """Reset the environment for a new episode."""
        self.scene.clear_debug_objects()

        # Choose new target
        self.target = 

        # Place robot aboce centre of the rope
        skeleton = dlo_utils.get_skeleton(self.rope.get_particles(),
                                            downsample_number=NUMBER_OF_PARTICLES,
                                            average_number=PARTICLES_NUMBER_FOR_POS_SMOOTHING)
        skeleton_centre = np.array([np.mean(skeleton[:, 0]),
                                    np.mean(skeleton[:, 1])])
        
        target_pos = [skeleton_centre[0], skeleton_centre[1], HEIGHT_OFFSET + EE_OFFSET + EE_Z]
        target_quat = self.initial_pose[3:7]  # Use the initial pose's quaternion

        qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=target_pos,
            quat=target_quat,
        )
        qpos[-2:] = OPEN_GRIPPER_POSITION  # Open gripper at the start

        self.franka.set_qpos(qpos)
        self._step()


    def get_observation(self):
        pos_ee = self.end_effector.get_pos().cpu().numpy()
        theta = R.from_quat(self.end_effector.get_quat().cpu().numpy()).as_euler('xyz')[0] # dont ask why [0], but it works
        finger_qpos = self.franka.get_qpos().cpu().numpy()[-1]
        obs_ee = np.array([pos_ee[0], pos_ee[1], pos_ee[2], theta, finger_qpos])

        # self.scene.draw_debug_sphere(
        #     pos=np.array([obs_ee[0], obs_ee[1], obs_ee[2] - EE_OFFSET]),
        #     radius=0.001,
        #     color=(1, 0, 0, 1),  # Red color for end effector
        # )
        obs_dlo = dlo_utils.get_skeleton(self.rope.get_particles(),
                                            downsample_number=NUMBER_OF_PARTICLES,
                                            average_number=PARTICLES_NUMBER_FOR_POS_SMOOTHING)
        obs_target = self.target
        return obs_ee, obs_dlo, obs_target
 
    def move(self, 
             target_pos=None,
             target_quat=None,
             qpos=None,
             gripper_open=False,
             force_control=False,
             force_intensity=1,
             path_period=0.5,
             tolerance=1e-7):
        """ Primitive move function """
        # If I already provide qpos, i don't compute it
        if qpos is None:
            qpos = self.franka.inverse_kinematics(
                link=self.end_effector,
                pos=target_pos,
                quat=target_quat,
            )
        
        # Create the path to follow
        if path_period == DT:
            path = [qpos]
        else:
            path = self.franka.plan_path(
                qpos_goal=qpos,
                num_waypoints=int(path_period // DT),
                ignore_collision=True, # Otherwise cannot grasp in a good way the rope
            )

        # Control the gripper
        if not gripper_open:
            qpos[-2:] = CLOSE_GRIPPER_POSITION
        else:
            qpos[-2:] = OPEN_GRIPPER_POSITION

        # Control the robot along the path
        for p in path:

            self.franka.control_dofs_position(p[:-2], self.motors_dof)
            if force_control:
                self.franka.control_dofs_force([-force_intensity, -force_intensity], self.fingers_dof)
            else:
                self.franka.control_dofs_position(p[-2:], self.fingers_dof)
            self._step()

            # Check if the robot has reached the target position
            if np.linalg.norm(qpos.cpu().numpy() - self.franka.get_qpos().cpu().numpy()) < tolerance:
                break

    def release(self):
        """Release the grasped object."""
        # Open the gripper
        qpos = self.franka.get_qpos()
        qpos[-2:] = OPEN_GRIPPER_POSITION  # Open the gripper
        self.move(
            qpos=qpos,
            path_period=0.5,
            gripper_open=True,  # Open the gripper
        )

        # Move up
        self.move(
            target_pos=self.end_effector.get_pos().cpu().numpy() + np.array([0, 0, EE_Z]),
            target_quat=self.end_effector.get_quat().cpu().numpy(),
            path_period=0.5,
            gripper_open=True,  # Keep the gripper open
        )

    def run(self):
        for episode in tqdm(range(self.n_episodes)):
            self.reset()
            for action in tqdm(range(self.n_actions)):
                # Get observation
                obs_ee, obs_dlo, obs_target = self.get_observation()

                # Select idx randomly
                idx = np.random.randint(0, len(skeleton))

                # Grasp
                self.grasp(idx)

                # Move randomly
                self.random_action()

                # Release
                # self.release() # currently not savind release data

                # Update all previous action data with current dlo state as target
                self.update_action_data(show_target=False)

                # Save the episode
                self.data_logger.save_episode()
                



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teleop Push Data Generator")
    parser.add_argument("-v", "--vis", action="store_true")
    parser.add_argument("-g", "--gui", action="store_true", help="Enable GUI mode")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of GPU")
    parser.add_argument("-f", "--show_fps", action="store_true", help="Show FPS in the viewer")
    parser.add_argument("-e", "--n_episodes", type=int, default=NUMBER_OF_EPISODES, help="Number of episodes to run")
    parser.add_argument("-a", "--n_actions", type=int, default=NUMBER_OF_ACTIONS_PER_EPISODE, help="Number of actions per episode")
    parser.add_argument("-n", "--save_name", type=str, default="dummy", help="save name")
    args = parser.parse_args()

    gripper_env = GripperEnv(vis=args.vis, 
                             gui=args.gui, 
                             cpu=args.cpu, 
                             n_episodes=args.n_episodes,
                             show_fps=args.show_fps, 
                             save_name=args.save_name)
    gripper_env.run()