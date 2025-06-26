import os
import argparse
import numpy as np
import genesis as gs
import time
import random
from tqdm import tqdm
import collections
import torch
from genesis.engine.entities import RigidEntity, MPMEntity
from genesis.engine.entities.rigid_entity import RigidLink
import diffusion_mk2.utils.dlo_computations as dlo_utils
import diffusion_mk2.utils.gs_utils as gs_utils
from diffusion_mk2.utils.utils import load_yaml
from scipy.spatial.transform import Rotation as R
from diffusion_mk2.utils.dlo_shapes import U_SHAPE, S_SHAPE
from diffusion_mk2.inference.inference_state import InferenceState


SHAPES = [U_SHAPE, S_SHAPE]
PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class GripperEnv():
    def __init__(self, args):
        
        cfg = load_yaml(os.path.join(PROJECT_FOLDER, args.cfg))

        # simulation
        self.vis = args.vis if args.vis is not None else cfg["simulation"].get("visualization", True)
        self.gui = args.gui if args.gui is not None else cfg["simulation"].get("camera_gui", False)
        self.cpu = args.cpu if args.cpu is not None else cfg["simulation"].get("cpu", False)
        self.show_fps = args.show_fps if args.show_fps is not None else cfg["simulation"].get("show_fps", False)
        self.dt = cfg["simulation"].get("dt")
        self.substeps = cfg["simulation"].get("substeps")
        self.show_real_time_factor = cfg["simulation"].get("show_real_time_factor", True)

        # viewer
        self.viewer_resolution = cfg["simulation"]["viewer"].get("resolution")
        self.viewer_camera_position = cfg["simulation"]["viewer"].get("position")
        self.viewer_camera_lookat = cfg["simulation"]["viewer"].get("lookat")
        self.viewer_camera_fov = cfg["simulation"]["viewer"].get("fov")
        self.viewer_refresh_rate = cfg["simulation"]["viewer"].get("refresh_rate")
        self.viewer_max_fps = cfg["simulation"]["viewer"].get("max_fps")

        # camera:
        self.camera_gui = cfg["simulation"]["camera"].get("gui", False)

        # entities
        self.table_height = cfg["entities"]["table"].get("height")
        self.table_position = cfg["entities"]["table"].get("position")
        self.table_orientation = cfg["entities"]["table"].get("orientation")
        self.table_scale = cfg["entities"]["table"].get("scale")
        self.mpm_grid_density = cfg["entities"]["dlo"].get("mpm_grid_density")
        self.mpm_lower_bound = cfg["entities"]["dlo"].get("mpm_lower_bound")
        self.mpm_upper_bound = cfg["entities"]["dlo"].get("mpm_upper_bound")
        self.dlo_position = cfg["entities"]["dlo"].get("position")
        self.dlo_orientation = cfg["entities"]["dlo"].get("orientation")
        self.dlo_number_of_particles = cfg["entities"]["dlo"].get("number_of_particles")
        self.dlo_number_of_particles_smoothing = cfg["entities"]["dlo"].get("particles_smoothing")
        self.dlo_length = cfg["entities"]["dlo"].get("length")
        self.dlo_radius = cfg["entities"]["dlo"].get("radius")
        self.dlo_E = cfg["entities"]["dlo"].get("E")  # Young's modulus
        self.dlo_nu = cfg["entities"]["dlo"].get("nu")  # Poisson's ratio
        self.dlo_rho = cfg["entities"]["dlo"].get("rho")  # Density
        self.dlo_sampler = cfg["entities"]["dlo"].get("sampler") # Sampler type


        # franka
        self.franka_position = cfg["entities"]["franka"].get("position")
        self.franka_orientation = cfg["entities"]["franka"].get("orientation")
        print("franka orientation: ", self.franka_orientation)
        self.franka_gravity_compensation = cfg["entities"]["franka"].get("gravity_compensation")
        self.ee_friction = cfg["entities"]["franka"]["end_effector"].get("friction")
        self.ee_needs_coup = cfg["entities"]["franka"]["end_effector"].get("needs_coup")
        self.ee_coup_friction = cfg["entities"]["franka"]["end_effector"].get("coup_friction")
        self.ee_sdf_cell_size = cfg["entities"]["franka"]["end_effector"].get("sdf_cell_size")
        self.ee_z_offset = cfg["entities"]["franka"]["end_effector"].get("offset")
        self.ee_z_lift = cfg["entities"]["franka"]["end_effector"].get("z_lift")
        self.ee_rot_offset = cfg["entities"]["franka"]["end_effector"].get("rot_offset")
        self.gripper_open_position = cfg["entities"]["franka"]["end_effector"].get("gripper_open_position")
        self.gripper_closed_position = cfg["entities"]["franka"]["end_effector"].get("gripper_closed_position")

        # inference
        self.model_path = os.path.join(PROJECT_FOLDER, cfg["inference"].get("model_path"))
        self.n_episodes = args.n_episodes if args.n_episodes is not None else cfg["inference"].get("n_episodes")
        self.n_actions = args.n_actions if args.n_actions is not None else cfg["inference"].get("n_actions")



        gs.init(
            backend=gs.cpu if self.cpu else gs.gpu,
            logging_level="error",
        )

        ########################## create a scene ##########################
        self.scene: gs.Scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=self.substeps,
            ),
            viewer_options=gs.options.ViewerOptions(
                res=self.viewer_resolution,
                camera_pos=self.viewer_camera_position,
                camera_lookat=self.viewer_camera_lookat,
                camera_fov=self.viewer_camera_fov,
                refresh_rate=self.viewer_refresh_rate,
                max_FPS=self.viewer_max_fps,
            ),
            vis_options=gs.options.VisOptions(
                visualize_mpm_boundary=True,
                show_world_frame=True,
            ),
            mpm_options=gs.options.MPMOptions(
                lower_bound=self.mpm_lower_bound,
                upper_bound=self.mpm_upper_bound,
                grid_density=self.mpm_grid_density,
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
                pos=self.table_position,
                euler=self.table_orientation,
                scale=self.table_scale,
                fixed=True,
            ),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Default(),
        )

        self.rope: MPMEntity = self.scene.add_entity(
            material=gs.materials.MPM.Elastic(
                E=self.dlo_E,  # Determines the squishiness of the rope (very low values act as a sponge)
                nu=self.dlo_nu,
                rho=self.dlo_rho,
                sampler=self.dlo_sampler,
            ),
            morph=gs.morphs.Cylinder(
                height=self.dlo_length,
                radius=self.dlo_radius,
                pos=self.dlo_position,
                euler=self.dlo_orientation,
            ),
            surface=gs.surfaces.Default(roughness=2, vis_mode="particle"),
        )
        self.franka: RigidEntity = self.scene.add_entity(
            gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda.xml",
                pos=self.franka_position,
                euler=self.franka_orientation,
            ),
            material=gs.materials.Rigid(
                friction=self.ee_friction,
                needs_coup=self.ee_needs_coup,
                coup_friction=self.ee_coup_friction,
                sdf_cell_size=self.ee_sdf_cell_size,
                gravity_compensation=self.franka_gravity_compensation,
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

        self.initial_pose = np.array([0.45, 0.0, self.table_height + self.ee_z_offset + self.ee_z_lift, 
                                      0.0, 0.707, 0.707, 0.0])
        self.target = random.choice(SHAPES)

        #### Initialize model and observation deque ####
        self.model = InferenceState(self.model_path, device=gs.device)
        obs_horizon = self.model.obs_horizon
        
        # Initialize observation buffer
        obs = self.get_observation()
        self.obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)


    def _step(self):
        start_time = time.time()

        self.scene.step()
        end_time = time.time()
        self.real_time_factor = self.dt / (end_time - start_time)

        if self.show_real_time_factor:
            print(f"Real-time factor: {self.real_time_factor:.2f}")


    def reset(self):
        """Reset the environment for a new episode."""
        self.scene.clear_debug_objects()

        # Choose new target
        self.target = random.choice(SHAPES)
        dlo_utils.draw_skeleton(self.target, self.scene, self.dlo_radius)

        # Place robot aboce centre of the rope
        skeleton = dlo_utils.get_skeleton(self.rope.get_particles(),
                                            downsample_number=self.dlo_number_of_particles,
                                            average_number=self.dlo_number_of_particles_smoothing)
        skeleton_centre = np.array([np.mean(skeleton[:, 0]),
                                    np.mean(skeleton[:, 1])])
        
        target_pos = [skeleton_centre[0], skeleton_centre[1], self.table_height + self.ee_z_offset + self.ee_z_lift]
        target_quat = self.initial_pose[3:7]  # Use the initial pose's quaternion

        qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=target_pos,
            quat=target_quat,
        )
        qpos[-2:] = self.gripper_open_position # Open gripper at the start

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
                                            downsample_number=self.dlo_number_of_particles,
                                            average_number=self.dlo_number_of_particles_smoothing)
        obs_target = self.target

        obs_ee = np.array(obs_ee).flatten()
        obs_dlo = np.array(obs_dlo).flatten()
        obs_target = np.array(obs_target).flatten()
        obs = np.concatenate(
                [obs_ee, obs_dlo, obs_target], axis=-1
                )


        return obs
 
    def move(self, 
             target_pos=None,
             target_quat=None,
             qpos=None,
             gripper_open=None,
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
        if path_period == self.dt:
            path = [qpos]
        else:
            path = self.franka.plan_path(
                qpos_goal=qpos,
                num_waypoints=int(path_period // self.dt),
                ignore_collision=True, # Otherwise cannot grasp in a good way the rope
            )

        # Control the gripper
        if gripper_open is not None:
            if not gripper_open:
                qpos[-2:] = self.gripper_closed_position
            else:
                qpos[-2:] = self.gripper_open_position  


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

    def do_action(self, action):
        for a in action:
            target_pos = a[:3]

            q_old = R.from_euler('xyz', [a[3], 0, 0]).as_quat()
            # q_old ≈ [0.77541212, 0., 0., 0.63145549]

            # 2) costruisci l’asse usando (x_old, w_old):
            v = np.array([0.0, q_old[0], q_old[3]])      # [0, 0.7754, 0.6315]
            u = v / np.linalg.norm(v)                    # asse unitario

            # 3) crea il nuovo quaternion con angle = π rad
            target_quat = R.from_rotvec(np.pi * u).as_quat()
            
            target_gripper = a[4]


            print(f"quat =  {target_quat}, theta = {a[3]}")

            qpos = self.franka.inverse_kinematics(
                link=self.end_effector,
                pos=target_pos,
                quat=target_quat,
            )
            qpos[-2:] = torch.tensor([target_gripper, target_gripper], dtype=qpos.dtype, device=qpos.device)


            self.move(
                qpos=qpos,
                path_period=0.1,
            )

    def release(self):
        """Release the grasped object."""
        # Open the gripper
        qpos = self.franka.get_qpos()
        qpos[-2:] = self.gripper_open_position  # Open the gripper
        self.move(
            qpos=qpos,
            path_period=0.5,
            gripper_open=True,  # Open the gripper
        )

        # Move up
        self.move(
            target_pos=self.end_effector.get_pos().cpu().numpy() + np.array([0, 0, self.ee_z_lift]),
            target_quat=self.end_effector.get_quat().cpu().numpy(),
            path_period=0.5,
            gripper_open=True,  # Keep the gripper open
        )

    def run(self):
        for episode in tqdm(range(self.n_episodes)):
            self.reset()
            for action in tqdm(range(self.n_actions)):
                # Get observation
                obs = self.get_observation()

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
                gs_utils.draw_action_trajectory(self.scene, pred_action, self.ee_z_offset, radius=0.001)

                self.do_action(pred_action)
                
            # Release
            self.release() 




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teleop Push Data Generator")
    parser.add_argument("--cfg", type=str, default="diffusion_mk2/config/dlo_shapes_with_grasping_env.yaml", help="Path to the configuration file")
    parser.add_argument("-v", "--vis", action="store_true")
    parser.add_argument("-g", "--gui", action="store_true", help="Enable GUI mode")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of GPU")
    parser.add_argument("-f", "--show_fps", action="store_true", help="Show FPS in the viewer")
    parser.add_argument("-e", "--n_episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("-a", "--n_actions", type=int, default=1, help="Number of actions per episode")
    args = parser.parse_args()


    gripper_env = GripperEnv(args=args)
    gripper_env.run()