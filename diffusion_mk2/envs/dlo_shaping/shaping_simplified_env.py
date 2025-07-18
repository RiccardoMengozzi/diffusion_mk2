import os
import genesis as gs
import numpy as np
import argparse
from tqdm import tqdm
import time
import collections
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from genesis.engine.entities import RigidEntity, MPMEntity
from genesis.engine.entities.rigid_entity import RigidLink
from diffusion_mk2.config.shaping_simplified_env_config import ShapingConfig
from diffusion_mk2.utils import dlo_computations
from diffusion_mk2.utils.dlo_shapes import ONE_ACTION_SHAPE2
from diffusion_mk2.inference.shaping_simplified_inference import ShapingInference

PROJECT_FOLDER = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


class ShapingSimplifiedEnv:
    def __init__(self, config: ShapingConfig):
        self.config = config
        gs.init(
            backend=gs.cpu if self.config.simulation.cpu else gs.gpu,
            logging_level="error",
        )
        ########################## create a scene ##########################
        self.scene: gs.Scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.config.simulation.dt,
                substeps=self.config.simulation.substeps,
            ),
            viewer_options=gs.options.ViewerOptions(
                res=self.config.viewer.resolution,
                camera_pos=self.config.viewer.position,
                camera_lookat=self.config.viewer.lookat,
                camera_fov=self.config.viewer.fov,
                refresh_rate=self.config.viewer.refresh_rate,
                max_FPS=self.config.viewer.max_fps,
            ),
            vis_options=gs.options.VisOptions(
                visualize_mpm_boundary=True,
                show_world_frame=True,
            ),
            mpm_options=gs.options.MPMOptions(
                lower_bound=self.config.dlo.mpm_lower_bound,
                upper_bound=self.config.dlo.mpm_upper_bound,
                grid_density=self.config.dlo.mpm_grid_density,
            ),
            show_FPS=self.config.simulation.show_fps,
            show_viewer=self.config.simulation.visualization,
        )

        self.cam = self.scene.add_camera(
            res    = self.config.simulation.camera.resolution,
            pos    = self.config.simulation.camera.position,
            lookat = self.config.simulation.camera.lookat,
            fov    = self.config.simulation.camera.fov,
            GUI    = self.config.simulation.camera.gui
        )

        ########################## entities ##########################
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )

        self.table = self.scene.add_entity(
            morph=gs.morphs.URDF(
                file=os.path.join(
                    PROJECT_FOLDER, "models/SimpleTable/SimpleTable.urdf"
                ),
                pos=self.config.table.position,
                euler=self.config.table.orientation,
                scale=self.config.table.scale,
                fixed=True,
            ),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Default(),
        )

        self.dlo: MPMEntity = self.scene.add_entity(
            material=gs.materials.MPM.Elastic(
                E=self.config.dlo.E,  # Determines the squishiness of the rope (very low values act as a sponge)
                nu=self.config.dlo.nu,
                rho=self.config.dlo.rho,
                sampler=self.config.dlo.sampler,
            ),
            morph=gs.morphs.Cylinder(
                height=self.config.dlo.length,
                radius=self.config.dlo.radius,
                pos=self.config.dlo.position,
                euler=self.config.dlo.orientation,
            ),
            surface=gs.surfaces.Default(roughness=2, vis_mode="particle"),
        )
        self.franka: RigidEntity = self.scene.add_entity(
            gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda.xml",
                pos=self.config.franka.position,
                euler=self.config.franka.orientation,
            ),
            material=gs.materials.Rigid(
                friction=self.config.franka.end_effector.friction,
                needs_coup=self.config.franka.end_effector.needs_coup,
                coup_friction=self.config.franka.end_effector.coup_friction,
                sdf_cell_size=self.config.franka.end_effector.sdf_cell_size,
                gravity_compensation=self.config.franka.gravity_compensation,
            ),
        )

        ########################## build ##########################
        self.scene.build()

        self.motors_dof = np.arange(7)
        self.fingers_dof = np.arange(7, 9)
        self.end_effector: RigidLink = self.franka.get_link("hand")

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
        self.dlo_init_pos = self.dlo.get_particles()
        self.initial_pose = np.array(
            [
                0.45,
                0.0,
                self.config.table.height
                + self.config.franka.end_effector.offset
                + self.config.franka.end_effector.z_lift,
                0.0,
                0.707,
                0.707,
                0.0,
            ]
        )
        self.model = ShapingInference(ckp_path=self.config.inference.model_path,
                                 device="cuda" if not self.config.simulation.cpu else "cpu",
                                 num_timesteps=10)

        self.obs_horizon = self.model.obs_horizon

        # Initialize observation buffer
        self.target = np.array(ONE_ACTION_SHAPE2)
        obs = self.get_obs()
        self.obs_deque = collections.deque([obs] * self.obs_horizon, maxlen=self.obs_horizon)

        self.ready_to_plot = False
        self.current_pred_action = None

    def plot(self):
        plt.ion()  # Enable interactive mode

        if not hasattr(self, "_fig") or self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
        else:
            self._ax.clear()

        current_dlo_shape = dlo_computations.get_skeleton(
            self.dlo.get_particles(),
            downsample_number=self.config.dlo.number_of_particles,
            average_number=self.config.dlo.particles_smoothing,
        )
        target_shape = self.target[:, :2]
        pred_action = self.current_pred_action[:, 1:3]

        self._ax.plot(current_dlo_shape[:, 0], current_dlo_shape[:, 1], "o-", label="DLO", linewidth=2, markersize=4)
        self._ax.plot(target_shape[:, 0], target_shape[:, 1], "o-", label="Target", linewidth=2, markersize=4)

        if pred_action is not None:
            pred_pt = []
            pt = self.end_effector.get_pos().cpu().numpy()[:2]
            for delta in pred_action:
                pt += delta
                pred_pt.append(pt.copy())
            pred_pt = np.array(pred_pt)
            self._ax.plot(pred_pt[:, 0], pred_pt[:, 1], "^-", label="Predicted Action", linewidth=2, markersize=4)

        self._ax.set_xlabel("X")
        self._ax.set_ylabel("Y")
        self._ax.set_title("DLO vs Target in XY Plane")
        self._ax.legend()
        self._ax.axis("equal")
        self._ax.grid(True)

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()


    def _step(self):
        start_time = time.time()

        self.scene.step()
        end_time = time.time()
        self.real_time_factor = self.config.simulation.dt / (end_time - start_time)

        if self.config.simulation.camera.record:
            self.cam.render()

        if self.config.inference.plot and self.ready_to_plot:
            self.plot()


        if self.config.simulation.show_real_time_factor:
            print(f"Real-time factor: {self.real_time_factor:.2f}")


    def reset_robot_pose(self):
        # Place robot above centre of the rope
        skeleton = dlo_computations.get_skeleton(
            self.dlo.get_particles(),
            downsample_number=self.config.dlo.number_of_particles,
            average_number=self.config.dlo.particles_smoothing,
        )
        skeleton_centre = np.array([np.mean(skeleton[:, 0]), np.mean(skeleton[:, 1])])

        target_pos = [
            skeleton_centre[0],
            skeleton_centre[1],
            self.config.table.height
            + self.config.franka.end_effector.offset
            + self.config.franka.end_effector.z_lift,
        ]
        target_quat = self.initial_pose[3:7]  # Use the initial pose's quaternion

        qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=target_pos,
            quat=target_quat,
        )
        qpos[-2:] = (
            self.config.franka.end_effector.gripper_open_position
        )  # Open gripper at the start

        self.franka.set_qpos(qpos)


    def reset_dlo_pose(self):
        self.dlo.set_pos(self.dlo._sim.cur_substep_local, self.dlo_init_pos)


    def reset_episode(self):
        """Reset the environment for a new episode."""
        self.scene.clear_debug_objects()

        # Choose new target
        self.target = np.array(ONE_ACTION_SHAPE2)
        dlo_computations.draw_skeleton(self.target, self.scene, self.config.dlo.radius)

        # Reset robot pose
        self.reset_robot_pose()

        # Reset DLO pose
        self.reset_dlo_pose()
       
        self._step()

    def reset_action(self):
        """Reset the environment for a new action."""
        self.scene.clear_debug_objects()

        # Redraw the target shape
        dlo_computations.draw_skeleton(self.target, self.scene, self.config.dlo.radius)

        # Reset robot pose
        self.reset_robot_pose()

        self._step()

    def get_obs(self):
        pos_ee = self.end_effector.get_pos().cpu().numpy()[:2]
        theta = R.from_quat(self.end_effector.get_quat().cpu().numpy()).as_euler('xyz')[0] # dont ask why [0], but it works
        obs_ee = np.array([pos_ee[0], pos_ee[1], theta])
        obs_dlo = dlo_computations.get_skeleton(self.dlo.get_particles(),
                                            downsample_number=self.config.dlo.number_of_particles,
                                            average_number=self.config.dlo.particles_smoothing)[:, :2]  # Only x and y coordinates of the DLO
        obs_target = self.target[:, :2]  # Only x and y coordinates of the target
        obs_ee = np.array(obs_ee).flatten()
        obs_dlo = np.array(obs_dlo).flatten()
        obs_target = np.array(obs_target).flatten()
        obs = np.concatenate([obs_ee, obs_dlo, obs_target]) 
        return obs

    def draw_trajectory(self, traj):
        """Draw the trajectory of the end-effector."""
        target_pos = self.end_effector.get_pos().cpu().numpy()
        target_pos[2] -= self.config.franka.end_effector.offset

        for i, action in enumerate(traj):
            # Red to blue gradient
            t = i / (len(action) - 1)
            color = [1.0 - t, 0.0, t, 1.0]

            dx, dy = action[1:3]

            target_pos += np.array([dx, dy, 0.0])

            self.scene.draw_debug_sphere(
                target_pos,
                color=color,
                radius=0.001,
            )


    def execute_action(self, target_pos, target_quat, gripper="open", path_period=1.0, tolerance=1e-7):
        qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=target_pos,
            quat=target_quat,
        )
        if gripper == "open":
            force_control = [0.0, 0.0]
            # qpos[-2:] = self.config.franka.end_effector.gripper_open_position
        elif gripper == "close":
            force_control = [-1.0, -1.0]
            # qpos[-2:] = self.config.franka.end_effector.gripper_close_position
        
        path = self.franka.plan_path(
            qpos_goal=qpos,
            num_waypoints=int(path_period // self.config.simulation.dt),
            ignore_collision=True, # Otherwise cannot grasp in a good way the rope
        )

        # Control the robot along the path
        for p in path:
            self.franka.control_dofs_position(p[:-2], self.motors_dof)
            self.franka.control_dofs_force(force_control, self.fingers_dof)

            self._step()

            # Check if the robot has reached the target position
            if np.linalg.norm(qpos.cpu().numpy() - self.franka.get_qpos().cpu().numpy()) < tolerance:
                break

    def execute_trajectory(self, traj):
        # Grasp
        pred_idx = int(traj[0, 0])
        skeleton = dlo_computations.get_skeleton(
            self.dlo.get_particles(),
            downsample_number=self.config.dlo.number_of_particles,
            average_number=self.config.dlo.particles_smoothing,
        )
        target_pos, target_quat = dlo_computations.compute_pose_from_paticle_index(
            skeleton,
            pred_idx,
            self.config.franka.end_effector.rot_offset,
            self.config.franka.end_effector.offset,
        )
        print("target_quat", target_quat)
        print("initial target_quat", self.initial_pose[3:7])
        self.execute_action(
            target_pos=target_pos,
            target_quat=target_quat,
            gripper="open",
            path_period=2.0,
        )

        self.execute_action(
            target_pos=target_pos,
            target_quat=target_quat,
            gripper="close",
            path_period=1.0,
        )

        # Execute action
        for action in traj:
            dx, dy, dt = action[1:4]

            current_pos = self.end_effector.get_pos().cpu().numpy()
            current_quat = self.end_effector.get_quat().cpu().numpy()

            current_R = (R.from_quat(current_quat)).as_matrix()
            delta_R = R.from_euler("xyz", [dt, 0.0, 0.0]).as_matrix()
            target_R = current_R @ delta_R

            target_pos = current_pos + np.array([dx, dy, 0.0])
            target_quat = (R.from_matrix(target_R)).as_quat()

            self.execute_action(
                target_pos=target_pos,
                target_quat=target_quat,
                gripper="close",
                path_period=0.5,
            )

        # Release

        self.execute_action(
            target_pos=target_pos,
            target_quat=target_quat,
            gripper="open",
            path_period=1.0,
        )


    def run(self):
        if self.config.simulation.camera.record:
            self.cam.start_recording()

        for _ in tqdm(range(self.config.inference.n_episodes), desc="Episodes"):
            self.reset_episode()
            for _ in tqdm(range(self.config.inference.n_actions), desc="Actions"):
                self.reset_action()
                self.ready_to_plot = True
                obs = self.get_obs()

                self.obs_deque.append(obs)
                obs = np.stack(self.obs_deque)
                obs = obs.reshape(self.model.obs_horizon, -1)

                pred_action, pred_actions = self.model.run_inference(
                    observation=obs,
                )
                self.current_pred_action = pred_action
                self.draw_trajectory(pred_action)
                self.execute_trajectory(pred_action)

        if self.config.simulation.camera.record:
            self.cam.stop_recording(save_to_filename='video.mp4', fps=60)
        if self.config.inference.plot:
            plt.ioff()
            plt.close("all")

def main():
    parser = argparse.ArgumentParser(description="Teleop Push Data Generator")
    parser.add_argument(
        "--cfg",
        type=str,
        default="diffusion_mk2/config/shaping_simplified_env_config.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument("-v", "--vis", action="store_true")
    parser.add_argument("-g", "--gui", action="store_true", help="Enable GUI mode")
    parser.add_argument(
        "-c", "--cpu", action="store_true", help="Run on CPU instead of GPU"
    )
    parser.add_argument(
        "-f", "--show_fps", action="store_true", help="Show FPS in the viewer"
    )
    parser.add_argument(
        "-e", "--n_episodes", type=int, default=1, help="Number of episodes to run"
    )
    parser.add_argument(
        "-a", "--n_actions", type=int, default=1, help="Number of actions per episode"
    )
    parser.add_argument(
        "-r", "--record", action="store_true", help="Record the simulation"
    )
    parser.add_argument(
        "-p", "--plot", action="store_true", help="Plot the results"
    )
    args = parser.parse_args()

    config = ShapingConfig.from_yaml_and_args(args.cfg, args)
    env = ShapingSimplifiedEnv(config)
    env.run()


if __name__ == "__main__":
    main()
