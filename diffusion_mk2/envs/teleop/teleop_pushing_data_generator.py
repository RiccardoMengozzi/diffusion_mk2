import os
import argparse
import numpy as np
import genesis as gs
import time
import colorsys
from pynput import keyboard
from genesis.engine.entities import RigidEntity, MPMEntity
from genesis.engine.entities.rigid_entity import RigidLink
from diffusion_mk2.utils.dlo_shapes import U_SHAPE, S_SHAPE
import diffusion_mk2.utils.dlo_computations as dlo_utils
from diffusion_mk2.envs.teleop.data_logger import JSONLDataLogger
from diffusion_mk2.envs.teleop.monitor import Monitor
from scipy.spatial.transform import Rotation as R

from rich.live import Live




PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
NUMBER_OF_EPISODES = 3
NUMBER_OF_ACTIONS_PER_EPISODE = 8 # This is not directly used in the teleop script, but kept for consistency
ACTION_TIME = 2.0  # seconds (for 256 batch size)
VELOCITY = 0.05  # m/s
DT = 1e-2  # simulation time step
MPM_GRID_DENSITY = 256
SUBSTEPS = 40
TABLE_HEIGHT = 0.7005
HEIGHT_OFFSET = TABLE_HEIGHT
EE_OFFSET = 0.108
EE_Z = 0.04
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




class TeleopPushDataGenerator():
    def __init__(self, vis=False, gui=False, cpu=False, show_fps=False, save_name="dummy"):
        self.vis = vis
        self.gui = gui
        self.cpu = cpu
        self.show_fps = show_fps

        save_path = os.path.join(PROJECT_FOLDER, "json_data")
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        self.data_logger = JSONLDataLogger(save_path, save_name + ".jsonl")
        self.data_logger.initialize_file()

        self.monitor = Monitor()

        self.current_key_pressed = "None"
        self.current_episode = 0
        self.total_episodes = 0
        self.real_time_factor = 0.0

        self.step_counter = 0

        # Add flag for Enter key detection
        self.grasp_command = False
        self.save_command = False
        self.reset_command = False
        self.exit_command = False
        self.gripper_close_command = False

        self.grasping = False

        gs.init(
            backend=gs.cpu if self.cpu else gs.gpu,
            logging_level="warning",
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

        self.cam_front = self.scene.add_camera(
            res=(900, 1080),
            pos=(1.0, 0.0, TABLE_HEIGHT + 0.1),
            lookat=(0.5, 0.0, TABLE_HEIGHT),
            fov=70,
            GUI=self.gui,
        )
        self.cam_side = self.scene.add_camera(
            res=(900, 1080),
            pos=(0.5, -1.0, TABLE_HEIGHT + 0.1),
            lookat=(0.5, 0.0, TABLE_HEIGHT),
            fov=70,
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

        self.shapes = {
            "U": U_SHAPE,
            "S": S_SHAPE,
        }

        self.delta = np.zeros(4)  # Initialize delta for movement and rotation

        self.episode_observations = []
        self.episode_actions = []
        self.previous_qpos = np.zeros(9)  # Initialize previous qpos for the robot

        # Setup keyboard listener
        self.setup_keyboard_listener()


    def _step(self):
        start_time = time.time()

        self.scene.step()
        self.cam_front.render()
        self.cam_side.render()

        end_time = time.time()
        self.real_time_factor = DT / (end_time - start_time)


    def setup_keyboard_listener(self):
        """Setup pynput keyboard listener"""
        def on_press(key):
            try:
                if key == keyboard.Key.esc:
                    # Stop the listener and exit
                    self.exit_command = True
                elif key == keyboard.Key.enter:
                    self.save_command = True
                    self.current_key_pressed = "ENTER (Save & Next Episode)"
                elif key == keyboard.Key.backspace:
                    self.reset_command = True
                    self.current_key_pressed = "BACKSPACE (Skip Episode)"
                elif key == keyboard.Key.space:
                    self.grasp_command = True
                    self.current_key_pressed = "SPACEBAR (Grasp)"
                elif key == keyboard.Key.left:
                    self.delta[3] = -EE_ANG_VELOCITY
                    self.current_key_pressed = "Q (Rotate -Z)"
                elif key == keyboard.Key.right:
                    self.delta[3] = EE_ANG_VELOCITY
                    self.current_key_pressed = "E (Move +Z)"
                elif key == keyboard.Key.up:
                    self.delta[2] = EE_VELOCITY
                    self.current_key_pressed = "UP (Move Up)"
                elif key == keyboard.Key.down:
                    z = self.end_effector.get_pos().cpu().numpy()[2]
                    self.delta[2] = -EE_VELOCITY/2 if z > HEIGHT_OFFSET + EE_OFFSET + 0.003 else 0.0
                    self.current_key_pressed = "DOWN (Move Down)"
                elif key.char == 'g':
                    self.gripper_close_command = not self.gripper_close_command
                    self.current_key_pressed = "G (Toggle Gripper Open)"
                elif key.char == 's':
                    self.delta[0] = EE_VELOCITY
                    self.current_key_pressed = "S (Move +X)"
                elif key.char == 'a':
                    self.delta[1] = -EE_VELOCITY
                    self.current_key_pressed = "A (Move -Y)"
                elif key.char == 'd':
                    self.delta[1] = EE_VELOCITY
                    self.current_key_pressed = "D (Move +Y)"
                elif key.char == 'w':
                    self.delta[0] = -EE_VELOCITY
                    self.current_key_pressed = "W (Move -X)"
                else:
                    self.current_key_pressed = f"'{key.char}'"
            except AttributeError:
                self.current_key_pressed = f"Special Key {key}"

        def on_release(key):
            try:
                if key == keyboard.Key.space:
                    self.save_command = False
                if key == keyboard.Key.enter:
                    self.grasp_command = False
                if key == keyboard.Key.backspace:
                    self.reset_command = False
                if key == keyboard.Key.left:
                    self.delta[3] = 0.0
                if key == keyboard.Key.right:
                    self.delta[3] = 0.0
                if key == keyboard.Key.up:
                    self.delta[2] = 0.0
                if key == keyboard.Key.down:
                    self.delta[2] = 0.0
                if key.char == 's':
                    self.delta[0] = 0.0
                if key.char == 'a':
                    self.delta[1] = 0.0
                if key.char == 'd':
                    self.delta[1] = 0.0
                if key.char == 'w':
                    self.delta[0] = 0.0
                if key.char == 'q':
                    self.delta[3] = 0.0
                if key.char == 'e':
                    self.delta[3] = 0.0
                # Only reset current_key_pressed if no movement keys are held
                if all(d == 0.0 for d in self.delta[:3]):
                    self.current_key_pressed = "None"
            except AttributeError:
                if all(d == 0.0 for d in self.delta[:3]):
                    self.current_key_pressed = "None"


        # Start the listener in non-blocking mode
        self.listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release
        )
        self.listener.start()

    def reset(self, pose):
        """Reset the environment for a new episode."""
        # Ensure the status update is rendered before proceeding
        # This will be handled by the Live context in run_episode, but for standalone reset calls:
        # self.console.print(self.monitor.get_layout()) # uncomment if you call reset outside Live context

        self.scene.clear_debug_objects()

        # Reset franka position
        qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=pose[:3],
            quat=pose[3:],
        )
        qpos[-2:] = OPEN_GRIPPER_POSITION  # Set fingers to closed position

        self.franka.set_qpos(qpos)
        self._step()


        # Reset the target rope state
        shape_key = np.random.choice(list(self.shapes.keys()))
        self.target_shape = self.shapes[shape_key] #choose random index from the shapes
        # Center the shape at the origin
        origin_offset = np.mean(self.target_shape, axis=0)
        self.target_shape -= origin_offset
        # Randomly rotate the target shape around the Z-axis
        angle = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        self.target_shape = np.dot(self.target_shape, rotation_matrix.T)
        # Reapply the origin offset to the rotated shape
        self.target_shape += origin_offset
        # Randomly offset the target shape in the XY plane (no Z offset)
        xy_offset = np.zeros(3)
        xy_offset[:2] = np.random.uniform(-0.1, 0.1, size=2)
        self.target_shape += xy_offset
        dlo_utils.draw_skeleton(self.target_shape, self.scene, ROPE_RADIUS)
        # Update camera position to focus on the new target shape
        rope_center_xy = origin_offset + xy_offset
        cam_front_pos = np.array([rope_center_xy[0] + 0.3,
                            rope_center_xy[1],
                            HEIGHT_OFFSET + 0.2])
        cam_side_pos = np.array([rope_center_xy[0],
                            rope_center_xy[1] - 0.3,
                            HEIGHT_OFFSET + 0.2])
        lookat_front = np.array([rope_center_xy[0],
                            rope_center_xy[1],
                            HEIGHT_OFFSET])
        lookat_side = np.array([rope_center_xy[0],
                            rope_center_xy[1],
                            HEIGHT_OFFSET])
        self.cam_front.set_pose(pos=cam_front_pos, lookat=lookat_front)
        self.cam_side.set_pose(pos=cam_side_pos, lookat=lookat_side)

        # Reset grasping state
        self.grasping = False

        self.previous_qpos = self.franka.get_qpos().cpu().numpy()   



    def get_observation(self):
        pos_ee = self.end_effector.get_pos().cpu().numpy()
        theta = R.from_quat(self.end_effector.get_quat().cpu().numpy()).as_euler('xyz')[0] # dont ask why [0], but it works
        finger_qpos = self.franka.get_qpos().cpu().numpy()[-1]
        obs_ee = np.array([pos_ee[0], pos_ee[1], pos_ee[2], theta, finger_qpos])

        self.scene.draw_debug_sphere(
            pos=np.array([obs_ee[0], obs_ee[1], obs_ee[2] - EE_OFFSET]),
            radius=0.001,
            color=(1, 0, 0, 1),  # Red color for end effector
        )
        obs_dlo = dlo_utils.get_skeleton(self.rope.get_particles(),
                                            downsample_number=NUMBER_OF_PARTICLES,
                                            average_number=PARTICLES_NUMBER_FOR_POS_SMOOTHING)
        obs_target = self.target_shape
        return obs_ee, obs_dlo, obs_target

    def get_action(self):
        pos_ee = self.end_effector.get_pos().cpu().numpy()
        theta = R.from_quat(self.end_effector.get_quat().cpu().numpy()).as_euler('xyz')[0]
        finger_qpos = self.franka.get_qpos().cpu().numpy()[-1]

        action = np.array([pos_ee[0], pos_ee[1], pos_ee[2], theta, finger_qpos])
        return action

 
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
            ### REASONS TO EXIT ###
            if self.exit_command or self.reset_command:
                return

            movement = np.linalg.norm(self.franka.get_qpos().cpu().numpy() - self.previous_qpos)
            self.previous_qpos = self.franka.get_qpos().cpu().numpy()
            is_moving = movement > 5e-4

            # If we are saving data, get the observation
            if self.step_counter % SAVE_DATA_INTERVAL == 0 and is_moving:
                obs_ee, obs_dlo, obs_target = self.get_observation()

            self.franka.control_dofs_position(p[:-2], self.motors_dof)
            if force_control:
                self.franka.control_dofs_force([-force_intensity, -force_intensity], self.fingers_dof)
            else:
                self.franka.control_dofs_position(p[-2:], self.fingers_dof)
            self._step()

            # If we are saving data, get the action and append data
            if self.step_counter % SAVE_DATA_INTERVAL == 0 and is_moving:
                action = self.get_action()
                self.data_logger.append_data(obs_ee, obs_dlo, obs_target, action)
            
            # Update global step counter
            self.step_counter += 1

            # Check if the robot has reached the target position
            if np.linalg.norm(qpos.cpu().numpy() - self.franka.get_qpos().cpu().numpy()) < tolerance:
                break


    def control(self):
        current_pos = self.end_effector.get_pos().cpu().numpy()
        current_quat = self.end_effector.get_quat().cpu().numpy()
        target_pos = current_pos + self.delta[:3]

        current_R = (R.from_quat(current_quat)).as_matrix()
        delta_R = R.from_euler("xyz", [self.delta[3], 0.0, 0.0]).as_matrix()
        target_R = current_R @ delta_R

        target_quat = (R.from_matrix(target_R)).as_quat()
        gripper_open = False if self.gripper_close_command else not self.grasping

        # Move to the target position
        self.move(
            target_pos=target_pos,
            target_quat=target_quat,
            path_period=DT, # only one step
            gripper_open=gripper_open,
        )


    def grasp(self):
        self.grasping = True
        self.gripper_close_command = False # Override the gripper close command

        current_ee_pos = self.end_effector.get_pos().cpu().numpy()
        particles = dlo_utils.get_skeleton(self.rope.get_particles(),
                                            downsample_number=NUMBER_OF_PARTICLES,
                                            average_number=PARTICLES_NUMBER_FOR_POS_SMOOTHING)
        # Find the closest DLO particle
        idx = dlo_utils.get_closest_particle_index(
            particles,
            current_ee_pos,
        )

        # Get the target pose based on the index of the particle
        target_pos, target_quat = dlo_utils.compute_pose_from_paticle_index(
            particles,
            idx,
            ee_quat_offset=EE_QUAT_ROTATION,
            ee_offset=EE_OFFSET,
        )

        # If gripper is closed, open it first
        qpos = self.franka.get_qpos()
        if qpos[-1] < CLOSE_GRIPPER_POSITION:
            print("Opening gripper before grasping...")
            qpos[-2:] = OPEN_GRIPPER_POSITION  # Open the gripper
            self.move(
                qpos=qpos,
                path_period=0.5,
                gripper_open=True,  # Open the gripper
            )


        self.move(
            target_pos=target_pos,
            target_quat=target_quat,
            path_period=0.5,  
            gripper_open=True,  
        )

        # Close the gripper
        self.move(
            qpos=self.franka.get_qpos(),
            path_period=DT,
            gripper_open=False,  # Close the gripper
            force_control=True,  # Use force control to grasp
        )

    def release(self):
        """Release the grasped object."""
        self.grasping = False
        self.gripper_close_command = False # Override the gripper close command

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



    def show_grasping_point(self):
        self.scene.clear_debug_objects()
        dlo_utils.draw_skeleton(self.target_shape, self.scene, ROPE_RADIUS)
        
        current_ee_pos = self.end_effector.get_pos().cpu().numpy()
        particles = dlo_utils.get_skeleton(self.rope.get_particles(),
                                            downsample_number=NUMBER_OF_PARTICLES,
                                            average_number=PARTICLES_NUMBER_FOR_POS_SMOOTHING)
        # Find the closest DLO particle
        idx = dlo_utils.get_closest_particle_index(
            particles,
            current_ee_pos,
        )
        grasping_point = particles[idx]

        self.scene.draw_debug_sphere(
            pos=grasping_point,
            radius=ROPE_RADIUS + 0.001,
            color=(0.0, 0.0, 1.0, 1.0),
        )



    def run_episode(self, episode, total_episodes):
        self.current_episode = episode + 1
        self.total_episodes = total_episodes
        self.save_command = False
        self.reset_command = False
        self.current_key_pressed = "None"
        
        resetting = False

        while not self.save_command:
            resetting = False
            if self.exit_command:
                # Stop the listener and exit
                self.listener.stop()
                break
            
            if self.reset_command:
                resetting = True
                # Reset the robot to initial pose after skipping (will show its own reset status)
                self.reset(self.initial_pose)
                # Clear the episode data for this skipped episode
                self.data_logger.delete_episode_data()  
                # Reset the step counter


            # if not self.grasping:
            #     self.show_grasping_point()

            if self.grasp_command:
                self.grasp_command = False 
                if self.grasping:
                    self.release()
                else:
                    self.grasp()
            else:
                self.control()

            # Update Rich layout
            layout = self.monitor.get_layout()
            layout["header"].update(self.monitor.make_header_panel())
            layout["controls"].update(self.monitor.make_controls_panel())
            layout["status"].update(
                self.monitor.make_status_panel(resetting=resetting, saving=False,
                                                current_episode=self.current_episode,
                                                current_step=self.data_logger.episode_current_step,
                                                style="bold green")
            )
            layout["info"].update(self.monitor.make_info_panel(self.current_episode, 
                                                                self.total_episodes,
                                                                self.data_logger.episode_current_step,
                                                                self.current_key_pressed,
                                                                self.end_effector,
                                                                self.real_time_factor))
            layout["footer"].update(self.monitor.make_footer_panel())
            
            # self.live.update(layout)
            if resetting:
                # just to show for a while the status message
                time.sleep(1)


        # Show saving message
        layout["status"].update(
            self.monitor.make_status_panel(resetting=False, saving=True,
                                            current_episode=self.current_episode,
                                            current_step=self.data_logger.episode_current_step,
                                            style="bold green")
        )
        # self.live.update(layout)
        time.sleep(1)  # Give some time to show the saving status


    def run(self, n_episodes):
        """Run the entire teleoperation data generation process."""
        # with Live(self.monitor.get_layout(), screen=True, refresh_per_second=10) as self.live:
        try:
            for i in range(n_episodes):
                # Reset is called here, and it will now show the "Resetting Episode..." status
                self.reset(self.initial_pose) 
                
                self.run_episode(i, n_episodes)
                
                if self.exit_command:
                    # Exit if ESC is pressed without saving episode
                    print("Esc pressed, exiting...")
                    break
                
                # Save all episode data
                self.data_logger.save_episode()


        finally:
            # Clean up the keyboard listener
            if hasattr(self, 'listener') and self.listener.running:
                self.listener.stop()
            if hasattr(self, 'live'):
                self.live.stop()
            self.data_logger.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teleop Push Data Generator")
    parser.add_argument("-v", "--vis", action="store_true")
    parser.add_argument("-g", "--gui", action="store_true", help="Enable GUI mode")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of GPU")
    parser.add_argument("-f", "--show_fps", action="store_true", help="Show FPS in the viewer")
    parser.add_argument("-e", "--n_episodes", type=int, default=NUMBER_OF_EPISODES, help="Number of episodes to run")
    parser.add_argument("-n", "--save_name", type=str, default="dummy", help="save name")
    args = parser.parse_args()

    generator = TeleopPushDataGenerator(vis=args.vis, gui=args.gui, cpu=args.cpu, show_fps=args.show_fps, save_name=args.save_name)
    generator.run(n_episodes=args.n_episodes)