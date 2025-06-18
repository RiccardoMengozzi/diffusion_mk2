import os
import argparse
import numpy as np
import genesis as gs
import time
import json
from pynput import keyboard
from genesis.engine.entities import RigidEntity, MPMEntity
from genesis.engine.entities.rigid_entity import RigidLink
from diffusion_mk2.utils.dlo_shapes import U_SHAPE, S_SHAPE
import diffusion_mk2.utils.dlo_computations as dlo_utils

from rich.console import Console
from rich.live import Live
from rich.text import Text
from rich.panel import Panel
from rich.layout import Layout
from rich.align import Align
from rich.box import ROUNDED
from rich.prompt import Prompt


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
EE_OFFSET = 0.116
EE_QUAT_ROTATION = np.array([0, 0, -1, 0])
ROPE_LENGTH = 0.2
ROPE_RADIUS = 0.003
ROPE_BASE_POSITION = np.array([0.5, 0.0, HEIGHT_OFFSET + ROPE_RADIUS])
NUMBER_OF_PARTICLES = 15
PARTICLES_NUMBER_FOR_POS_SMOOTHING = 10


EE_VELOCITY = 0.015
SAVE_DATA_INTERVAL = 3 # every 3 steps


class JSONLDataLogger:
    """Handles JSONL file operations for streaming data"""
    
    def __init__(self, save_path, save_name):
        self.save_dir = save_path
        self.save_name = save_name
        self.filename = os.path.join(self.save_dir, self.save_name)

        # If folder does not exist, create it
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

        # If the file already exist, append data to it
        last_episode_idx = 0
        if os.path.exists(self.filename):
            with open(self.filename, "r") as f:
                for line in f:
                    data = json.loads(line)
                    if data.get("type") == "episode_end":
                        last_episode_idx = data.get("episode_idx", last_episode_idx)

        self.total_steps = last_episode_idx       
        self.episode_data = []
        self.file = None
        
    def initialize_file(self):
        """Initialize JSONL file"""
        self.file = open(self.filename, 'a')
        
    
    
    def append_data(self, observation, action):
        """Append observation and action to file"""
        data = {
            "type": "data",
            "observation": observation.tolist(),  # Convert numpy array to list
            "action": action.tolist()
        }
        self.episode_data.append(data)

    def delete_episode_data(self):
        self.episode_data = []  # Clear the episode data after saving


    def save_episode(self):
        """Mark the end of an episode and save all data"""

        # Save episode data
        for data in self.episode_data:
            self.file.write(json.dumps(data) + '\n')
            self.total_steps += 1

        # Save episode end idx
        episode_end = {
            "type": "episode_end",
            "episode_idx": self.total_steps,
        }
        self.file.write(json.dumps(episode_end) + '\n')
        self.file.flush()
        self.delete_episode_data()  # Clear the episode data after saving



    
    def close(self):
        """Close the file"""
        if self.file:
            self.file.close()
            print(f"Closed JSONL file '{self.save_name}.jsonl' with {self.total_steps} total steps")




class Monitor():
    def __init__(self):
        self.status_message = None
        self.status_active = False
    
    def get_layout(self):
        """Define the layout for the Rich Live display."""
        layout = Layout(name="root")
        

        layout.split(
            Layout(name="header", size=3),
            Layout(name="status", size=3), # New layout for status
            Layout(name="main", ratio=1),
            Layout(name="footer", size=5)
        )
        layout["main"].split_row(
            Layout(name="controls"),
            Layout(name="info")
        )

        return layout

    def make_header_panel(self):
        return Panel(
            Align.center(
                Text(f"Teleoperation Data Generator", justify="center", style="bold yellow"),
                vertical="middle"
            ),
            title="[bold blue]DLO Pushin Teleop[/bold blue]",
            border_style="blue",
            box=ROUNDED
        )

    def make_controls_panel(self):
        controls_text = Text()
        controls_text.append("Movement Controls:\n", style="bold underline")
        controls_text.append("  W: Move End-Effector -X (forward)\n")
        controls_text.append("  S: Move End-Effector +X (backward)\n")
        controls_text.append("  A: Move End-Effector -Y (left)\n")
        controls_text.append("  D: Move End-Effector +Y (right)\n\n")
        controls_text.append("Episode Controls:\n", style="bold underline")
        controls_text.append("  SPACE: Save current episode data and start next episode\n")
        controls_text.append("  BACKSPACE: Skip current episode (data will NOT be saved)\n", style="red")

        return Panel(
            controls_text,
            title="[bold green]Controls[/bold green]",
            border_style="green",
            box=ROUNDED
        )

    def make_info_panel(self, 
                        current_episode, 
                        total_episodes, 
                        current_step, 
                        current_key_pressed, 
                        end_effector: RigidLink,
                        real_time_factor: float):
        info_text = Text()
        info_text.append("Real-time factor: ", style="none")
        info_text.append(f"{real_time_factor:.3f}x\n", style="bold cyan")
        info_text.append("Current Episode: ", style="none")
        info_text.append(f"{current_episode}/{total_episodes}\n", style="bold cyan")
        info_text.append("Current Step in Episode: ", style="none")
        info_text.append(f"{current_step}\n", style="bold magenta")
        info_text.append("Key Pressed: ", style="none")
        info_text.append(f"{current_key_pressed}\n", style="bold yellow")
        info_text.append("End-effector X: ", style="none")
        info_text.append(f"{end_effector.get_pos().cpu().numpy()[0]:.4f}\n", style="bold white")
        info_text.append("End-effector Y: ", style="none")
        info_text.append(f"{end_effector.get_pos().cpu().numpy()[1]:.4f}\n", style="bold white")


        return Panel(
            info_text,
            title="[bold blue]Information[/bold blue]",
            border_style="blue",
            box=ROUNDED
        )

    def make_footer_panel(self):
        return Panel(
            Align.center(
                Text("Press ESC or close the window to exit.", justify="center", style="dim white"),
                vertical="middle"
            ),
            border_style="white",
            box=ROUNDED
        )

    def make_status_panel(self, 
                          resetting : bool, 
                          saving : bool, 
                          current_episode: int = 0,
                          current_step: int = 0,
                          style: str = "bold yellow"):
        if resetting:
            message = "Resetting Episode..."
        elif saving:
            message = f"Saving Episode {current_episode} with {current_step} steps..."
        else:
            message = "Teleoperation in progress..."
        return Panel(
            Align.center(
                Text(message, justify="center", style=style),
                vertical="middle"
            ),
            title="[bold red]STATUS[/bold red]",
            border_style="red",
            box=ROUNDED
        )



class TeleopPushDataGenerator():
    def __init__(self, gui=False, cpu=False, show_fps=False, save_name="dummy"):
        self.gui = gui
        self.cpu = cpu
        self.show_fps = show_fps

        save_path = os.path.join(PROJECT_FOLDER, "json_data")
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        self.data_logger = JSONLDataLogger(save_path, save_name + ".jsonl")
        self.data_logger.initialize_file()

        self.monitor = Monitor()

        self.console = Console()
        self.current_key_pressed = "None"
        self.current_episode = 0
        self.total_episodes = 0
        self.real_time_factor = 0.0


        # Add flag for Enter key detection
        self.spacebar_pressed = False
        self.backspace_pressed = False
        self.esc_pressed = False

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
            show_viewer=False,
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
                # friction=2.0,
                # needs_coup=True,
                # coup_friction=2.0,
                # sdf_cell_size=0.005,
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

        self.initial_pose = np.array([0.45, 0.0, HEIGHT_OFFSET + EE_OFFSET, 0.0, 0.707, 0.707, 0.0])

        self.shapes = {
            "U": U_SHAPE,
            "S": S_SHAPE,
        }

        self.delta = np.zeros(3)  # Initialize delta for movement
        self.previous_ee_position = np.zeros(2)
        self.accumulated_ee_displacement = 0.0

        self.episode_observations = []
        self.episode_actions = []


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
                    self.esc_pressed = True
                elif key == keyboard.Key.space:
                    self.spacebar_pressed = True
                    self.current_key_pressed = "SPACE (Save & Next Episode)"
                elif key == keyboard.Key.backspace:
                    self.backspace_pressed = True
                    self.current_key_pressed = "BACKSPACE (Skip Episode)"
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
                    self.spacebar_pressed = False
                if key == keyboard.Key.backspace:
                    self.backspace_pressed = False
                if key.char == 's':
                    self.delta[0] = 0.0
                if key.char == 'a':
                    self.delta[1] = 0.0
                if key.char == 'd':
                    self.delta[1] = 0.0
                if key.char == 'w':
                    self.delta[0] = 0.0
                # Only reset current_key_pressed if no movement keys are held
                if all(d == 0.0 for d in self.delta[:2]):
                    self.current_key_pressed = "None"
            except AttributeError:
                if all(d == 0.0 for d in self.delta[:2]):
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
        qpos[-2:] = 0.0  # Set fingers to closed position

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

        # Save current position as previous position
        self.previous_ee_position = np.array([pose[0], pose[1]])





    def move(self):
        """Move the robot by an x,y offset."""
        current_pos = self.end_effector.get_pos().cpu().numpy()
        # Keep the Z position constant, otherwise it can drift...
        current_pos[2] = HEIGHT_OFFSET + EE_OFFSET
        target_pos = current_pos + self.delta
        qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=target_pos,
            quat=[0.0, 0.707, 0.707],
        )
        qpos[-2:] = 0.0
        self.franka.control_dofs_position(
            qpos,
            [*self.motors_dof, *self.fingers_dof],
        )
        self._step()


    def get_observation(self):
        obs_ee = self.end_effector.get_pos().cpu().numpy()[:2]
        # self.scene.draw_debug_sphere(
        #     pos=np.array([obs_ee[0], obs_ee[1], HEIGHT_OFFSET + ROPE_RADIUS/2]),
        #     radius=0.001,
        #     color=(1, 0, 0, 1),  # Red color for end effector
        # )
        obs_dlo = dlo_utils.get_skeleton(self.rope.get_particles(),
                                            downsample_number=NUMBER_OF_PARTICLES,
                                            average_number=PARTICLES_NUMBER_FOR_POS_SMOOTHING)[:, :2]
        obs_target = self.target_shape[:, :2]
        return np.vstack([obs_ee, obs_dlo, obs_target])

    def get_action(self):
        action = self.end_effector.get_pos().cpu().numpy()[:2]
        return action

 
    def run_episode(self, episode, total_episodes):
        self.current_episode = episode + 1
        self.total_episodes = total_episodes
        self.spacebar_pressed = False
        self.backspace_pressed = False
        self.current_key_pressed = "None"
        
        current_step = 0
        loop_counter = 0

        resetting = False

    
        while not self.spacebar_pressed:
            resetting = False
            if self.esc_pressed:
                # Stop the listener and exit
                self.listener.stop()
                break
            
            if self.backspace_pressed:
                resetting = True
                # Reset the robot to initial pose after skipping (will show its own reset status)
                self.reset(self.initial_pose)
                # Clear the episode data for this skipped episode
                self.data_logger.delete_episode_data()  
                # Reset the step counter
                current_step = 0



            is_moving = self.delta[0] != 0.0 or self.delta[1] != 0.0
            loop_counter += 1

            if loop_counter % SAVE_DATA_INTERVAL == 0 and is_moving:
                current_step += 1
                observation = self.get_observation()

            self.move()

            if loop_counter % SAVE_DATA_INTERVAL == 0 and is_moving:
                action = self.get_action()
                self.data_logger.append_data(observation, action)

            # Update Rich layout
            layout = self.monitor.get_layout()
            layout["header"].update(self.monitor.make_header_panel())
            layout["controls"].update(self.monitor.make_controls_panel())
            layout["status"].update(
                self.monitor.make_status_panel(resetting=resetting, saving=False,
                                                current_episode=self.current_episode,
                                                current_step=current_step,
                                                style="bold green")
            )
            layout["info"].update(self.monitor.make_info_panel(self.current_episode, 
                                                                self.total_episodes,
                                                                current_step,
                                                                self.current_key_pressed,
                                                                self.end_effector,
                                                                self.real_time_factor))
            layout["footer"].update(self.monitor.make_footer_panel())
            
            self.live.update(layout)
            if resetting:
                # just to show for a while the status message
                time.sleep(1)


        # Show saving message
        layout["status"].update(
            self.monitor.make_status_panel(resetting=False, saving=True,
                                            current_episode=self.current_episode,
                                            current_step=current_step,
                                            style="bold green")
        )
        self.live.update(layout)
        time.sleep(1)  # Give some time to show the saving status


    def run(self, n_episodes):
        """Run the entire teleoperation data generation process."""
        with Live(self.monitor.get_layout(), screen=True, refresh_per_second=10) as self.live:
            try:
                for i in range(n_episodes):
                    # Reset is called here, and it will now show the "Resetting Episode..." status
                    self.reset(self.initial_pose) 
                    
                    self.run_episode(i, n_episodes)
                    
                    if self.esc_pressed:
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
    parser.add_argument("-g", "--gui", action="store_true", help="Enable GUI mode")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of GPU")
    parser.add_argument("-f", "--show_fps", action="store_true", help="Show FPS in the viewer")
    parser.add_argument("-e", "--n_episodes", type=int, default=NUMBER_OF_EPISODES, help="Number of episodes to run")
    parser.add_argument("-n", "--save_name", type=str, default="dummy", help="save name")
    args = parser.parse_args()

    generator = TeleopPushDataGenerator(gui=args.gui, cpu=args.cpu, show_fps=args.show_fps, save_name=args.save_name)
    generator.run(n_episodes=args.n_episodes)