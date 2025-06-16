import os
import argparse
import numpy as np
import genesis as gs
import time
from pynput import keyboard
from genesis.engine.entities import RigidEntity, MPMEntity
from genesis.engine.entities.rigid_entity import RigidLink
from diffusion_mk2.utils.dlo_shapes import U_SHAPE, S_SHAPE
import diffusion_mk2.utils.dlo_computations as dlo_utils


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
EE_OFFSET = 0.116
EE_QUAT_ROTATION = np.array([0, 0, -1, 0])
ROPE_LENGTH = 0.2
ROPE_RADIUS = 0.003
ROPE_BASE_POSITION = np.array([0.5, 0.0, HEIGHT_OFFSET + ROPE_RADIUS])
NUMBER_OF_PARTICLES = 15
PARTICLES_NUMBER_FOR_POS_SMOOTHING = 10


EE_VELOCITY = 0.015 # move 1.5 cm every step
SAVE_DATA_DISPLACEMENT = 0.01 # every cm


class TeleopPushDataGenerator():
    def __init__(self, gui=False, cpu=False, show_fps=False, save_name="teleop_push_data.npz"):
        self.gui = gui
        self.cpu = cpu
        self.show_fps = show_fps
        self.npz_save_path = os.path.join(PROJECT_FOLDER, "npz_data", "teleop", save_name)
        
        # Add flag for Enter key detection
        self.enter_pressed = False
        self.backspace_pressed = False

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
            fov=50,
            GUI=self.gui,
        )
        self.cam_side = self.scene.add_camera(
            res=(900, 1080),
            pos=(0.5, -1.0, TABLE_HEIGHT + 0.1),
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
        self.scene.step()
        self.cam_front.render()
        self.cam_side.render()


    def setup_keyboard_listener(self):
        """Setup pynput keyboard listener"""
        def on_press(key):
            try:
                if key == keyboard.Key.enter:
                    self.enter_pressed = True
                if key == keyboard.Key.backspace:
                    self.backspace_pressed = True
                if key.char == 's':
                    self.delta[0] = EE_VELOCITY
                if key.char == 'a':
                    self.delta[1] = -EE_VELOCITY
                if key.char == 'd':
                    self.delta[1] = EE_VELOCITY
                if key.char == 'w':
                    self.delta[0] = -EE_VELOCITY
            except AttributeError:
                pass
        
        def on_release(key):
            # Optional: handle key releases
            try:
                if key == keyboard.Key.enter:
                    self.enter_pressed = False
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
            except AttributeError:
                pass
        
        # Start the listener in non-blocking mode
        self.listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release
        )
        self.listener.start()

    def reset(self, pose):
        ### Reset the environment for a new episode ###
        self.scene.clear_debug_objects()
        
        ### reset franka position ###
        qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=pose[:3],
            quat=pose[3:],
        )
        qpos[-2:] = 0.0  # Set fingers to closed position

        self.franka.set_qpos(qpos)
        self._step()


        ### reset the target rope state ###
        shape_key = np.random.choice(list(self.shapes.keys()))
        print(f"Chosen shape: {shape_key}")
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

        ## Save current position as previous position ##
        self.previous_ee_position = np.array([pose[0], pose[1]])


    def move(self):
        ### Move the robot by an x,y offset ###
        current_pos = self.end_effector.get_pos().cpu().numpy()
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

    def save_observation(self):
        if self.delta[0] == 0.0 and self.delta[1] == 0.0:
            # If no movement, skip saving observation
            return
        print("Saving observation...")
        obs_ee = self.end_effector.get_pos().cpu().numpy()[:2]
        self.scene.draw_debug_sphere(
            pos=np.array([obs_ee[0], obs_ee[1], HEIGHT_OFFSET + ROPE_RADIUS/2]),
            radius=0.001,
            color=(1, 0, 0, 1),  # Red color for end effector
        )
        obs_dlo = dlo_utils.get_skeleton(self.rope.get_particles(),
                                            downsample_number=NUMBER_OF_PARTICLES,
                                            average_number=PARTICLES_NUMBER_FOR_POS_SMOOTHING)[:, :2]
        obs_target = self.target_shape[:, :2]
        self.episode_observations.append(np.vstack([obs_ee, obs_dlo, obs_target]))


    def save_action(self):
        if self.delta[0] == 0.0 and self.delta[1] == 0.0:
            # If no movement, skip saving action
            return
        action = self.end_effector.get_pos().cpu().numpy()[:2]
        self.episode_actions.append(action)

    def run_episode(self):
        ### run the episode ###
        print("Press Enter to continue to next episode...")
        skip_episode = False
        counter = 0
        while not self.enter_pressed:
            counter += 1
            # Add a small sleep to prevent busy waiting            
            if self.backspace_pressed:
                print("Skipping episode, cancelling epsode's data")
                skip_episode = True
                break
            if counter % 4 == 0:
                self.save_observation()
            self.move()
            if counter % 4 == 0:
                self.save_action()


        print("Enter pressed, continuing...")

    def run(self, n_episodes):
        ### run entire process ###
        try:
            for episode in range(n_episodes):
                print(f"Starting episode {episode + 1}/{n_episodes}")
                self.reset(self.initial_pose)
                steps_counter, skip_episode = self.run_episode()
                ## Save the episode data if not skipped ##
                if not skip_episode:
                    np.savez(f"{self.npz_save_path}_{steps_counter}", 
                            observations=self.episode_observations, 
                            actions=self.episode_actions, 
                            episode_ends=steps_counter)
                ## reset episode data ##
                self.episode_observations = []
                self.episode_actions = []


        finally:
            # Clean up the keyboard listener
            self.listener.stop()

    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'listener'):
            self.listener.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teleop Push Data Generator")
    parser.add_argument("-g", "--gui", action="store_true", help="Enable GUI mode")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of GPU")
    parser.add_argument("-f", "--show_fps", action="store_true", help="Show FPS in the viewer")
    parser.add_argument("-e", "--n_episodes", type=int, default=NUMBER_OF_EPISODES, help="Number of episodes to run")
    parser.add_argument("-n", "--save_name", type=str, default="dummy.npz", help="save name")
    args = parser.parse_args()

    generator = TeleopPushDataGenerator(gui=args.gui, cpu=args.cpu, show_fps=args.show_fps, save_name=args.save_name)
    generator.run(n_episodes=args.n_episodes)