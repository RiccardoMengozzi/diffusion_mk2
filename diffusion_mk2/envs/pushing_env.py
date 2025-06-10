import argparse
import numpy as np
import genesis as gs
from genesis.engine.entities import RigidEntity, MPMEntity
from genesis.engine.entities.rigid_entity import RigidLink
from numpy.typing import NDArray
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation as R
import time
from tqdm import tqdm
from diffusion_mk2.diffusion_mk2.inference.inference_state import InferenceState
import collections


class PushingEnv:
    """
    A simulation environment for rope manipulation using Genesis physics engine.
    
    This class provides a complete simulation setup for robotic rope manipulation,
    including a Franka Panda robot arm, a deformable rope, and an AI model for
    generating manipulation actions.
    
    Attributes:
        verbose (bool): Controls verbosity of output messages
        scene (gs.Scene): The Genesis physics scene
        cam (gs.Camera): The camera for visualization
        franka (RigidEntity): The Franka Panda robot arm
        rope (MPMEntity): The deformable rope entity
        end_effector (RigidLink): The robot's end effector
        model (InferenceState): The AI model for action prediction
        obs_deque (collections.deque): Rolling buffer for observations
    """
    
    # Simulation constants
    NUMBER_OF_EPISODES = 3
    ACTION_TIME = 2.56  # seconds (for 256 batch size)
    DT = 1e-2  # simulation time step
    MPM_GRID_DENSITY = 256
    SUBSTEPS = 40
    TABLE_HEIGHT = 0.7005
    HEIGHT_OFFSET = TABLE_HEIGHT
    EE_OFFSET = 0.122
    EE_QUAT_ROTATION = np.array([0, 0, -1, 0])
    ROPE_LENGTH = 0.2
    ROPE_RADIUS = 0.003
    ROPE_BASE_POSITION = np.array([0.5, 0.0, HEIGHT_OFFSET + 0.003])
    NUMBER_OF_PARTICLES = 15
    PARTICLES_NUMBER_FOR_POS_SMOOTHING = 10
    
    def __init__(self, 
                 verbose: bool = False,
                 use_cpu: bool = False,
                 show_fps: bool = False,
                 enable_vis: bool = False,
                 enable_gui: bool = False):
        """
        Initialize the rope manipulation simulator.
        
        Args:
            verbose (bool): Enable verbose output for debugging and monitoring
            use_cpu (bool): Use CPU backend instead of GPU for Genesis
            show_fps (bool): Display FPS counter in the viewer
            enable_vis (bool): Enable visualization window
            enable_gui (bool): Enable GUI controls for the camera
        """
        self.verbose = verbose
        self.scene = None
        self.cam = None
        self.franka = None
        self.rope = None
        self.end_effector = None
        self.model = None
        self.obs_deque = None
        
        # Configure numpy printing for better output readability
        np.set_printoptions(
            precision=4,
            suppress=True,
            linewidth=120,
            threshold=1000,
            edgeitems=3,
            formatter={
                float: "{: .4f}".format,
                int: "{: d}".format,
            },
        )
        
        # Initialize Genesis
        self._initialize_genesis(use_cpu)
        
        # Create scene and entities
        self._create_scene(show_fps, enable_vis)
        self._create_camera(enable_gui)
        self._create_entities()
        self._setup_robot_control()
        
        if self.verbose:
            print("RopeManipulationSimulator initialized successfully")
    
    def _initialize_genesis(self, use_cpu: bool) -> None:
        """Initialize the Genesis physics engine."""
        if self.verbose:
            print("Initializing Genesis physics engine...")
        
        gs.init(
            backend=gs.cpu if use_cpu else gs.gpu,
            logging_level="info" if self.verbose else "warning",
        )
    
    def _create_scene(self, show_fps: bool, enable_vis: bool) -> None:
        """Create the physics scene with appropriate options."""
        if self.verbose:
            print("Creating physics scene...")
        
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.DT,
                substeps=self.SUBSTEPS,
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
                lower_bound=(0.2, -0.3, self.HEIGHT_OFFSET - 0.05),
                upper_bound=(0.8, 0.3, self.HEIGHT_OFFSET + 0.1),
                grid_density=self.MPM_GRID_DENSITY,
            ),
            show_FPS=show_fps,
            show_viewer=enable_vis,
        )
    
    def _create_camera(self, enable_gui: bool) -> None:
        """Create the camera for visualization."""
        if self.verbose:
            print("Creating camera...")
        
        self.cam = self.scene.add_camera(
            res=(1080, 720),
            pos=(0.5, -0.5, self.TABLE_HEIGHT + .6),
            lookat=(0.5, 0.0, self.TABLE_HEIGHT),
            fov=50,
            GUI=enable_gui,
        )
    
    def _create_entities(self) -> None:
        """Create all entities in the scene (plane, table, rope, robot)."""
        if self.verbose:
            print("Creating scene entities...")
        
        # Ground plane
        plane = self.scene.add_entity(gs.morphs.Plane())
        
        # Table
        table = self.scene.add_entity(
            morph=gs.morphs.URDF(
                file="models/SimpleTable/SimpleTable.urdf",
                pos=(0.0, 0.0, 0.0),
                euler=(0, 0, 90),
                scale=1,
                fixed=True,
            ),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Default(),
        )
        
        # Deformable rope
        self.rope = self.scene.add_entity(
            material=gs.materials.MPM.Elastic(
                E=5e4,  # Young's modulus - determines rope stiffness
                nu=0.45,  # Poisson's ratio
                rho=2000,  # Density
                sampler="pbs",
            ),
            morph=gs.morphs.Cylinder(
                height=self.ROPE_LENGTH,
                radius=self.ROPE_RADIUS,
                pos=self.ROPE_BASE_POSITION,
                euler=(90, 0, 0),
            ),
            surface=gs.surfaces.Default(roughness=2, vis_mode="particle"),
        )
        
        # Franka Panda robot
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda.xml",
                pos=(0.0, 0.0, self.HEIGHT_OFFSET),
            ),
            material=gs.materials.Rigid(
                friction=2.0,
                needs_coup=True,
                coup_friction=2.0,
                sdf_cell_size=0.005,
            ),
        )
        
        # Build the scene
        self.scene.build()
        
        # Get the end effector link
        self.end_effector = self.franka.get_link("hand")
        
        if self.verbose:
            print("Scene entities created and built successfully")
    
    def _setup_robot_control(self) -> None:
        """Configure robot control parameters."""
        if self.verbose:
            print("Setting up robot control parameters...")
        
        # Set control gains
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
    
    def initialize_model(self, model_path: str) -> None:
        """
        Initialize the AI model for action prediction.
        
        Args:
            model_path (str): Path to the trained model file
        """
        if self.verbose:
            print(f"Loading AI model from {model_path}...")
        
        self.model = InferenceState(model_path, device=gs.device)
        obs_horizon = self.model.obs_horizon
        
        # Initialize observation buffer
        observation_ee = self.end_effector.get_pos()[:2].cpu().numpy()
        observation_dlo = self.sample_skeleton_particles(
            self.rope.get_particles(), 
            self.NUMBER_OF_PARTICLES, 
            self.PARTICLES_NUMBER_FOR_POS_SMOOTHING
        )[:, :2]
        obs = np.vstack([observation_ee[None, :], observation_dlo])
        self.obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
        
        if self.verbose:
            print(f"Model loaded with observation horizon: {obs_horizon}")
    
    @staticmethod
    def compute_particle_frames(particles: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Compute coordinate frames for each particle along the rope.
        
        This method creates a coordinate frame for each particle by computing
        the tangent vector between consecutive particles and using cross products
        to establish orthogonal axes.
        
        Args:
            particles (NDArray[np.float32]): Array of particle positions (N, 3)
        
        Returns:
            NDArray[np.float32]: Array of rotation matrices (N, 3, 3)
        """
        vectors = np.diff(particles, axis=0)  # Tangent vectors
        reference_axis = np.array([0.0, 0.0, 1.0])  # Z-axis reference
        perpendicular_vectors = -np.cross(vectors, reference_axis)
        reference_axiss = np.tile(reference_axis, (vectors.shape[0], 1))

        # Normalize vectors
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        perpendicular_vectors = perpendicular_vectors / np.linalg.norm(
            perpendicular_vectors, axis=1, keepdims=True
        )
        
        # Stack to form rotation matrices
        particle_frames = np.stack(
            (vectors, perpendicular_vectors, reference_axiss), axis=2
        )

        # Ensure proper rotation matrices using SVD
        for i, particle_frame in enumerate(particle_frames):
            U, _, Vt = np.linalg.svd(particle_frame)
            det_uv = np.linalg.det(U @ Vt)
            D = np.diag([1.0, 1.0, det_uv])
            particle_frames[i] = U @ D @ Vt

        # Duplicate last frame for the final particle
        last_frame = particle_frames[-1].copy()
        particle_frames = np.concatenate((particle_frames, last_frame[None, ...]), axis=0)
        
        return particle_frames
    
    def compute_pose_from_particle_index(self, 
                                       particles: NDArray, 
                                       particle_index: int) -> Tuple[NDArray, NDArray]:
        """
        Compute end-effector pose from a specific particle's frame.
        
        Args:
            particles (NDArray): Array of particle positions
            particle_index (int): Index of the particle to use
        
        Returns:
            Tuple[NDArray, NDArray]: Position and quaternion for the pose
        """
        particle_frames = self.compute_particle_frames(particles)
        R_offset = gs.quat_to_R(self.EE_QUAT_ROTATION)
        quaternion = gs.R_to_quat(particle_frames[particle_index] @ R_offset)
        pos = particles[particle_index] + np.array([0.0, 0.0, self.EE_OFFSET])
        return pos, quaternion
    
    def draw_skeleton(self, particles: NDArray[np.float32]) -> None:
        """
        Draw coordinate frames for visualization of rope skeleton.
        
        Args:
            particles (NDArray[np.float32]): Array of skeleton particle positions
        """
        self.scene.clear_debug_objects()
        particle_frames = self.compute_particle_frames(particles)
        axis_length = np.linalg.norm(particles[1] - particles[0])
        
        for i, frame3x3 in enumerate(particle_frames):
            R = frame3x3
            t = particles[i]
            
            # Create homogeneous transformation matrix
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = R
            T[:3, 3] = t
            
            # Draw coordinate frame
            self.scene.draw_debug_frame(
                T,
                axis_length=axis_length,
                origin_size=self.ROPE_RADIUS,
                axis_radius=self.ROPE_RADIUS / 2
            )
    
    def step_simulation(self, 
                       track: bool = False,
                       track_offset: NDArray = np.array([0.0, 0.0, 0.0]),
                       render_interval: int = 1,
                       current_step: int = 1,
                       draw_skeleton_frames: bool = False,
                       show_real_time_factor: bool = False) -> None:
        """
        Step the simulation forward by one time step.
        
        Args:
            track (bool): Whether to track end-effector with camera
            track_offset (NDArray): Offset for camera tracking
            render_interval (int): Render every N steps
            current_step (int): Current simulation step number
            draw_skeleton_frames (bool): Whether to draw rope skeleton frames
            show_real_time_factor (bool): Whether to display real-time performance
        """
        start_time = time.time()
        
        render = (current_step % render_interval == 0)
        
        self.scene.step(update_visualizer=render)
        
        if draw_skeleton_frames:
            particles = self.sample_skeleton_particles(
                self.rope.get_particles(), 
                self.NUMBER_OF_PARTICLES, 
                self.PARTICLES_NUMBER_FOR_POS_SMOOTHING
            )
            self.draw_skeleton(particles)
        
        if render:
            self.cam.render()
        
        if track:
            ee_pos = self.end_effector.get_pos().cpu().numpy() + track_offset
            self.cam.set_pose(pos=ee_pos, lookat=[ee_pos[0], ee_pos[1], 0.0])
        
        if show_real_time_factor:
            end_time = time.time()
            real_time_factor = self.DT / (end_time - start_time)
            print(f"Real-time factor: {real_time_factor:.4f}x")
    
    @staticmethod
    def sample_skeleton_particles(particles: NDArray[np.float32],
                                downsample_number: int,
                                average_number: int) -> NDArray[np.float32]:
        """
        Sample skeleton points from rope particles with smoothing.
        
        This method selects evenly spaced points along the rope and smooths
        each point by averaging with its neighbors.
        
        Args:
            particles (NDArray[np.float32]): All rope particles (n, d)
            downsample_number (int): Number of skeleton points to extract
            average_number (int): Window size for smoothing each point
        
        Returns:
            NDArray[np.float32]: Smoothed skeleton points (downsample_number, d)
        """
        n, dim = particles.shape
        m = downsample_number
        
        if m < 2 or m > n:
            raise ValueError(f"downsample_number (={m}) must satisfy 2 <= m <= n (={n}).")
        if average_number < 1:
            raise ValueError("average_number must be >= 1.")
        
        # Compute evenly spaced indices
        indices = []
        for k in range(m):
            idx_float = k * (n - 1) / (m - 1)
            idx_int = int(np.floor(idx_float))
            indices.append(idx_int)
        
        # Smooth each selected point
        half_window = average_number // 2
        skeleton_positions = np.zeros((m, dim), dtype=particles.dtype)
        
        for out_i, center_idx in enumerate(indices):
            start = max(0, center_idx - half_window)
            end = min(n, center_idx + half_window + 1)
            window = particles[start:end]
            skeleton_positions[out_i] = window.mean(axis=0)
        
        return skeleton_positions
    
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
            pos_3d = [pos[0], pos[1], self.HEIGHT_OFFSET]
            
            self.scene.draw_debug_sphere(
                pos=pos_3d,
                radius=0.005,
                color=color
            )
    
    def reset_episode(self) -> None:
        """Reset the environment for a new episode."""
        if self.verbose:
            print("Resetting episode...")
        
        # Reset robot to initial position
        qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])
        self.franka.set_qpos(qpos)
        
        # Reset rope with random position
        rope_pos = self.ROPE_BASE_POSITION + np.array([
            np.random.uniform(low=-0.1, high=0.1),
            np.random.uniform(low=-0.1, high=0.1),
            0.0
        ])
        self.rope.set_position(rope_pos)
        
        if self.verbose:
            print(f"Rope reset to position: {rope_pos}")
        
        # Step to initialize positions
        self.step_simulation()
    
    def get_observation(self) -> NDArray:
        """
        Get current observation state.
        
        Returns:
            NDArray: Current observation including end-effector and rope state
        """
        observation_ee = self.end_effector.get_pos()[:2].cpu().numpy()
        observation_dlo = self.sample_skeleton_particles(
            self.rope.get_particles(), 
            self.NUMBER_OF_PARTICLES, 
            self.PARTICLES_NUMBER_FOR_POS_SMOOTHING
        )[:, :2]
        
        return np.vstack([observation_ee[None, :], observation_dlo])
    
    def update_observation_buffer(self) -> NDArray:
        """
        Update the observation buffer and return stacked observations.
        
        Returns:
            NDArray: Stacked observations for model input
        """
        obs = self.get_observation()
        self.obs_deque.append(obs)
        return np.stack(self.obs_deque)
    
    def execute_action_sequence(self, actions: NDArray) -> None:
        """
        Execute a sequence of actions by moving the robot.
        
        Args:
            actions (NDArray): Sequence of 2D positions to move to
        """
        if self.verbose:
            print(f"Executing {len(actions)} actions...")
        
        # Get current rope skeleton for pose computation
        particles = self.sample_skeleton_particles(
            self.rope.get_particles(), 
            self.NUMBER_OF_PARTICLES, 
            self.PARTICLES_NUMBER_FOR_POS_SMOOTHING
        )
        
        # Use middle particle for target orientation
        idx = len(particles) // 2
        _, target_quat = self.compute_pose_from_particle_index(particles, idx)
        
        motors_dof = np.arange(7)
        fingers_dof = np.arange(7, 9)
        
        for i, action in enumerate(actions):
            if self.verbose:
                print(f"Executing action {i+1}/{len(actions)}: {action}")
            
            target_pos = np.array([action[0], action[1], 
                                 self.HEIGHT_OFFSET + self.EE_OFFSET])
            
            # Compute inverse kinematics
            qpos = self.franka.inverse_kinematics(
                link=self.end_effector,
                pos=target_pos,
                quat=target_quat,
                rot_mask=[False, False, True],
            )
            qpos[-2:] = 0.0  # Keep fingers closed
            
            # Plan and execute path
            path = self.franka.plan_path(qpos, num_waypoints=25)
            
            for waypoint in path:
                self.franka.control_dofs_position(waypoint, [*motors_dof, *fingers_dof])
                self.step_simulation()
    
    def run_episode(self) -> None:
        """Run a complete episode of rope manipulation."""
        if self.verbose:
            print("Starting new episode...")
        
        # Reset environment
        self.reset_episode()
        
        # Get rope skeleton and compute initial target pose
        particles = self.sample_skeleton_particles(
            self.rope.get_particles(), 
            self.NUMBER_OF_PARTICLES, 
            self.PARTICLES_NUMBER_FOR_POS_SMOOTHING
        )
        
        # Move to pre-action position (middle of rope with offset)
        idx = len(particles) // 2
        target_pos, target_quat = self.compute_pose_from_particle_index(particles, idx)
        target_pos += np.array([-0.05 + np.random.uniform(low=-0.01, high=0.01), 0.0, 0.0])
        
        qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=target_pos,
            quat=target_quat,
        )
        qpos[-2:] = 0.0  # Close fingers
        self.franka.set_qpos(qpos)
        self.step_simulation()
        
        # Update observations and get model prediction
        obs = self.update_observation_buffer()
        pred_action = self.model.run_inference(observation=obs)
        
        if self.verbose:
            print(f"Model predicted action: {pred_action}")
        
        # Visualize and execute actions
        self.scene.clear_debug_objects()
        self.draw_action_trajectory(pred_action)
        self.execute_action_sequence(pred_action)
    
    def run_simulation(self, n_episodes: int) -> None:
        """
        Run the complete simulation for specified number of episodes.
        
        Args:
            n_episodes (int): Number of episodes to run
        """
        if self.verbose:
            print(f"Starting simulation with {n_episodes} episodes...")
        
        for i in tqdm(range(n_episodes), desc="Running episodes"):
            if self.verbose:
                print(f"\n--- Episode {i+1}/{n_episodes} ---")
            self.run_episode()
        
        if self.verbose:
            print("Simulation completed successfully!")


def main():
    """Main function to run the rope manipulation simulation."""
    parser = argparse.ArgumentParser(description="Rope Manipulation Simulation")
    parser.add_argument("-v", "--vis", action="store_true", default=False,
                       help="Enable visualization window")
    parser.add_argument("-g", "--gui", action="store_true", default=False,
                       help="Enable GUI controls")
    parser.add_argument("-c", "--cpu", action="store_true", default=False,
                       help="Use CPU backend instead of GPU")
    parser.add_argument("-f", "--show_fps", action="store_true", default=False,
                       help="Show FPS counter")
    parser.add_argument("--verbose", action="store_true", default=False,
                       help="Enable verbose output")
    parser.add_argument("-n", "--n_episodes", type=int, 
                       default=PushingEnv.NUMBER_OF_EPISODES,
                       help="Number of episodes to run")
    args = parser.parse_args()
    
    # Create simulator instance
    simulator = PushingEnv(
        verbose=args.verbose,
        use_cpu=args.cpu,
        show_fps=args.show_fps,
        enable_vis=args.vis,
        enable_gui=args.gui
    )
    
    # Initialize the AI model
    simulator.initialize_model("pushing_model.pt")
    
    # Run the simulation
    simulator.run_simulation(args.n_episodes)


if __name__ == "__main__":
    main()