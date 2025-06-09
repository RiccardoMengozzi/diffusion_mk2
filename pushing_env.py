import argparse
import numpy as np
import genesis as gs
from genesis.engine.entities import RigidEntity, MPMEntity
from genesis.engine.entities.rigid_entity import RigidLink
from numpy.typing import NDArray
from typing import Tuple
from scipy.spatial.transform import Rotation as R
import time
from tqdm import tqdm
from inference_state import InferenceState
import collections


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

NUMBER_OF_EPISODES = 3
ACTION_TIME = 2.56 # seconds (for 256 batch size)
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


def compute_particle_frames(particles: NDArray[np.float32]) -> NDArray[np.float32]:
    vectors = np.diff(
        particles, axis=0
    )  # Compute vectors between consecutive particles
    reference_axis = np.array([0.0, 0.0, 1.0])  # Z-axis as reference
    perpendicular_vectors = -np.cross(
        vectors, reference_axis
    )  # Compute perpendicular vectors
    reference_axiss = np.tile(reference_axis, (vectors.shape[0], 1))

    vectors = vectors / np.linalg.norm(
        vectors, axis=1, keepdims=True
    )  # Normalize vectors
    perpendicular_vectors = perpendicular_vectors / np.linalg.norm(
        perpendicular_vectors, axis=1, keepdims=True
    )
    particle_frames = np.stack(
        (vectors, perpendicular_vectors, reference_axiss), axis=2
    )

    for i, particle_frame in enumerate(particle_frames):
        # SVD della singola 3×3
        U, _, Vt = np.linalg.svd(particle_frame)
        # calcola determinante del proiettato U@Vt
        det_uv = np.linalg.det(U @ Vt)
        # costruisci D = diag(1,1,sign(det(UVt)))
        D = np.diag([1.0, 1.0, det_uv])
        # rettifica in SO(3)
        particle_frames[i] = U @ D @ Vt

    last_frame = particle_frames[-1].copy()
    particle_frames = np.concatenate((particle_frames, last_frame[None, ...]), axis=0)
    return particle_frames


def compute_pose_from_paticle_index(
    particles: NDArray, particle_index: int
) -> Tuple[NDArray, NDArray]:
    particle_frames = compute_particle_frames(particles)
    R_offset = gs.quat_to_R(EE_QUAT_ROTATION)
    quaternion = gs.R_to_quat(particle_frames[particle_index] @ R_offset)
    pos = particles[particle_index] + np.array([0.0, 0.0, EE_OFFSET])
    return pos, quaternion

def draw_skeleton(particles: NDArray[np.float32], scene: gs.Scene):
    scene.clear_debug_objects()
    particle_frames = compute_particle_frames(particles)
    axis_length = np.linalg.norm(particles[1] - particles[0])
    for i, frame3x3 in enumerate(particle_frames):
        # 1) Estrai la rotazione R (3×3) e la posizione t (x,y,z)
        R = frame3x3
        t = particles[i]  # array di forma (3,)

        # 2) Costruisci T ∈ ℝ⁴ˣ⁴ in forma omogenea
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3,  3] = t

        # 3) Disegna il frame coordinato con i tre assi
        #    Puoi regolare axis_length, origin_size e axis_radius a piacere
        scene.draw_debug_frame(
            T,
            axis_length=axis_length,   # lunghezza delle frecce (adatta al tuo caso)
            origin_size=ROPE_RADIUS,  # raggio della sfera in origine
            axis_radius=ROPE_RADIUS / 2    # spessore delle frecce
        )


def step(
    scene: gs.Scene,
    cam,
    track: bool=False,
    link=None,
    track_offset=np.array([0.0, 0.0, 0.0]),
    gui: bool = False,
    render_interval: int = 1,
    current_step: int = 1,
    rope: MPMEntity = None,
    draw_skeleton_frames: bool = False,
    show_real_time_factor: bool = False,
):
    """
    Step the scene and update the camera.
    """
    start_time = time.time()

    render = False
    if current_step % render_interval == 0:
        render = True 

    scene.step(update_visualizer=render)
    if draw_skeleton_frames:
        assert rope is not None, "Rope entity must be provided to draw skeleton frames."
        particles = sample_skeleton_particles(
            rope.get_particles(), NUMBER_OF_PARTICLES, PARTICLES_NUMBER_FOR_POS_SMOOTHING
        )
        draw_skeleton(particles, scene)
    if gui:
        if render: 
            cam.render()
    if track:
        assert link is not None, "Link must be provided to track the camera."
        ee_pos = link.get_pos().cpu().numpy() + track_offset
        cam.set_pose(pos=ee_pos, lookat=[ee_pos[0], ee_pos[1], 0.0])

    end_time = time.time()
    if show_real_time_factor:
        real_time_factor = DT / (end_time - start_time)
        print(f"Real-time factor: {real_time_factor:.4f}x")


def sample_skeleton_particles(
    particles: NDArray[np.float32],
    downsample_number: int,
    average_number: int
) -> NDArray[np.float32]:
    """
    Pick `downsample_number` skeleton points from `particles` (shape (n, d))
    so that their indices go from 0 to n-1 as evenly as possible, and then
    for each chosen index i_k, return the average of ~`average_number` neighbors
    (floor(average_number/2) before and floor(average_number/2) after, clamped).

    Args:
        particles:  NumPy array of shape (n, d), where n >= 1, d >= 1.
                    Each row is the (x,y,…) position of a rope particle.
        downsample_number:   m = how many skeleton points you want (2 <= m <= n).
        average_number:      window size used to average around each chosen index.
                             Must be >= 1. If even, the window will be
                             of length (2*(average_number//2) + 1), so effectively
                             it’s “average_number//2 before + center + average_number//2 after”.

    Returns:
        A NumPy array of shape (m, d). Each row is the mean of all particles
        whose indices lie in [i_k - half, i_k + half], clamped to [0, n-1].
        The first returned point corresponds to index 0; the last to index n-1.
    """
    n, dim = particles.shape
    m = downsample_number
    if m < 2 or m > n:
        raise ValueError(f"downsample_number (={m}) must satisfy 2 <= m <= n (={n}).")
    if average_number < 1:
        raise ValueError("average_number must be >= 1.")

    # 1) Compute the m “ideal” floating‐point indices in [0, n-1]:
    #    i_k* = k * (n-1)/(m-1), for k=0,...,m-1.
    # 2) Take floor so that i_0 = 0, i_{m-1} = n-1 exactly, and the sequence is strictly increasing.
    indices: list[int] = []
    for k in range(m):
        idx_float = k * (n - 1) / (m - 1)
        idx_int = int(np.floor(idx_float))
        indices.append(idx_int)

    # 3) For each chosen index, define a symmetric window of radius half_window = average_number//2.
    half_window = average_number // 2

    skeleton_positions = np.zeros((m, dim), dtype=particles.dtype)
    for out_i, center_idx in enumerate(indices):
        # window runs from center_idx - half_window ... center_idx + half_window
        start = max(0, center_idx - half_window)
        end = min(n, center_idx + half_window + 1)  # +1 because slice end is exclusive

        window = particles[start:end]  # shape maybe smaller than (average_number, dim) near edges
        skeleton_positions[out_i] = window.mean(axis=0)

    return skeleton_positions


def draw_action(scene : gs.Scene, action):
    for i, pos in enumerate(action):
        if len(action) > 1:
            t = i / (len(action) - 1)
        else:
            t = 0.0

        # Define a red→blue gradient: red at t=0, blue at t=1
        color = [1.0 - t, 0.0, t, 1.0]

        pos = [pos[0], pos[1], HEIGHT_OFFSET]
        scene.draw_debug_sphere(
            pos=pos,
            radius=0.005,
            color=color
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-g", "--gui", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    parser.add_argument("-f", "--show_fps", action="store_true", default=False)
    parser.add_argument(
        "-n", "--n_episodes", type=int, default=NUMBER_OF_EPISODES,
        help="Number of episodes to run. Default is 3."
    )
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu if args.cpu else gs.gpu,
            logging_level="info",
            )

    ########################## create a scene ##########################
    scene: gs.Scene = gs.Scene(
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
        show_FPS=args.show_fps,
        show_viewer=args.vis,
    )

    cam = scene.add_camera(
        res=(1080, 720),
        pos=(0.5, -0.5, TABLE_HEIGHT + .6),
        lookat=(0.5, 0.0, TABLE_HEIGHT),
        fov=50,
        GUI=args.gui,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )

    table = scene.add_entity(
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

    rope: MPMEntity = scene.add_entity(
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
    franka: RigidEntity = scene.add_entity(
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
    scene.build()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)

    # Optional: set control gains
    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )

    end_effector : RigidLink = franka.get_link("hand")
    model = InferenceState("pushing_model.pt", device=gs.device)
    obs_horizon = model.obs_horizon
    observation_ee = end_effector.get_pos()[:2].cpu().numpy()
    observation_dlo = sample_skeleton_particles(
        rope.get_particles(), NUMBER_OF_PARTICLES, PARTICLES_NUMBER_FOR_POS_SMOOTHING
    )[:, :2]
    obs = np.vstack([observation_ee[None, :], observation_dlo])
    obs_deque = collections.deque(
        [obs] * obs_horizon, maxlen=obs_horizon)
    for i in tqdm(range(args.n_episodes)):
        # Set franka to initial position
        qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])
        franka.set_qpos(qpos)

        # Reset rope position with some random noise
        rope.set_position(
            ROPE_BASE_POSITION + np.array([
                np.random.uniform(low=-0.1, high=0.1),
                np.random.uniform(low=-0.1, high=0.1),
                0.0]),
        )

        # Step the scene to initialize the positions
        step(scene, cam, link=end_effector)

        # Create the rope skeleton
        particles = rope.get_particles()
        particles = sample_skeleton_particles(
            particles, NUMBER_OF_PARTICLES, PARTICLES_NUMBER_FOR_POS_SMOOTHING
        )

        # Compute the target pose for middle particle
        idx = len(particles) // 2  # middle particle
        target_pos, target_quat = compute_pose_from_paticle_index(particles, idx)
        target_pos += np.array([-0.05 + np.random.uniform(low=-0.01, high=0.01), 0.0, 0.0])

        # move to pre action position
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=target_pos,
            quat=target_quat,
        )
        qpos[-2:] = 0.0  # fingers closed
        franka.set_qpos(qpos)
        step(scene, cam, link=end_effector)

        observation_ee = end_effector.get_pos()[:2].cpu().numpy()
        observation_dlo = sample_skeleton_particles(
            rope.get_particles(), NUMBER_OF_PARTICLES, PARTICLES_NUMBER_FOR_POS_SMOOTHING
        )[:, :2]
        obs = np.vstack([observation_ee[None, :], observation_dlo])
        obs_deque.append(obs)
        obs = np.stack(obs_deque)

        pred_action = model.run_inference(
            observation=obs,
        )
        print(f"predicted action: {pred_action}")

        scene.clear_debug_objects()
        draw_action(scene, pred_action)

        for action in pred_action:
            target_pos = np.array([action[0], action[1], HEIGHT_OFFSET + EE_OFFSET])
            qpos = franka.inverse_kinematics(
                link=end_effector,
                pos=target_pos,
                quat=target_quat,
                rot_mask=[False,False,True],
            )
            qpos[-2:] = 0.0
            path = franka.plan_path(
                qpos,
                num_waypoints=25
            )

            for waypoint in path:
                franka.control_dofs_position(waypoint, [*motors_dof, *fingers_dof])
                step(
                    scene,
                    cam,
                    gui=args.gui,
                    rope=rope,
                    draw_skeleton_frames=False,
                    show_real_time_factor=False,
                )
          



if __name__ == "__main__":
    main()
