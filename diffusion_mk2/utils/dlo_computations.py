from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import genesis as gs


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
    particles: NDArray,
    particle_index: int,
    ee_quat_offset: NDArray,
    ee_offset: float,
) -> Tuple[NDArray, NDArray]:
    particle_frames = compute_particle_frames(particles)
    R_offset = gs.quat_to_R(ee_quat_offset)
    quaternion = gs.R_to_quat(particle_frames[particle_index] @ R_offset)
    pos = particles[particle_index] + np.array([0.0, 0.0, ee_offset])
    return pos, quaternion


def draw_skeleton(particles: NDArray[np.float32], 
                  scene: gs.Scene,
                  rope_radius: float,
                  ) -> None:
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
        T[:3, 3] = t

        # 3) Disegna il frame coordinato con i tre assi
        #    Puoi regolare axis_length, origin_size e axis_radius a piacere
        scene.draw_debug_frame(
            T,
            axis_length=axis_length,  # lunghezza delle frecce (adatta al tuo caso)
            origin_size=rope_radius,  # raggio della sfera in origine
            axis_radius=rope_radius / 2,  # spessore delle frecce
        )


def get_skeleton(
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

    indices = downsample_array(array=particles, final_length=m)
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


def downsample_array(array: NDArray, final_length: int) -> NDArray:
    n = array.shape[0]
    m = final_length
    indices: list[int] = []
    for k in range(m):
        idx_float = k * (n - 1) / (m - 1)
        idx_int = int(np.floor(idx_float))
        indices.append(idx_int)

    return np.array(indices, dtype=int)