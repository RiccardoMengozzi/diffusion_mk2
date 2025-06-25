


def draw_action_trajectory(scene, action, offset, radius) -> None:
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

        pos_3d = [pos[0], pos[1], pos[2] - offset]
        
        scene.draw_debug_sphere(
            pos=pos_3d,
            radius=radius,
            color=color
        )

