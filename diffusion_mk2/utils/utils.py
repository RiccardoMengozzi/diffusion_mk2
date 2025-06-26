import colorsys

def next_gradient_color(
    color,
    step = 0.1
):
    print("color, ", color)
    r, g, b, a = color
    # Convert RGB to HSV
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    # Advance hue, wrapping around at 1.0
    h = (h + step) % 1.0
    # Convert back to RGB
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
    return (r2, g2, b2, a)

def load_yaml(file_path: str):
    """
    Load a YAML file and return its content.
    
    Args:
        file_path (str): Path to the YAML file.
        
    Returns:
        dict: Content of the YAML file.
    """
    import yaml
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)