import subprocess
import os
import argparse
PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_multiple_generators(
    n_instances: int = 4,
    n_episodes: int = 3,
    n_actions: int = 10,
    action_length: int = 10,
    script_path: str = "pushing_dataset_generator.py",
    vis: bool = False,
    gui: bool = False,
    cpu: bool = False,
    show_fps: bool = False
):
    """Launch multiple instances of the dataset generator."""
    
    processes = []
    
    for i in range(n_instances):
        # Create unique save path for each instance
        save_name = f"dataset_{i}.npz"
        
        # Build command
        cmd = [
            "python", script_path,
            "-e", str(n_episodes),
            "-a", str(n_actions), 
            "-l", str(action_length),
            "-n", save_name
        ]
        
        # Add boolean flags only if True
        if vis:
            cmd.append("-v")
        if gui:
            cmd.append("-g")
        if cpu:
            cmd.append("-c")
        if show_fps:
            cmd.append("-f")
        
        print(f"Starting instance {i}: {' '.join(cmd)}")
        
        # Launch process
        process = subprocess.Popen(cmd)
        processes.append(process)
    
    print(f"Launched {n_instances} instances. Waiting for completion...")
    
    # Wait for all to finish
    for i, process in enumerate(processes):
        process.wait()
        print(f"Instance {i} finished with return code: {process.returncode}")
    
    print("All instances completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_instances", type=int, default=4, help="Number of instances to run")
    parser.add_argument("-e", "--n_episodes", type=int, default=3, help="Episodes per instance")
    parser.add_argument("-a", "--n_actions", type=int, default=10, help="Actions per episode")
    parser.add_argument("-l", "--action_length", type=int, default=10, help="Action length")
    parser.add_argument("--script_path", type=str, default="pushing_dataset_generator.py", help="Path to the generator script")
    parser.add_argument("-v", "--vis", action="store_true", help="Enable visualization")
    parser.add_argument("-g", "--gui", action="store_true", help="Enable GUI")
    parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("-s", "--save_path", type=str, default="npz_data/dataset.npz", help="Path to save the dataset")
    parser.add_argument("-f", "--show_fps", action="store_true", help="Show FPS")
    
    args = parser.parse_args()
    
    # Make sure output directory exists
    os.makedirs("npz_data", exist_ok=True)
    
    run_multiple_generators(
        n_instances=args.n_instances,
        n_episodes=args.n_episodes,
        n_actions=args.n_actions,
        action_length=args.action_length,
        script_path=args.script_path,
        vis=args.vis,
        gui=args.gui,
        cpu=args.cpu,
        show_fps=args.show_fps
    )