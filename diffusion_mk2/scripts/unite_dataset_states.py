import numpy as np
import os
import glob
import argparse

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



def combine_datasets(input_pattern: str = "npz_data/dataset_*.npz", output_name: str = "npz_data/combined_dataset.npz"):
    """Combine multiple dataset files into one."""
    input_pattern = os.path.join(PROJECT_DIR, input_pattern)
    output_path = os.path.join(PROJECT_DIR, output_name)
    
    # Find all matching files
    files = glob.glob(input_pattern)
    files.sort()  # Sort for consistent ordering
    
    if not files:
        print(f"No files found matching pattern: {input_pattern}")
        return
    
    print(f"Found {len(files)} dataset files:")
    for f in files:
        print(f"  {f}")
    
    all_observations = []
    all_actions = []
    all_episode_ends = []
    total_steps = 0
    
    for file_path in files:
        print(f"Loading {file_path}...")
        try:
            data = np.load(file_path)
            observations = data["observations"]
            actions = data["actions"]
            episode_ends = data["episode_ends"]
            
            all_observations.append(observations)
            all_actions.append(actions)
            
            # Adjust episode ends for cumulative counting
            adjusted_episode_ends = [end + total_steps for end in episode_ends]
            all_episode_ends.extend(adjusted_episode_ends)
            total_steps += len(observations)
            
            print(f"  Added {len(observations)} observations")
            
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")
    
    if not all_observations:
        print("No valid datasets found!")
        return
    
    # Combine all data
    combined_observations = np.concatenate(all_observations, axis=0)
    combined_actions = np.concatenate(all_actions, axis=0)
    
    # Save combined dataset
    np.savez(
        output_path,
        observations=combined_observations,
        actions=combined_actions,
        episode_ends=all_episode_ends
    )
    
    print(f"\nCombined dataset saved to: {output_path}")
    print(f"Total observations: {len(combined_observations)}")
    print(f"Total actions: {len(combined_actions)}")
    print(f"Total episodes: {len(all_episode_ends)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pattern", type=str, default="npz_data/dataset_*.npz", 
                       help="Pattern to match input files")
    parser.add_argument("--output_name", type=str, default="npz_data/combined_dataset.npz", 
                       help="Path for combined output")
    
    args = parser.parse_args()
    
    combine_datasets(args.input_pattern, args.output_name)