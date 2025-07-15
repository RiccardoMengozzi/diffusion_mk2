import os
import json
import numpy as np
from typing import List, Dict, Any, Optional

# ===== CONFIGURABLE VARIABLES =====
MIN_EPISODE_LENGTH = 10          # Minimum number of observations required per episode
DELTA_Z_THRESHOLD = 1e-4         # Threshold for Z-axis movement detection
INPUT_FILENAME = "combined_dataset.jsonl"
OUTPUT_FILENAME = "combined_dataset_simplified.jsonl"
DATA_SUBDIRECTORY = "json_data"
# ==================================


def get_file_paths() -> tuple[str, str]:
    """Get input and output file paths."""
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_file = os.path.join(project_dir, DATA_SUBDIRECTORY, INPUT_FILENAME)
    output_file = os.path.join(project_dir, DATA_SUBDIRECTORY, OUTPUT_FILENAME)
    return input_file, output_file


def process_episode_data(data: Dict[str, Any], current_z_state: float, 
                        is_first_obs: bool) -> tuple[Optional[Dict[str, Any]], float, bool]:
    """Process a single data observation within an episode."""
    if is_first_obs:
        return None, data["obs_ee"][2], False
    
    new_z_state = data["obs_ee"][2]
    delta_z = abs(new_z_state - current_z_state)
    
    if delta_z < DELTA_Z_THRESHOLD:
        return data, new_z_state, False
    
    return None, new_z_state, False


def extract_valid_episodes(filename: str) -> tuple[List[Dict[str, Any]], List[List[Any]], int, int]:
    """Extract valid observations and their corresponding true targets from episodes."""
    valid_observations = []
    episode_true_targets = []
    
    is_first_obs_of_episode = True
    current_z_state = 0.0
    current_episode_obs = []
    current_true_target = None
    
    # Statistics tracking
    original_episodes = 0
    original_observations = 0
    
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                data_type = data.get("type")
                
                if data_type == "data":
                    original_observations += 1
                    processed_data, current_z_state, is_first_obs_of_episode = process_episode_data(
                        data, current_z_state, is_first_obs_of_episode
                    )
                    
                    if processed_data is not None:
                        current_episode_obs.append(processed_data)
                        current_true_target = data["obs_target"]
                
                elif data_type == "episode_end":
                    original_episodes += 1
                    if current_episode_obs:
                        valid_observations.extend(current_episode_obs)
                        episode_true_targets.append(current_true_target)
                        valid_observations.append(data)  # Add episode_end marker
                    
                    # Reset for next episode
                    is_first_obs_of_episode = True
                    current_episode_obs = []
                    current_true_target = None
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                continue
    
    return valid_observations, episode_true_targets, original_episodes, original_observations


def write_filtered_episodes(observations: List[Dict[str, Any]], 
                           true_targets: List[List[Any]], 
                           output_file: str) -> tuple[int, int]:
    """Write filtered episodes to output file."""
    episode_counter = 0
    episode_end_idx = 0
    current_episode_obs = []
    
    # Statistics tracking
    final_episodes = 0
    final_observations = 0
    
    with open(output_file, "w") as outfile:
        for obs in observations:
            if obs.get("type") == "data":
                # Update target for current episode
                if episode_counter < len(true_targets):
                    obs["obs_target"] = true_targets[episode_counter]
                current_episode_obs.append(obs)
                
            elif obs.get("type") == "episode_end":
                # Only write episode if it meets minimum length requirement
                if len(current_episode_obs) >= MIN_EPISODE_LENGTH:
                    # Write all observations in this episode
                    for episode_obs in current_episode_obs:
                        outfile.write(json.dumps(episode_obs) + "\n")
                    
                    # Update episode end index and write episode end marker
                    episode_end_idx += len(current_episode_obs)
                    obs["episode_idx"] = episode_end_idx
                    outfile.write(json.dumps(obs) + "\n")
                    
                    # Update statistics
                    final_episodes += 1
                    final_observations += len(current_episode_obs)
                
                # Reset for next episode
                episode_counter += 1
                current_episode_obs = []
    
    return final_episodes, final_observations


def main() -> None:
    """Main function to process the dataset."""
    input_file, output_file = get_file_paths()
    
    print(f"Processing {input_file}...")
    print(f"Configuration:")
    print(f"  - Minimum episode length: {MIN_EPISODE_LENGTH}")
    print(f"  - Delta Z threshold: {DELTA_Z_THRESHOLD}")
    print()
    
    # Extract valid episodes
    valid_observations, episode_true_targets, original_episodes, original_observations = extract_valid_episodes(input_file)
    
    print(f"Original dataset statistics:")
    print(f"  - Episodes: {original_episodes}")
    print(f"  - Observations: {original_observations}")
    print()
    
    # Write filtered episodes
    final_episodes, final_observations = write_filtered_episodes(valid_observations, episode_true_targets, output_file)
    
    print(f"Final dataset statistics:")
    print(f"  - Episodes: {final_episodes}")
    print(f"  - Observations: {final_observations}")
    print()
    
    print(f"Filtering results:")
    print(f"  - Episodes retained: {final_episodes}/{original_episodes} ({final_episodes/original_episodes*100:.1f}%)")
    print(f"  - Observations retained: {final_observations}/{original_observations} ({final_observations/original_observations*100:.1f}%)")
    print()
    
    print(f"Processed dataset saved to {output_file}")


if __name__ == "__main__":
    main()