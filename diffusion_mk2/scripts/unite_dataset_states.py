import os
import glob
import argparse
import json
import numpy as np # Still useful for internal array handling if needed, though less critical for direct JSONL output

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def combine_datasets_to_jsonl(input_pattern: str = "json_data/prova_*.jsonl", output_name: str = "json_data/combined_dataset.jsonl"):
    """Combine multiple JSONL dataset files into one JSONL file,
    correctly accumulating episode ends."""
    
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
    
    combined_output_lines = []
    total_observations_so_far = 0 # This will track the cumulative number of 'data' entries
    
    total_number_of_observations = 0
    total_number_of_episode_ends = 0
    for file_path in files:
        print(f"Loading {file_path}...")
        try:
            observations_in_current_file = 0
            ep_ends_counter = 0
            ep_ends = []
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    
                    if data.get("type") == "data":
                        # For 'data' entries, just append them as they are
                        combined_output_lines.append(json.dumps(data))
                        observations_in_current_file += 1
                        total_number_of_observations += 1
                    elif data.get("type") == "episode_end":
                        # For 'episode_end' entries, adjust the 'episode_idx'
                        # The 'episode_idx' should refer to the index in the *global* sequence of observations
                        # So, we add the total observations encountered *before* this file.
                        
                        # Ensure 'episode_idx' exists and is an integer
                        if "episode_idx" in data and isinstance(data["episode_idx"], int):
                            adjusted_episode_idx = data["episode_idx"] + total_observations_so_far
                            data["episode_idx"] = adjusted_episode_idx
                            combined_output_lines.append(json.dumps(data))
                            # for loggin
                            ep_ends_counter += 1
                            total_number_of_episode_ends += 1
                            ep_ends.append(adjusted_episode_idx)
                        else:
                            print(f"  Warning: 'episode_end' entry in {file_path} without valid 'episode_idx': {data}")
            
            total_observations_so_far += observations_in_current_file
            
            print(f"  Added {observations_in_current_file} observations and {ep_ends_counter} episode ends {ep_ends} from {file_path}") # This count is a bit tricky, simpler to just say "lines"
            
        except json.JSONDecodeError as e:
            print(f"  Error decoding JSON in {file_path} at line: {line.strip()}. Error: {e}")
        except Exception as e:
            print(f"  An unexpected error occurred loading {file_path}: {e}")
    
    if not combined_output_lines:
        print("No valid data loaded from any files. Exiting.")
        return

    # Write all accumulated lines to the output JSONL file
    with open(output_path, 'w') as f:
        for line in combined_output_lines:
            f.write(line + '\n')
    
    print(f"\nCombined dataset saved to: {output_path}")
    print(f"Total lines in combined JSONL: {len(combined_output_lines)}")
    print(f"Total observations: {total_number_of_observations}")
    print(f"Total episode ends: {total_number_of_episode_ends}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pattern", type=str, default="json_data/dataset_*.jsonl", 
                       help="Pattern to match input JSONL files (e.g., 'json_data/prova_*.jsonl')")
    parser.add_argument("--output_name", type=str, default="json_data/combined_dataset.jsonl", 
                       help="Path for combined output (will be saved as .jsonl)")
    
    args = parser.parse_args()
    
    combine_datasets_to_jsonl(args.input_pattern, args.output_name)