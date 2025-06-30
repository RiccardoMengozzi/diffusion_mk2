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

    for file_path in files:
        print(f"Loading {file_path}...")
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    
                    if data.get("type") == "data":
                        # For 'data' entries, just append them as they are
                        combined_output_lines.append(json.dumps(data))

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pattern", type=str, default="json_data/simple_fc_*.jsonl", 
                       help="Pattern to match input JSONL files (e.g., 'json_data/prova_*.jsonl')")
    parser.add_argument("--output_name", type=str, default="json_data/simple_fc_combined.jsonl", 
                       help="Path for combined output (will be saved as .jsonl)")
    
    args = parser.parse_args()
    
    combine_datasets_to_jsonl(args.input_pattern, args.output_name)