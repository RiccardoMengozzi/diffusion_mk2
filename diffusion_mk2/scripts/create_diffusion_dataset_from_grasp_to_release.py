
import os
import json
import numpy as np

project_dir   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
filename  = os.path.join(project_dir, "json_data", "test.jsonl")
output = os.path.join(project_dir, "json_data", "test_simplified.jsonl")

is_first_obs_of_episode = True
current_z_state = 0.0


simplified_lines = []
episodes_true_targets = []
with open(filename, 'r') as f:
    for line in f:
        try:
            data = json.loads(line.strip())
            if data.get("type") == "data":
                if is_first_obs_of_episode:
                    current_z_state = data["obs_ee"][2]
                    is_first_obs_of_episode = False
                    continue
                else:
                    delta_z = np.abs(data["obs_ee"][2] - current_z_state)
                    current_z_state = data["obs_ee"][2]
                    if delta_z < 1e-4:
                        simplified_lines.append(line)
                        true_target = data["obs_target"]

            elif data.get("type") == "episode_end":
                is_first_obs_of_episode = False
                simplified_lines.append(line)
                episodes_true_targets.append(true_target)
    

        except json.JSONDecodeError as e:
            print(f"Warning: Skipping invalid JSON line: {e}")
            continue

print("episoded = ", np.array(episodes_true_targets).shape)

episode_counter = 0
with open(output, "w") as outfile:
    for line in simplified_lines:
        data = json.loads(line.strip())
        if data.get("type") == "data":
            data["obs_target"]= episodes_true_targets[episode_counter]
        elif data.get("type") == "episode_end":
            episode_counter += 1
        outfile.write(json.dumps(data) + "\n")
        

