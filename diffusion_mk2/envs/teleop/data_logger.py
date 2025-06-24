import os
import json


class JSONLDataLogger:
    """Handles JSONL file operations for streaming data"""
    
    def __init__(self, save_path, save_name):
        self.save_dir = save_path
        self.save_name = save_name
        self.filename = os.path.join(self.save_dir, self.save_name)

        # If folder does not exist, create it
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

        # If the file already exist, append data to it
        last_episode_idx = 0
        if os.path.exists(self.filename):
            with open(self.filename, "r") as f:
                for line in f:
                    data = json.loads(line)
                    if data.get("type") == "episode_end":
                        last_episode_idx = data.get("episode_idx", last_episode_idx)

        self.total_steps = last_episode_idx       
        self.episode_current_step = 0
        self.action_current_step = 0
        self.episode_data = []
        self.action_data = []
        self.file = None
        
    def initialize_file(self):
        """Initialize JSONL file"""
        self.file = open(self.filename, 'a')
        
    
    
    def append_data(self, obs_ee, obs_dlo, obs_target, action):
        """Append observation and action to file"""
        data = {
            "type": "data",
            "obs_ee": obs_ee.tolist(),  # Convert numpy array to list
            "obs_dlo": obs_dlo.tolist(),  # Convert numpy array to list
            "obs_target": obs_target.tolist(),  # Convert
            "action": action.tolist()
        }
        self.episode_data.append(data)
        self.action_data.append(data)
        self.episode_current_step += 1
        self.action_current_step += 1

    def delete_episode_data(self):
        self.episode_data = []  # Clear the episode data after saving
        self.episode_current_step = 0

    def delete_action_data(self):
        self.action_data = []
        self.action_current_step = 0

    def update_action_data(self, obs_target):
        """Once action is finished, update the action data with the target observation"""
        print("Updating action data with target observation...")
        if self.action_data:
            action_data_length = len(self.action_data)
            print(f"Action data length: {action_data_length}")
            # Update the last action data with the target observation
            for i in range(1, action_data_length + 1):
                self.action_data[-i]["obs_target"] = obs_target.tolist()
        else:
            print("No action data to update.")
        self.delete_action_data()


    def save_episode(self):
        """Mark the end of an episode and save all data"""
        # Save episode data
        for data in self.episode_data:
            self.file.write(json.dumps(data) + '\n')
            self.total_steps += 1

        # Save episode end idx
        episode_end = {
            "type": "episode_end",
            "episode_idx": self.total_steps,
        }
        self.file.write(json.dumps(episode_end) + '\n')
        self.file.flush()
        self.delete_episode_data()  # Clear the episode data after saving


    
    def close(self):
        """Close the file"""
        if self.file:
            self.file.close()
            print(f"Closed JSONL file '{self.save_name} with {self.total_steps} total steps")



