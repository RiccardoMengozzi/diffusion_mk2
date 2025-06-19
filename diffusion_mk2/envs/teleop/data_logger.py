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
        self.episode_data = []
        self.file = None
        
    def initialize_file(self):
        """Initialize JSONL file"""
        self.file = open(self.filename, 'a')
        
    
    
    def append_data(self, observation, action):
        """Append observation and action to file"""
        data = {
            "type": "data",
            "observation": observation.tolist(),  # Convert numpy array to list
            "action": action.tolist()
        }
        self.episode_data.append(data)
        self.episode_current_step += 1

    def delete_episode_data(self):
        self.episode_data = []  # Clear the episode data after saving
        self.episode_current_step = 0


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
            print(f"Closed JSONL file '{self.save_name}.jsonl' with {self.total_steps} total steps")



