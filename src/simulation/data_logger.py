import pandas as pd
import os
class TrajectoryLogger:
    def __init__(self, output_dir="src/simulation/trajectory_data"):
        self.trajectory_data = []
        self.buffer_size = 1000  # Adjust the buffer size as needed
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)


    def log(self, episode_id, timestep, before_state, after_state, action, reward):
       
        cx_pos = before_state[0][0]
        cy_pos = before_state[0][1]
        cz_pos = before_state[0][2]
        cx_vel = before_state[1][0]
        cy_vel = before_state[1][1]
        cz_vel = before_state[1][2]
        nx_pos = after_state[0][0]
        ny_pos = after_state[0][1]
        nz_pos = after_state[0][2]
        nx_vel = after_state[1][0]
        ny_vel = after_state[1][1]
        nz_vel = after_state[1][2]

        self.trajectory_data.append({
            "episode_id": episode_id,
            "timestep": timestep,
            "cx_pos": cx_pos,
            "cy_pos": cy_pos,
            "cz_pos": cz_pos,
            "cx_vel": cx_vel,
            "cy_vel": cy_vel,
            "cz_vel": cz_vel,
            "nx_pos": nx_pos,
            "ny_pos": ny_pos,
            "nz_pos": nz_pos,
            "nx_vel": nx_vel,
            "ny_vel": ny_vel,
            "nz_vel": nz_vel,
            "action": action,
            "reward": reward
        })

        if len(self.trajectory_data) >= self.buffer_size:
            self.save_to_parquet(os.path.join(self.output_dir, f"trajectory_data_{episode_id}.parquet"))
    
    def save_to_parquet(self, file_path):
        df = pd.DataFrame(self.trajectory_data)
        df.to_parquet(file_path, index=False)
        self.trajectory_data = []  # Clear the buffer after saving