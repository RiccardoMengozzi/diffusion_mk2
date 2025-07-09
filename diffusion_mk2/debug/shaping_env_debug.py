from diffusion_mk2.envs.dlo_shaping.shaping_env import ShapingEnv
from diffusion_mk2.inference.shaping_inference import ShapingInference
import argparse
from tqdm import tqdm
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Teleop Push Data Generator")
    parser.add_argument("--cfg", type=str, default="diffusion_mk2/config/dlo_shapes_with_grasping_env.yaml", help="Path to the configuration file")
    parser.add_argument("-v", "--vis", action="store_true")
    parser.add_argument("-g", "--gui", action="store_true", help="Enable GUI mode")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of GPU")
    parser.add_argument("-f", "--show_fps", action="store_true", help="Show FPS in the viewer")
    parser.add_argument("-e", "--n_episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("-a", "--n_actions", type=int, default=1, help="Number of actions per episode")
    args = parser.parse_args()


    env = ShapingEnv(args=args)
    inf = ShapingInference(
        ckp_path=env.model_path,
    )

    for episode in tqdm(range(env.n_episodes)):
        env.reset()
        for action in tqdm(range(env.n_actions)):
            # Get observation
            obs = env.get_observation()

            env.obs_deque.append(obs)
            obs = np.stack(env.obs_deque)
            obs = obs.reshape(env.model.obs_horizon, -1)

            obs_n = inf._prepare_observation_conditioning(obs)
            
            last_obs = obs_n[-1]
            last_obs_n = obs_n[-1]

            print(f"Observation shape: {last_obs_n.shape}")




if __name__ == "__main__":
    main()