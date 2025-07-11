from diffusion_mk2.envs.dlo_shaping.shaping_env import ShapingEnv
from diffusion_mk2.inference.shaping_inference import ShapingInference
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


np.set_printoptions(precision=4,    # number of decimal places
                    suppress=True,  # suppress scientific notation
                    linewidth=100,  # characters per line
                    threshold=1000) # controls summarization of large arrays


RANDOM_SHAPE = [
    [0.5167030096054077, 0.09585455805063248, 0.7029711604118347],
    [0.5155951976776123, 0.08386475592851639, 0.7035097479820251],
    [0.514057457447052, 0.06953224539756775, 0.7032835483551025],
    [0.5125030875205994, 0.05566858872771263, 0.7034311890602112],
    [0.5108361840248108, 0.041574593633413315, 0.7035626173019409],
    [0.5087506771087646, 0.02741190977394581, 0.7033372521400452],
    [0.5060816407203674, 0.01369813084602356, 0.7035087943077087],
    [0.5026120543479919, -0.0008324492955580354, 0.7033385634422302],
    [0.498716801404953, -0.014265832491219044, 0.7035095691680908],
    [0.4937152862548828, -0.027831846848130226, 0.7032997608184814],
    [0.48850688338279724, -0.04080011323094368, 0.7034475803375244],
    [0.4843168556690216, -0.05432935804128647, 0.7040033340454102],
    [0.4825930595397949, -0.06851319223642349, 0.7037916779518127],
    [0.482631653547287, -0.08237171173095703, 0.703498899936676],
    [0.4834268391132355, -0.0949200913310051, 0.7038154006004333],
]

RANDOM_TARGET = [
    [0.5167364478111267, 0.09626814723014832, 0.7029712200164795],
    [0.5156363844871521, 0.08427868783473969, 0.7035102844238281],
    [0.5141038298606873, 0.06994378566741943, 0.7032839059829712],
    [0.5125567317008972, 0.05607892572879791, 0.7034321427345276],
    [0.5109126567840576, 0.0419762097299099, 0.7035642266273499],
    [0.5088381171226501, 0.027801604941487312, 0.7033417820930481],
    [0.5060871839523315, 0.014096233062446117, 0.7035146355628967],
    [0.5025281310081482, -0.0004252632206771523, 0.7033442854881287],
    [0.4994399845600128, -0.014070695266127586, 0.7035157680511475],
    [0.49714547395706177, -0.02833779901266098, 0.7033063173294067],
    [0.49673935770988464, -0.04229258745908737, 0.7034505605697632],
    [0.49893248081207275, -0.05626344308257103, 0.7040116786956787],
    [0.5046352744102478, -0.06933712959289551, 0.703833281993866],
    [0.5118126273155212, -0.0812348872423172, 0.7038783431053162],
    [0.5182612538337708, -0.09208345413208008, 0.7042266726493835],
]


def main():
    parser = argparse.ArgumentParser(description="Teleop Push Data Generator")
    parser.add_argument(
        "--cfg",
        type=str,
        default="diffusion_mk2/config/dlo_shapes_with_grasping_env.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument("-v", "--vis", action="store_true")
    parser.add_argument("-g", "--gui", action="store_true", help="Enable GUI mode")
    parser.add_argument(
        "-c", "--cpu", action="store_true", help="Run on CPU instead of GPU"
    )
    parser.add_argument(
        "-f", "--show_fps", action="store_true", help="Show FPS in the viewer"
    )
    parser.add_argument(
        "-e", "--n_episodes", type=int, default=1, help="Number of episodes to run"
    )
    parser.add_argument(
        "-a", "--n_actions", type=int, default=1, help="Number of actions per episode"
    )
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
            obs = np.concatenate(
                [
                    obs[:inf.obs_ee_dim],
                    np.array(RANDOM_SHAPE).flatten(),
                    np.array(RANDOM_TARGET).flatten(),
                ]
            )
            env.obs_deque.append(obs)
            obs = np.stack(env.obs_deque)
            obs = obs.reshape(env.model.obs_horizon, -1)

            # Get normalized observation
            obs_n = inf._process_observation_conditioning(obs)
            obs_n = obs_n.reshape(env.model.obs_horizon, -1)

            last_obs = obs[-1]
            last_obs_n = obs_n[-1]
            ee_state = last_obs[: inf.obs_ee_dim]
            dlo_state = last_obs[inf.obs_ee_dim : inf.obs_ee_dim + inf.obs_dlo_dim]
            target_shape = last_obs[
                inf.obs_ee_dim
                + inf.obs_dlo_dim : inf.obs_ee_dim
                + inf.obs_dlo_dim
                + inf.obs_target_dim
            ]
            dlo_state = dlo_state.reshape(inf.obs_dlo_dim // 3, 3)
            target_shape = target_shape.reshape(inf.obs_target_dim // 3, 3)

            ee_state_n = last_obs_n[: inf.obs_ee_dim]
            dlo_state_n = last_obs_n[inf.obs_ee_dim : inf.obs_ee_dim + inf.obs_dlo_dim]
            target_shape_n = last_obs_n[
                inf.obs_ee_dim
                + inf.obs_dlo_dim : inf.obs_ee_dim
                + inf.obs_dlo_dim
                + inf.obs_target_dim
            ]
            dlo_state_n = dlo_state_n.reshape(inf.obs_dlo_dim // 3, 3).cpu().numpy()
            target_shape_n = (
                target_shape_n.reshape(inf.obs_target_dim // 3, 3).cpu().numpy()
            )

            # Get action
            # Run denoising process
            obs_n = obs_n.reshape(1, -1)  # Reshape to match
            final_action, trajectory = inf._denoise_actions(obs_n)
            deltas_normalized = final_action.cpu().detach().numpy().squeeze()[:, :3]  # Include z-axis
            print("deltas_normalized", deltas_normalized)

            # Post-process results
            executable_actions, full_trajectory = inf._postprocess_actions(final_action, trajectory)
            deltas = executable_actions[:, :3]  # Include z-axis
            print("deltas", deltas)

            current_ee_pos = env.get_observation()[:inf.obs_ee_dim]
            print("current_ee_pos", current_ee_pos)
            target_position = current_ee_pos[:3].copy()  # Include z-axis
            target_positions = []
            for d in deltas:
                target_position += d
                target_positions.append(target_position.copy())
            target_positions = np.array(target_positions)

            # Create 2x2 subplot layout with 3D plots
            fig = plt.figure(figsize=(20, 12))

            # Original DLO and target in 3D (top-left)
            ax1 = fig.add_subplot(2, 2, 1, projection='3d')
            ax1.plot(dlo_state[:, 0], dlo_state[:, 1], dlo_state[:, 2], "o-", label="DLO original", color='blue', linewidth=2, markersize=6)
            ax1.plot(target_shape[:, 0], target_shape[:, 1], target_shape[:, 2], "o-", label="Target original", color='red', linewidth=2, markersize=6)
            ax1.set_title("Original DLO and Target (3D)")
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.set_zlabel("Z")
            ax1.legend()
            ax1.grid(True)
            
            # Set equal aspect ratio and limits
            ax1.set_xlim(0.4, 0.6)
            ax1.set_ylim(-0.1, 0.1)
            ax1.set_zlim(0.6, 0.8)

            # Normalized DLO and target in 3D (top-right)
            ax2 = fig.add_subplot(2, 2, 2, projection='3d')
            ax2.plot(dlo_state_n[:, 0], dlo_state_n[:, 1], dlo_state_n[:, 2], "o-", label="DLO normalized", color='blue', linewidth=2, markersize=6)
            ax2.plot(target_shape_n[:, 0], target_shape_n[:, 1], target_shape_n[:, 2], "o-", label="Target normalized", color='red', linewidth=2, markersize=6)
            ax2.set_title("Normalized DLO and Target (3D)")
            ax2.set_xlabel("X")
            ax2.set_ylabel("Y")
            ax2.set_zlabel("Z")
            ax2.legend()
            ax2.grid(True)
            
            # Set equal aspect ratio and limits
            ax2.set_xlim(-0.1, 0.1)
            ax2.set_ylim(-0.1, 0.1)
            ax2.set_zlim(-0.1, 0.1)

            # Delta Action Positions in 3D (bottom-left)
            ax3 = fig.add_subplot(2, 2, 3, projection='3d')
            ax3.plot(deltas[:, 0], deltas[:, 1], deltas[:, 2], 'o-', 
                   label='Delta Actions', color='green', linewidth=3, markersize=8)
            ax3.set_title("Delta Action Positions (3D)")
            ax3.set_xlabel("X")
            ax3.set_ylabel("Y")
            ax3.set_zlabel("Z")
            ax3.legend()
            ax3.grid(True)

            ax3.set_xlim(-0.2, 0.2)
            ax3.set_ylim(-0.2, 0.2)
            ax3.set_zlim(-0.2, 0.2)
            
            # Add step numbers on points
            for i, (x, y, z) in enumerate(deltas):
                ax3.text(x, y, z, f'{i+1}', fontsize=10, color='darkgreen')

            # Final Action Positions in 3D (bottom-right)
            ax4 = fig.add_subplot(2, 2, 4, projection='3d')
            ax4.plot(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2], 'o-', 
                   label='Final Positions', color='purple', linewidth=3, markersize=8)
            ax4.set_title("Final Action Positions (3D)")
            ax4.set_xlabel("X")
            ax4.set_ylabel("Y")
            ax4.set_zlabel("Z")
            ax4.legend()
            ax4.grid(True)
            

            ax4.set_xlim(-0.5, 0.5)
            ax4.set_ylim(-0.5, 0.5)
            ax4.set_zlim(0.5, 1.5)


            # Add step numbers on points
            for i, (x, y, z) in enumerate(target_positions):
                ax4.text(x, y, z, f'{i+1}', fontsize=10, color='darkmagenta')

            print("final_positions", target_positions)

            plt.tight_layout()
            plt.suptitle(f"Episode {episode+1}, Action {action+1}", fontsize=16, y=0.98)
            plt.show()


if __name__ == "__main__":
    main()