import argparse
import asyncio
import os
import math
import torch
import plotext
from backend.hex_env import HexEnv
from backend.muzero_nets import MuZeroModels
from backend.latent_mcts import LatentMCTS
from backend.classic_mcts import ClassicMCTS

async def play_single_game(board_size, muzero_sims, classic_sims, model, muzero_color):
    env = HexEnv(board_size=board_size)
    latent_mcts = LatentMCTS(model=model)
    classic_mcts = ClassicMCTS()
    stop_event = asyncio.Event()

    while env.winner == 0:
        legal = env.legal_actions()
        if not legal:
            break
        if env.current_player == muzero_color:
            obs = env.get_observation()
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            root = await latent_mcts.search(
                initial_state_tensor=obs_tensor,
                legal_actions=legal,
                num_simulations=muzero_sims,
                stop_event=stop_event
            )
            best_act = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
            env.step(best_act)
        else:
            best_act, _ = classic_mcts.search(env, num_simulations=classic_sims)
            env.step(best_act)
            
    return 1 if env.winner == muzero_color else 0

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-size", type=int, default=7)
    parser.add_argument("--muzero-sims", type=int, default=400)
    parser.add_argument("--run-id", type=str, default=None, help="Run ID for versioned weights")
    args = parser.parse_args()

    board_size = args.board_size
    action_space_size = board_size ** 2
    latent_channels = 96
    num_res_blocks = 8

    if args.run_id:
        weights_path = os.path.join(os.path.dirname(__file__), "runs", args.run_id, "model_weights.pth")
    else:
        weights_path = os.path.join(os.path.dirname(__file__), "model_weights.pth")
    if os.path.exists(weights_path):
        saved_weights = torch.load(weights_path, map_location="cpu")
        if "prediction.policy_fc.weight" in saved_weights:
            action_space_size = saved_weights["prediction.policy_fc.weight"].shape[0]
            board_size = int(math.sqrt(action_space_size))
        if "representation.conv_init.weight" in saved_weights:
            latent_channels = saved_weights["representation.conv_init.weight"].shape[0]
            num_res_blocks = len([k for k in saved_weights.keys() if "representation.res_blocks" in k and "conv1.weight" in k])
        
        model = MuZeroModels(
            board_size=board_size,
            action_space_size=action_space_size,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks
        )
        model.load_state_dict(saved_weights)
    else:
        model = MuZeroModels(
            board_size=board_size,
            action_space_size=action_space_size,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks
        )
    model.eval()

    # SPRT Parameters
    p0 = 0.50
    p1 = 0.55
    alpha = 0.05
    beta = 0.05
    
    A = math.log((1 - beta) / alpha)
    B = math.log(beta / (1 - alpha))
    
    llr_win = math.log(p1 / p0)
    llr_loss = math.log((1 - p1) / (1 - p0))

    llr = 0.0
    games_played = 0
    muzero_wins = 0

    llr_history = []
    game_history = []

    print("Starting SPRT Arena (MuZero vs Random). Press Ctrl+C to abort early.")
    print(f"H0: MuZero Win Rate <= {p0*100:.0f}%")
    print(f"H1: MuZero Win Rate >= {p1*100:.0f}%")
    print(f"Lower Bound (Accept H0): {B:.3f}")
    print(f"Upper Bound (Accept H1): {A:.3f}\n")

    try:
        while True:
            # Alternate first-mover advantage
            muzero_color = 1 if games_played % 2 == 0 else 2
            
            win = await play_single_game(
                board_size=board_size,
                muzero_sims=args.muzero_sims,
                classic_sims=1,
                model=model,
                muzero_color=muzero_color
            )
            
            muzero_wins += win
            games_played += 1
            
            if win == 1:
                llr += llr_win
            else:
                llr += llr_loss
                
            game_history.append(games_played)
            llr_history.append(llr)

            plotext.clear_terminal()
            plotext.clear_data()
            plotext.scatter(game_history, llr_history, color="cyan", marker="dot")
            plotext.plot([0, max(10, games_played)], [A, A], color="green")
            plotext.plot([0, max(10, games_played)], [B, B], color="red")
            plotext.title(f"SPRT LLR Trace (Games: {games_played}, Wins: {muzero_wins})")
            plotext.xlabel("Games Played")
            plotext.ylabel("Log-Likelihood Ratio (LLR)")
            plotext.show()

            print(f"\nGames: {games_played} | MuZero Wins: {muzero_wins} | Win Rate: {(muzero_wins/games_played)*100:.1f}%")
            print(f"Current LLR: {llr:.3f} | Bounds: [{B:.3f}, {A:.3f}]")

            if llr >= A:
                print(f"\n[SPRT COMPLETE] H1 ACCEPTED!")
                print(f"Mathematical Proof: Latent MCTS is statistically SIGNIFICANTLY BETTER than chance (>{p1*100}% win rate).")
                break
            elif llr <= B:
                print(f"\n[SPRT COMPLETE] H0 ACCEPTED!")
                print(f"Mathematical Proof: Latent MCTS is performing NO BETTER than chance (random play).")
                break

    except KeyboardInterrupt:
        print("\nSPRT stopped early by user.")

if __name__ == "__main__":
    asyncio.run(main())
