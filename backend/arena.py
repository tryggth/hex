import argparse
import asyncio
import os
import torch
import plotext
from backend.hex_env import HexEnv
from backend.muzero_nets import MuZeroModels
from backend.latent_mcts import LatentMCTS
from backend.classic_mcts import ClassicMCTS

async def play_match(board_size, muzero_sims, classic_sims, pairs_per_match, model):
    muzero_wins = 0
    total_games = pairs_per_match * 2
    stop_event = asyncio.Event()

    for pair_idx in range(pairs_per_match):
        # Game A: MuZero=Red (1), Classic=Blue (2)
        env_a = HexEnv(board_size=board_size)
        latent_mcts_a = LatentMCTS(model=model)
        classic_mcts_a = ClassicMCTS()

        while env_a.winner == 0:
            legal = env_a.legal_actions()
            if not legal:
                break
            if env_a.current_player == 1:
                obs = env_a.get_observation()
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                root = await latent_mcts_a.search(
                    initial_state_tensor=obs_tensor,
                    legal_actions=legal,
                    num_simulations=muzero_sims,
                    stop_event=stop_event
                )
                best_act = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
                env_a.step(best_act)
            else:
                best_act, _ = classic_mcts_a.search(env_a, num_simulations=classic_sims)
                env_a.step(best_act)
        if env_a.winner == 1:
            muzero_wins += 1

        # Game B: Classic=Red (1), MuZero=Blue (2)
        env_b = HexEnv(board_size=board_size)
        latent_mcts_b = LatentMCTS(model=model)
        classic_mcts_b = ClassicMCTS()

        while env_b.winner == 0:
            legal = env_b.legal_actions()
            if not legal:
                break
            if env_b.current_player == 1:
                best_act, _ = classic_mcts_b.search(env_b, num_simulations=classic_sims)
                env_b.step(best_act)
            else:
                obs = env_b.get_observation()
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                root = await latent_mcts_b.search(
                    initial_state_tensor=obs_tensor,
                    legal_actions=legal,
                    num_simulations=muzero_sims,
                    stop_event=stop_event
                )
                best_act = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
                env_b.step(best_act)
        if env_b.winner == 2:
            muzero_wins += 1

    return muzero_wins / total_games

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-size", type=int, default=7)
    parser.add_argument("--muzero-sims", type=int, default=400)
    parser.add_argument("--pairs-per-match", type=int, default=5)
    parser.add_argument("--min-sims", type=int, default=10)
    parser.add_argument("--max-sims", type=int, default=20000)
    parser.add_argument("--tolerance", type=int, default=200)
    args = parser.parse_args()

    model = MuZeroModels(
        board_size=args.board_size,
        action_space_size=args.board_size ** 2,
        latent_channels=96,
        num_res_blocks=8
    )
    weights_path = os.path.join(os.path.dirname(__file__), "model_weights.pth")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    tested_sims = []
    win_rates = []

    low = args.min_sims
    high = args.max_sims
    current_classic_sims = args.min_sims

    while (high - low) > args.tolerance:
        current_classic_sims = int((low + high) / 2)
        
        win_rate = await play_match(
            board_size=args.board_size,
            muzero_sims=args.muzero_sims,
            classic_sims=current_classic_sims,
            pairs_per_match=args.pairs_per_match,
            model=model
        )

        tested_sims.append(current_classic_sims)
        win_rates.append(win_rate)

        plotext.clear_terminal()
        plotext.clear_data()
        plotext.scatter(tested_sims, win_rates, color="cyan")
        plotext.plot([args.min_sims, args.max_sims], [0.5, 0.5], color="red")
        plotext.title("Simulation Parity: MuZero vs Classic MCTS")
        plotext.xlabel("Classic MCTS Simulations")
        plotext.ylabel("MuZero Win Rate")
        plotext.show()

        print(f"\nBracket: [{low}, {high}]")
        print(f"Tested Classic Sims: {current_classic_sims}")
        print(f"MuZero Win Rate: {win_rate*100:.1f}%")

        if win_rate > 0.50:
            low = current_classic_sims
        elif win_rate < 0.50:
            high = current_classic_sims
        else:
            break

    ratio = current_classic_sims / args.muzero_sims
    print(f"\nThe MuZero network ({args.muzero_sims} sims) is mathematically equivalent to a Classic MCTS running at {current_classic_sims} simulations per move. Compute Compression Ratio: {ratio:.1f}x")

if __name__ == "__main__":
    asyncio.run(main())
