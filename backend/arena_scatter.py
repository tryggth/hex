import argparse
import asyncio
import os
import torch
import plotext
import random
import collections
import math
from backend.muzero_nets import MuZeroModels
from backend.arena import play_match

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-size", type=int, default=7)
    parser.add_argument("--muzero-sims", type=int, default=400)
    parser.add_argument("--pairs-per-match", type=int, default=5)
    parser.add_argument("--min-sims", type=int, default=1)
    parser.add_argument("--max-sims", type=int, default=25)
    args = parser.parse_args()

    board_size = args.board_size
    action_space_size = board_size ** 2
    latent_channels = 96
    num_res_blocks = 8

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
        args.board_size = board_size
    else:
        model = MuZeroModels(
            board_size=board_size,
            action_space_size=action_space_size,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks
        )
    model.eval()

    # Dictionary to map Classic Sims -> List of Classic Win Rates
    results = collections.defaultdict(list)
    total_matches = 0

    print("Starting continuous random sampling arena. Press Ctrl+C to stop.")

    try:
        while True:
            current_classic_sims = random.randint(args.min_sims, args.max_sims)
            
            muzero_win_rate = await play_match(
                board_size=args.board_size,
                muzero_sims=args.muzero_sims,
                classic_sims=current_classic_sims,
                pairs_per_match=args.pairs_per_match,
                model=model
            )

            # Convert to Classic MCTS win rate
            classic_win_rate = 1.0 - muzero_win_rate
            results[current_classic_sims].append(classic_win_rate)
            total_matches += 1

            # Aggregate data for plotting
            x_data = sorted(results.keys())
            y_data = [sum(results[x]) / len(results[x]) for x in x_data]

            plotext.clear_terminal()
            plotext.clear_data()
            plotext.scatter(x_data, y_data, color="red")
            plotext.plot([args.min_sims, args.max_sims], [0.5, 0.5], color="green")
            plotext.title("Simulation Parity: Classic MCTS Win Rate vs Sims")
            plotext.xlabel("Classic MCTS Simulations")
            plotext.ylabel("Classic MCTS Win Rate")
            plotext.show()

            print(f"\nTotal Matches Played: {total_matches}")
            print(f"Just Tested: {current_classic_sims} sims | Match Classic Win Rate: {classic_win_rate*100:.1f}%")
            avg_win_rate = sum(results[current_classic_sims]) / len(results[current_classic_sims])
            print(f"Overall Average for {current_classic_sims} sims: {avg_win_rate*100:.1f}% (across {len(results[current_classic_sims])} matches)")

    except KeyboardInterrupt:
        print("\nArena stopped by user. Final results:")
        for x in sorted(results.keys()):
            avg = sum(results[x]) / len(results[x])
            print(f"{x:3d} sims: {avg*100:5.1f}% win rate ({len(results[x])} matches)")

if __name__ == "__main__":
    asyncio.run(main())
