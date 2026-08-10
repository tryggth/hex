import argparse
import asyncio
import os
import math
import numpy as np
import torch
import plotext
from scipy.optimize import curve_fit
from backend.muzero_nets import MuZeroModels
from backend.arena import play_match

def sigmoid(x, k, x0):
    """
    Logistic function (Sigmoid).
    x: independent variable (log of classic sims)
    k: steepness (expected to be negative since MuZero win rate drops as Classic sims increase)
    x0: the midpoint where y = 0.50
    """
    return 1.0 / (1.0 + np.exp(np.clip(-k * (x - x0), -500, 500)))

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-size", type=int, default=7)
    parser.add_argument("--muzero-sims", type=int, default=400)
    parser.add_argument("--pairs-per-anchor", type=int, default=10, help="20 games per anchor (10 pairs)")
    parser.add_argument("--anchors", type=int, nargs="+", default=[1, 5, 10, 25, 50, 100, 200, 400])
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

    print("Starting Logistic Regression CSE Evaluation")
    print(f"Anchors to test (Classic Sims): {args.anchors}")
    print(f"Games per anchor: {args.pairs_per_anchor * 2}")

    sims_data = []
    win_rates = []

    for anchor in sorted(args.anchors):
        print(f"\nEvaluating Anchor: {anchor} Classic Sims...")
        muzero_win_rate = await play_match(
            board_size=args.board_size,
            muzero_sims=args.muzero_sims,
            classic_sims=anchor,
            pairs_per_match=args.pairs_per_anchor,
            model=model
        )
        print(f"Result: {muzero_win_rate*100:.1f}% MuZero Win Rate")
        
        sims_data.append(anchor)
        win_rates.append(muzero_win_rate)

    print("\nFitting Logistic Curve...")
    
    # x is natural log of sims, y is win rate
    x_data = np.log(sims_data)
    y_data = np.array(win_rates)
    
    try:
        # Initial guess: k=-1.0 (win rate drops), x0=mean(x_data)
        popt, pcov = curve_fit(sigmoid, x_data, y_data, p0=[-1.0, np.mean(x_data)])
        k, x0 = popt
        
        cse = np.exp(x0)
        
        print(f"\n====================================")
        print(f"📊 CSE (Simulation Parity): {cse:.1f}")
        print(f"====================================\n")
        
        # Plotting
        plotext.clear_terminal()
        plotext.clear_data()
        
        x_line = np.linspace(min(x_data) - 0.5, max(x_data) + 0.5, 100)
        y_line = sigmoid(x_line, k, x0)
        sims_line = np.exp(x_line)
        
        plotext.scatter(sims_data, win_rates, color="cyan", label="Raw Anchors")
        plotext.plot(sims_line.tolist(), y_line.tolist(), color="green", label="Fitted Sigmoid Curve")
        plotext.plot([min(sims_data), max(sims_data)], [0.5, 0.5], color="red", label="Parity (0.50)")
        
        # Plot intercept
        plotext.scatter([cse], [0.5], color="yellow", marker="x", label="Calculated CSE")
        
        plotext.title(f"Logistic CSE Evaluation (CSE: {cse:.1f})")
        plotext.xlabel("Classic MCTS Simulations (Log Scale)")
        plotext.ylabel("MuZero Win Rate")
        plotext.xscale("log")
        plotext.show()

    except Exception as e:
        print(f"\nCurve fitting failed: {e}")
        print("This typically happens if the data is entirely flat (e.g. 0% win rate everywhere or 100% everywhere).")
        print("Raw Data:")
        for s, w in zip(sims_data, win_rates):
            print(f"{s:3d} sims: {w*100:.1f}%")

if __name__ == "__main__":
    asyncio.run(main())
