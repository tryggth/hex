import argparse
import asyncio
import os
import math
import numpy as np
import torch
import plotext
from scipy.optimize import curve_fit
from backend.hex_env import HexEnv
from backend.muzero_nets import MuZeroModels
from backend.latent_mcts import LatentMCTS
from backend.classic_mcts import ClassicMCTS

def sigmoid(x, k, x0):
    """
    Logistic function (Sigmoid).
    x: independent variable (log of classic sims)
    k: steepness (expected to be negative since MuZero win rate drops as Classic sims increase)
    x0: the midpoint where y = 0.50
    """
    return 1.0 / (1.0 + np.exp(np.clip(-k * (x - x0), -500, 500)))

def render_board_lines(env, board_size, extra_info):
    lines = []
    lines.append("┌────────── REALTIME GAME STATUS ──────────┐")
    for k, v in extra_info.items():
        lines.append(f" {k:<14}: {v}")
    lines.append("───────────────────────────────────────────")
    lines.append(" LIVE BOARD STATE:")
    
    header = "    " + " ".join(str(c) for c in range(board_size))
    lines.append(header)
    
    for r in range(board_size):
        indent = " " * (r + 1)
        row_symbols = []
        for c in range(board_size):
            val = env.board[r * board_size + c]
            if val == 1:
                row_symbols.append("\033[91mR\033[0m")
            elif val == 2:
                row_symbols.append("\033[94mB\033[0m")
            elif r == 0 or r == board_size - 1:
                row_symbols.append("\033[91m.\033[0m")  # Red top/bottom boundary
            elif c == 0 or c == board_size - 1:
                row_symbols.append("\033[94m.\033[0m")  # Blue left/right boundary
            else:
                row_symbols.append(".")
                
        row_str = " ".join(row_symbols)
        lines.append(f" {r:1d}{indent}{row_str}")
        
    return lines

def update_dashboard(sims_data, win_rates, current_env, board_size, extra_info, fit_params=None):
    plotext.clear_terminal()
    plotext.clear_data()
    plotext.plotsize(46, 17)
    
    if len(sims_data) >= 2:
        x_data = np.log(sims_data)
        plotext.scatter(sims_data, win_rates, color="cyan", label="Anchors")
        plotext.plot([min(sims_data), max(sims_data)], [0.5, 0.5], color="red", label="Parity")
        
        if fit_params is not None:
            k, x0 = fit_params
            cse = np.exp(x0)
            x_line = np.linspace(min(x_data) - 0.5, max(x_data) + 0.5, 100)
            y_line = sigmoid(x_line, k, x0)
            sims_line = np.exp(x_line)
            plotext.plot(sims_line.tolist(), y_line.tolist(), color="green", label="Fit")
            plotext.scatter([cse], [0.5], color="yellow", marker="x", label="CSE")
            plotext.title(f"Logistic CSE (Current CSE: {cse:.1f})")
        else:
            plotext.title("Logistic CSE Evaluation")
        plotext.xscale("log")
    elif len(sims_data) == 1:
        plotext.scatter(sims_data, win_rates, color="cyan")
        plotext.title(f"Logistic CSE (1 Anchor Done)")
    else:
        plotext.title("Logistic CSE (Evaluating...)")

    plotext.xlabel("Classic Sims (Log)")
    plotext.ylabel("MuZero Win Rate")
    
    chart_lines = plotext.build().split("\n")
    board_lines = render_board_lines(current_env, board_size, extra_info)
    
    max_l = max(len(chart_lines), len(board_lines))
    
    output = []
    for i in range(max_l):
        c_line = chart_lines[i] if i < len(chart_lines) else " " * 46
        b_line = board_lines[i] if i < len(board_lines) else ""
        output.append(f"{c_line} │ {b_line}")
        
    print("\n".join(output))

async def play_match_interactive(
    board_size,
    muzero_sims,
    classic_sims,
    pairs_per_match,
    model,
    anchor_idx,
    total_anchors,
    sims_data,
    win_rates,
    fit_params=None
):
    muzero_wins = 0
    total_games = pairs_per_match * 2
    stop_event = asyncio.Event()
    game_count = 0

    for pair_idx in range(pairs_per_match):
        # --- Game A: MuZero = Red (1), Classic = Blue (2) ---
        game_count += 1
        env_a = HexEnv(board_size=board_size)
        latent_mcts_a = LatentMCTS(model=model)
        classic_mcts_a = ClassicMCTS()
        move_num = 0

        while env_a.winner == 0:
            move_num += 1
            legal = env_a.legal_actions()
            if not legal:
                break
                
            info = {
                "Anchor": f"{anchor_idx+1}/{total_anchors} ({classic_sims} Sims)",
                "Game": f"{game_count}/{total_games} (MuZero=Red)",
                "Move": f"#{move_num}",
                "Turn": "MuZero (Red)" if env_a.current_player == 1 else "Classic (Blue)",
                "Match Score": f"MuZero {muzero_wins} - {game_count - 1 - muzero_wins} Classic"
            }
            update_dashboard(sims_data, win_rates, env_a, board_size, info, fit_params)

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

        info = {
            "Anchor": f"{anchor_idx+1}/{total_anchors} ({classic_sims} Sims)",
            "Game": f"{game_count}/{total_games} (MuZero=Red)",
            "Status": f"FINISHED ({'MuZero Won' if env_a.winner==1 else 'Classic Won'})",
            "Match Score": f"MuZero {muzero_wins} - {game_count - muzero_wins} Classic"
        }
        update_dashboard(sims_data, win_rates, env_a, board_size, info, fit_params)

        # --- Game B: Classic = Red (1), MuZero = Blue (2) ---
        game_count += 1
        env_b = HexEnv(board_size=board_size)
        latent_mcts_b = LatentMCTS(model=model)
        classic_mcts_b = ClassicMCTS()
        move_num = 0

        while env_b.winner == 0:
            move_num += 1
            legal = env_b.legal_actions()
            if not legal:
                break
                
            info = {
                "Anchor": f"{anchor_idx+1}/{total_anchors} ({classic_sims} Sims)",
                "Game": f"{game_count}/{total_games} (MuZero=Blue)",
                "Move": f"#{move_num}",
                "Turn": "Classic (Red)" if env_b.current_player == 1 else "MuZero (Blue)",
                "Match Score": f"MuZero {muzero_wins} - {game_count - 1 - muzero_wins} Classic"
            }
            update_dashboard(sims_data, win_rates, env_b, board_size, info, fit_params)

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

        info = {
            "Anchor": f"{anchor_idx+1}/{total_anchors} ({classic_sims} Sims)",
            "Game": f"{game_count}/{total_games} (MuZero=Blue)",
            "Status": f"FINISHED ({'MuZero Won' if env_b.winner==2 else 'Classic Won'})",
            "Match Score": f"MuZero {muzero_wins} - {game_count - muzero_wins} Classic"
        }
        update_dashboard(sims_data, win_rates, env_b, board_size, info, fit_params)

    return muzero_wins / total_games

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-size", type=int, default=7)
    parser.add_argument("--muzero-sims", type=int, default=400)
    parser.add_argument("--pairs-per-anchor", type=int, default=10, help="20 games per anchor (10 pairs)")
    parser.add_argument("--anchors", type=int, nargs="+", default=[1, 5, 10, 25, 50, 100, 200, 400])
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
        args.board_size = board_size
    else:
        model = MuZeroModels(
            board_size=board_size,
            action_space_size=action_space_size,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks
        )
    model.eval()

    sims_data = []
    win_rates = []
    fit_params = None

    anchors = sorted(args.anchors)
    for idx, anchor in enumerate(anchors):
        win_rate = await play_match_interactive(
            board_size=args.board_size,
            muzero_sims=args.muzero_sims,
            classic_sims=anchor,
            pairs_per_match=args.pairs_per_anchor,
            model=model,
            anchor_idx=idx,
            total_anchors=len(anchors),
            sims_data=sims_data,
            win_rates=win_rates,
            fit_params=fit_params
        )
        
        sims_data.append(anchor)
        win_rates.append(win_rate)

        # Update fit if we have enough points
        if len(sims_data) >= 2:
            try:
                x_data = np.log(sims_data)
                y_data = np.array(win_rates)
                popt, _ = curve_fit(sigmoid, x_data, y_data, p0=[-1.0, np.mean(x_data)])
                fit_params = popt
            except Exception:
                fit_params = None

    # Final summary display
    if fit_params is not None:
        cse = np.exp(fit_params[1])
        print(f"\n====================================")
        print(f"📊 FINAL CALCULATED CSE: {cse:.1f}")
        print(f"====================================\n")

if __name__ == "__main__":
    asyncio.run(main())
