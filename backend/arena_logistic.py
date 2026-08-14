import argparse
import asyncio
import os
import math
import time
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
            popt, pcov = fit_params
            k, x0 = popt
            cse = np.exp(x0)
            x_line = np.linspace(min(x_data) - 0.5, max(x_data) + 0.5, 100)
            y_line = sigmoid(x_line, k, x0)
            sims_line = np.exp(x_line)
            plotext.plot(sims_line.tolist(), y_line.tolist(), color="green", label="Fit")
            plotext.scatter([cse], [0.5], color="yellow", marker="x", label="CSE")
            
            if pcov is not None and not np.isinf(pcov[1, 1]) and pcov[1, 1] >= 0:
                se = np.sqrt(pcov[1, 1])
                lower_bound = np.exp(x0 - 1.96 * se)
                upper_bound = np.exp(x0 + 1.96 * se)
                plotext.title(f"Logistic CSE (Current CSE: {cse:.1f} | 95% CI: [{lower_bound:.1f}, {upper_bound:.1f}])")
            else:
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
    fit_params=None,
    input_channels=3,
    phase_str=None
):
    muzero_wins = 0
    total_games = pairs_per_match * 2
    stop_event = asyncio.Event()
    game_count = 0
    
    muzero_time_total = 0.0
    muzero_moves = 0
    classic_time_total = 0.0
    classic_moves = 0

    anchor_header = f"{anchor_idx+1}/{total_anchors} ({classic_sims} Sims)"
    if phase_str:
        anchor_header = f"[{phase_str}] {anchor_header}"

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
                "Anchor": anchor_header,
                "Game": f"{game_count}/{total_games} (MuZero=Red)",
                "Move": f"#{move_num}",
                "Turn": "MuZero (Red)" if env_a.current_player == 1 else "Classic (Blue)",
                "Match Score": f"MuZero {muzero_wins} - {game_count - 1 - muzero_wins} Classic"
            }
            update_dashboard(sims_data, win_rates, env_a, board_size, info, fit_params)

            if env_a.current_player == 1:
                obs = env_a.get_observation(v5_features=(input_channels == 5))
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                t0 = time.time()
                root = await latent_mcts_a.search(
                    initial_state_tensor=obs_tensor,
                    legal_actions=legal,
                    num_simulations=muzero_sims,
                    stop_event=stop_event
                )
                best_act = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
                t1 = time.time()
                muzero_time_total += (t1 - t0)
                muzero_moves += 1
                env_a.step(best_act)
            else:
                t0 = time.time()
                best_act, _ = classic_mcts_a.search(env_a, num_simulations=classic_sims)
                t1 = time.time()
                classic_time_total += (t1 - t0)
                classic_moves += 1
                env_a.step(best_act)

        if env_a.winner == 1:
            muzero_wins += 1

        info = {
            "Anchor": anchor_header,
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
                "Anchor": anchor_header,
                "Game": f"{game_count}/{total_games} (MuZero=Blue)",
                "Move": f"#{move_num}",
                "Turn": "Classic (Red)" if env_b.current_player == 1 else "MuZero (Blue)",
                "Match Score": f"MuZero {muzero_wins} - {game_count - 1 - muzero_wins} Classic"
            }
            update_dashboard(sims_data, win_rates, env_b, board_size, info, fit_params)

            if env_b.current_player == 1:
                t0 = time.time()
                best_act, _ = classic_mcts_b.search(env_b, num_simulations=classic_sims)
                t1 = time.time()
                classic_time_total += (t1 - t0)
                classic_moves += 1
                env_b.step(best_act)
            else:
                obs = env_b.get_observation(v5_features=(input_channels == 5))
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                t0 = time.time()
                root = await latent_mcts_b.search(
                    initial_state_tensor=obs_tensor,
                    legal_actions=legal,
                    num_simulations=muzero_sims,
                    stop_event=stop_event
                )
                best_act = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
                t1 = time.time()
                muzero_time_total += (t1 - t0)
                muzero_moves += 1
                env_b.step(best_act)

        if env_b.winner == 2:
            muzero_wins += 1

        info = {
            "Anchor": anchor_header,
            "Game": f"{game_count}/{total_games} (MuZero=Blue)",
            "Status": f"FINISHED ({'MuZero Won' if env_b.winner==2 else 'Classic Won'})",
            "Match Score": f"MuZero {muzero_wins} - {game_count - muzero_wins} Classic"
        }
        update_dashboard(sims_data, win_rates, env_b, board_size, info, fit_params)

    win_rate = muzero_wins / total_games
    return win_rate, muzero_wins, total_games, muzero_time_total, muzero_moves, classic_time_total, classic_moves

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default="v4_clone", help="Run ID for versioned weights")
    parser.add_argument("--board-size", type=int, default=7)
    parser.add_argument("--muzero-sims", type=int, default=400)
    parser.add_argument("--classic-anchors", type=str, default="250,500,1000,2500,5000,10000")
    parser.add_argument("--games-per-anchor", type=int, default=10)
    parser.add_argument("--use-fcn", action="store_true", help="Use Fully Convolutional Prediction Head")
    parser.add_argument("--input-channels", type=int, default=3, help="Number of input observation channels")
    parser.add_argument("--adaptive", action="store_true", help="Enable two-phase adaptive bracket hunting and concentrated sampling")
    parser.add_argument("--start-sims", type=int, default=5000, help="Initial classic simulation anchor for adaptive hunt")
    parser.add_argument("--min-sims", type=int, default=500, help="Minimum simulation floor for adaptive hunt")
    parser.add_argument("--max-sims", type=int, default=100000, help="Maximum simulation ceiling for adaptive hunt")
    parser.add_argument("--bracket-games", type=int, default=4, help="Number of games (paired Red/Blue) per step during Phase 1 hunt")
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
            num_res_blocks=num_res_blocks,
            input_channels=args.input_channels,
            use_fcn=args.use_fcn
        )
        model.load_state_dict(saved_weights)
        args.board_size = board_size
    else:
        model = MuZeroModels(
            board_size=board_size,
            action_space_size=action_space_size,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            input_channels=args.input_channels,
            use_fcn=args.use_fcn
        )
    model.eval()

    sims_data = []
    win_rates = []
    match_records = []
    fit_params = None
    
    total_muzero_time = 0.0
    total_muzero_moves = 0
    total_classic_time = 0.0
    total_classic_sims_completed = 0

    if args.adaptive:
        # Phase 1: Bracket Hunting (Geometric Stride with Early Exit)
        sims = args.start_sims
        phase1_step = 0
        bracket_pairs = max(1, args.bracket_games // 2)

        while True:
            phase1_step += 1
            win_rate, muz_w, tot_g, muz_t, muz_m, clas_t, clas_m = await play_match_interactive(
                board_size=args.board_size,
                muzero_sims=args.muzero_sims,
                classic_sims=sims,
                pairs_per_match=bracket_pairs,
                model=model,
                anchor_idx=phase1_step - 1,
                total_anchors="Hunt",
                sims_data=sims_data,
                win_rates=win_rates,
                fit_params=fit_params,
                input_channels=args.input_channels,
                phase_str="Phase 1: Hunt"
            )

            sims_data.append(sims)
            win_rates.append(win_rate)
            match_records.append({
                "phase": "Phase 1: Hunt",
                "anchor": sims,
                "muzero_wins": muz_w,
                "classic_wins": tot_g - muz_w,
                "total_games": tot_g,
                "win_rate": win_rate
            })

            total_muzero_time += muz_t
            total_muzero_moves += muz_m
            total_classic_time += clas_t
            total_classic_sims_completed += (clas_m * sims)

            if len(sims_data) >= 2:
                try:
                    x_data = np.log(sims_data)
                    y_data = np.array(win_rates)
                    popt, pcov = curve_fit(sigmoid, x_data, y_data, p0=[-1.0, np.mean(x_data)], maxfev=10000)
                    fit_params = (popt, pcov)
                except Exception:
                    fit_params = None

            p = win_rate
            has_high = any(w > 0.5 for w in win_rates)
            has_low = any(w < 0.5 for w in win_rates)

            # Early exit / termination criteria
            if (0.25 < p < 0.75) or (has_high and has_low):
                break

            if p >= 0.75:
                if sims >= args.max_sims:
                    break
                next_sims = min(args.max_sims, int(sims * 2.5))
                if next_sims == sims or next_sims in sims_data:
                    break
                sims = next_sims
            elif p <= 0.25:
                if sims <= args.min_sims:
                    break
                next_sims = max(args.min_sims, int(sims * 0.4))
                if next_sims == sims or next_sims in sims_data:
                    break
                sims = next_sims

        # Phase 2: Focused Grid Sampling
        # Determine [N_low, N_high] bracket that straddles parity
        tested_pairs = list(zip(sims_data, win_rates))
        high_win_sims = [s for s, w in tested_pairs if w >= 0.5]
        low_win_sims = [s for s, w in tested_pairs if w <= 0.5]

        if high_win_sims and low_win_sims:
            n_low = max(high_win_sims)
            n_high = min(low_win_sims)
            if n_low > n_high:
                n_low, n_high = n_high, n_low
            elif n_low == n_high:
                n_low = max(args.min_sims, int(n_low * 0.5))
                n_high = min(args.max_sims, int(n_high * 2.0))
        elif high_win_sims:
            n_low = max(high_win_sims)
            n_high = args.max_sims
            if n_low >= n_high:
                n_low = max(args.min_sims, int(n_high * 0.5))
        elif low_win_sims:
            n_low = args.min_sims
            n_high = min(low_win_sims)
            if n_low >= n_high:
                n_high = min(args.max_sims, int(n_low * 2.0))
        else:
            n_low = args.min_sims
            n_high = args.max_sims

        if n_low >= n_high:
            n_low = max(args.min_sims, int(n_high * 0.5))
            n_high = min(args.max_sims, int(n_low * 2.0))
            if n_low >= n_high:
                n_high = n_low + 100

        focused_anchors = sorted(list(set(int(round(x)) for x in np.geomspace(n_low, n_high, num=5))))
        if len(focused_anchors) < 4:
            focused_anchors = sorted(list(set(int(round(x)) for x in np.linspace(n_low, n_high, num=5))))

        pairs_per_anchor = max(1, args.games_per_anchor // 2)
        total_focused = len(focused_anchors)

        for idx, anchor in enumerate(focused_anchors):
            win_rate, muz_w, tot_g, muz_t, muz_m, clas_t, clas_m = await play_match_interactive(
                board_size=args.board_size,
                muzero_sims=args.muzero_sims,
                classic_sims=anchor,
                pairs_per_match=pairs_per_anchor,
                model=model,
                anchor_idx=idx,
                total_anchors=total_focused,
                sims_data=sims_data,
                win_rates=win_rates,
                fit_params=fit_params,
                input_channels=args.input_channels,
                phase_str="Phase 2: Focused"
            )

            sims_data.append(anchor)
            win_rates.append(win_rate)
            match_records.append({
                "phase": "Phase 2: Focused",
                "anchor": anchor,
                "muzero_wins": muz_w,
                "classic_wins": tot_g - muz_w,
                "total_games": tot_g,
                "win_rate": win_rate
            })

            total_muzero_time += muz_t
            total_muzero_moves += muz_m
            total_classic_time += clas_t
            total_classic_sims_completed += (clas_m * anchor)

            if len(sims_data) >= 2:
                try:
                    x_data = np.log(sims_data)
                    y_data = np.array(win_rates)
                    popt, pcov = curve_fit(sigmoid, x_data, y_data, p0=[-1.0, np.mean(x_data)], maxfev=10000)
                    fit_params = (popt, pcov)
                except Exception:
                    fit_params = None

    else:
        # Fixed Anchors Mode
        anchors = [int(x) for x in args.classic_anchors.split(',')]
        pairs_per_anchor = max(1, args.games_per_anchor // 2)

        for idx, anchor in enumerate(anchors):
            win_rate, muz_w, tot_g, muz_t, muz_m, clas_t, clas_m = await play_match_interactive(
                board_size=args.board_size,
                muzero_sims=args.muzero_sims,
                classic_sims=anchor,
                pairs_per_match=pairs_per_anchor,
                model=model,
                anchor_idx=idx,
                total_anchors=len(anchors),
                sims_data=sims_data,
                win_rates=win_rates,
                fit_params=fit_params,
                input_channels=args.input_channels,
                phase_str="Fixed Anchors"
            )
            
            sims_data.append(anchor)
            win_rates.append(win_rate)
            match_records.append({
                "phase": "Fixed Anchors",
                "anchor": anchor,
                "muzero_wins": muz_w,
                "classic_wins": tot_g - muz_w,
                "total_games": tot_g,
                "win_rate": win_rate
            })
            
            total_muzero_time += muz_t
            total_muzero_moves += muz_m
            total_classic_time += clas_t
            total_classic_sims_completed += (clas_m * anchor)

            # Update fit if we have enough points
            if len(sims_data) >= 2:
                try:
                    x_data = np.log(sims_data)
                    y_data = np.array(win_rates)
                    popt, pcov = curve_fit(sigmoid, x_data, y_data, p0=[-1.0, np.mean(x_data)], maxfev=10000)
                    fit_params = (popt, pcov)
                except Exception:
                    fit_params = None

    # Final summary display
    print(f"\n====================================")
    print(f"📋 MATCH & ROUND WIN/LOSS RECORDS")
    print(f"{'Phase':<18} {'Anchor':<14} {'Score (M - C)':<16} {'Win Rate':<10} {'Games':<8}")
    print(f"───────────────────────────────────────────────────────────────────")
    total_m_wins = 0
    total_c_wins = 0
    total_g_all = 0
    for r in match_records:
        total_m_wins += r["muzero_wins"]
        total_c_wins += r["classic_wins"]
        total_g_all += r["total_games"]
        p_str = f"{r['phase']}"
        a_str = f"{r['anchor']} sims"
        s_str = f"{r['muzero_wins']}W - {r['classic_wins']}L"
        w_str = f"{r['win_rate']*100:.1f}%"
        g_str = f"{r['total_games']}"
        print(f"{p_str:<18} {a_str:<14} {s_str:<16} {w_str:<10} {g_str:<8}")
    print(f"───────────────────────────────────────────────────────────────────")
    overall_wr = (total_m_wins / max(1, total_g_all)) * 100
    print(f"{'TOTAL':<18} {'All':<14} {f'{total_m_wins}W - {total_c_wins}L':<16} {f'{overall_wr:.1f}%':<10} {total_g_all:<8}")
    print(f"====================================")
    
    # Timing Stats
    t_muzero = total_muzero_time / max(1, total_muzero_moves)
    t_classic_sim = total_classic_time / max(1, total_classic_sims_completed)
    crossover = t_muzero / max(1e-9, t_classic_sim)
    
    print(f"⏱️ MuZero ({args.muzero_sims} sims): {t_muzero:.4f} sec/move")
    print(f"⏱️ Classic MCTS: {t_classic_sim*1000:.4f} ms/simulation")
    print(f"⚖️ Wall-Clock Crossover (N_time): {crossover:.1f} classic sims")
    
    if fit_params is not None:
        popt, pcov = fit_params
        cse = np.exp(popt[1])
        if pcov is not None and not np.isinf(pcov[1, 1]) and pcov[1, 1] >= 0:
            se = np.sqrt(pcov[1, 1])
            lower_bound = np.exp(popt[1] - 1.96 * se)
            upper_bound = np.exp(popt[1] + 1.96 * se)
            print(f"\n📊 SIMULATION CSE (N_50): {cse:.1f} (95% CI: [{lower_bound:.1f}, {upper_bound:.1f}])")
        else:
            print(f"\n📊 SIMULATION CSE (N_50): {cse:.1f} (95% CI: [N/A, N/A])")
        
        speedup = (cse * t_classic_sim) / max(1e-9, t_muzero)
        print(f"🚀 WALL-CLOCK SPEEDUP FACTOR: {speedup:.2f}x")
    else:
        # Check if saturated
        if np.mean(win_rates) < 0.1:
            print(f"\n⚠️ WARNING: The model win rate remained near 0% across all anchors.")
            print(f"It saturated the search ceiling. Could not fit logistic curve.")
        elif np.mean(win_rates) > 0.9:
            print(f"\n⚠️ WARNING: The model win rate remained near 100% across all anchors.")
            print(f"It completely dominates the classic anchor space. Could not fit logistic curve.")
        else:
            print(f"\n⚠️ Could not fit a reliable logistic curve.")

    print(f"====================================\n")

if __name__ == "__main__":
    asyncio.run(main())
