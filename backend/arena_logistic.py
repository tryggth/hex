import argparse
import asyncio
import os
import math
import time
import traceback
import numpy as np
import torch
import plotext
from scipy.optimize import minimize
from backend.hex_env import HexEnv
from backend.muzero_nets import MuZeroModels
from backend.latent_mcts import LatentMCTS
from backend.classic_mcts import ClassicMCTS

CRASH_DUMP_FILE = "arena_crash_dump.log"

def dump_crash_telemetry(exc: Exception, optimizer=None, extra_info: dict = None, filepath: str = CRASH_DUMP_FILE):
    """
    Captures full exception traceback, system metadata, and current optimizer/game state,
    writing a diagnostic dump to disk and printing it cleanly.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    report_lines = [
        "=" * 76,
        f"🚨 ARENA LOGISTIC CRASH DUMP: {timestamp}",
        "=" * 76,
        "",
        "── 1. EXCEPTION TRACEBACK ──",
        traceback.format_exc(),
        "",
        "── 2. SYSTEM & RUNTIME CONTEXT ──",
        f"  • Python/Platform: {os.sys.version}",
        f"  • PyTorch Version: {torch.__version__}",
        f"  • CUDA Available:  {torch.cuda.is_available()}",
        f"  • Current PID:     {os.getpid()}",
        f"  • Working Dir:     {os.getcwd()}",
    ]

    if extra_info:
        report_lines.append("")
        report_lines.append("── 3. RECENT GAME / STEP CONTEXT ──")
        for k, v in extra_info.items():
            report_lines.append(f"  • {k:<18}: {v}")

    if optimizer is not None:
        report_lines.append("")
        report_lines.append("── 4. OPTIMIZER STATE & ANCHOR DATA ──")
        report_lines.append(f"  • Is Bracketed:     {getattr(optimizer, 'is_bracketed', False)}")
        report_lines.append(f"  • Current Sims:     {getattr(optimizer, 'current_sims', 'N/A')}")
        report_lines.append(f"  • N_50 Estimate:    {getattr(optimizer, 'n50', 'N/A')}")
        report_lines.append(f"  • ln(N_50) (x0):    {getattr(optimizer, 'x0', 'N/A')}")
        report_lines.append(f"  • Standard Error:   {getattr(optimizer, 'se_x0', 'N/A')}")
        report_lines.append(f"  • 95% CI Bounds:    [{getattr(optimizer, 'ci_lower', 'N/A')}, {getattr(optimizer, 'ci_upper', 'N/A')}]")
        report_lines.append(f"  • 95% CI Ratio:     {getattr(optimizer, 'ci_ratio', 'N/A')}")
        report_lines.append(f"  • Beta0 / Beta1:    {getattr(optimizer, 'beta0', 'N/A')} / {getattr(optimizer, 'beta1', 'N/A')}")
        report_lines.append("")
        report_lines.append("  • Aggregated Anchor Data:")
        for s, d in getattr(optimizer, 'anchor_data', {}).items():
            report_lines.append(f"      Anchor {s:>7,d} sims: {d.get('wins', 0)}W / {d.get('games', 0)} games ({d.get('wins', 0)/max(1, d.get('games', 1))*100:.1f}%)")
        report_lines.append("")
        report_lines.append("  • Sequential Match History:")
        for m in getattr(optimizer, 'match_history', []):
            report_lines.append(f"      Step #{m.get('step', '?')}: {m.get('phase', '?')} @ {m.get('anchor', '?'):,} sims -> {m.get('wins', 0)}W - {m.get('losses', 0)}L ({m.get('win_rate', 0)*100:.1f}%)")

    report_lines.append("")
    report_lines.append("=" * 76)

    full_text = "\n".join(report_lines)
    try:
        with open(filepath, "w") as f:
            f.write(full_text)
    except Exception as write_err:
        print(f"⚠️ Could not write crash dump file: {write_err}")

    print("\n\n" + "!" * 76)
    print("🚨 AN UNHANDLED EXCEPTION OCCURRED DURING ARENA EVALUATION")
    print(f"📁 Full diagnostic report dumped to: {os.path.abspath(filepath)}")
    print("!" * 76)
    print(traceback.format_exc())
    print("!" * 76 + "\n")


class SequentialCSEOptimizer:
    """
    Active Sequential Experiment Design engine for Simulation Compute-Scale Equivalence (N_50).
    Uses Binomial Maximum Likelihood Estimation (GLM Logit Link) combined with Fisher Information-driven
    c- and D-optimal anchor placement and Delta Method variance estimation.
    """
    def __init__(self, start_sims: int = 5000, min_sims: int = 500, max_sims: int = 150000, target_ci_ratio: float = 1.30):
        self.start_sims = start_sims
        self.min_sims = min_sims
        self.max_sims = max_sims
        self.target_ci_ratio = target_ci_ratio

        # Aggregated anchor data: {sims: {"games": int, "wins": int}}
        self.anchor_data = {}

        # History of sequential match evaluations
        self.match_history = []

        # Trajectory of parameter estimates over steps
        self.step_history = []
        self.n50_history = []
        self.ci_lower_history = []
        self.ci_upper_history = []
        self.ci_ratio_history = []

        # Current GLM parameters
        self.beta0 = 0.0
        self.beta1 = -1.0
        self.x0 = math.log(max(1, start_sims))
        self.n50 = float(start_sims)
        self.se_x0 = 1.0
        self.ci_lower = self.n50 * 0.3
        self.ci_upper = self.n50 * 3.0
        self.ci_ratio = 10.0
        self.converged = False

        # Phase tracking
        self.is_bracketed = False
        self.fisher_cycle_idx = 0
        self.current_sims = start_sims

    def add_match_result(self, sims: int, wins: int, games: int, phase_name: str):
        if sims not in self.anchor_data:
            self.anchor_data[sims] = {"games": 0, "wins": 0}
        self.anchor_data[sims]["games"] += games
        self.anchor_data[sims]["wins"] += wins

        step = len(self.match_history) + 1
        self.match_history.append({
            "step": step,
            "phase": phase_name,
            "anchor": sims,
            "wins": wins,
            "losses": games - wins,
            "games": games,
            "win_rate": wins / max(1, games)
        })

        # Check empirical bracketing across parity: win rates > 0.60 and < 0.40
        anchor_rates = [d["wins"] / max(1, d["games"]) for d in self.anchor_data.values()]
        has_high = any(r > 0.60 for r in anchor_rates)
        has_low = any(r < 0.40 for r in anchor_rates)
        self.is_bracketed = (has_high and has_low)

        # Fit Binomial GLM
        try:
            self.fit_glm()
        except Exception as glm_err:
            print(f"⚠️ GLM Fit warning: {glm_err}")

        # Record trajectory
        if self.n50 is not None and self.ci_lower is not None and self.ci_upper is not None:
            self.step_history.append(step)
            self.n50_history.append(self.n50)
            self.ci_lower_history.append(self.ci_lower)
            self.ci_upper_history.append(self.ci_upper)
            self.ci_ratio_history.append(self.ci_ratio)

            total_games_played = sum(d["games"] for d in self.anchor_data.values())
            if total_games_played >= 12 and self.ci_ratio <= self.target_ci_ratio and self.is_bracketed:
                self.converged = True

    def fit_glm(self):
        unique_anchors = sorted(self.anchor_data.keys())
        if len(unique_anchors) == 0:
            return

        x_arr = np.array([math.log(max(1, s)) for s in unique_anchors], dtype=float)
        n_arr = np.array([self.anchor_data[s]["games"] for s in unique_anchors], dtype=float)
        k_arr = np.array([self.anchor_data[s]["wins"] for s in unique_anchors], dtype=float)

        if len(unique_anchors) < 2:
            sim = unique_anchors[0]
            self.n50 = float(sim)
            self.x0 = math.log(max(1.0, self.n50))
            self.ci_lower = self.n50 * 0.4
            self.ci_upper = self.n50 * 2.5
            self.ci_ratio = self.ci_upper / max(1e-9, self.ci_lower)
            self.se_x0 = 0.5
            return

        # Negative log-likelihood with mild L2 regularization to prevent complete separation
        def neg_log_lik(beta):
            b0, b1 = beta
            eta = np.clip(b0 + b1 * x_arr, -30, 30)
            p = 1.0 / (1.0 + np.exp(-eta))
            eps = 1e-12
            ll = np.sum(k_arr * np.log(p + eps) + (n_arr - k_arr) * np.log(1.0 - p + eps))
            reg = 1e-4 * (b0**2 + (b1 + 1.0)**2)
            return -(ll - reg)

        init_b1 = -1.0
        init_b0 = -init_b1 * float(np.mean(x_arr))
        try:
            res = minimize(neg_log_lik, [init_b0, init_b1], method='L-BFGS-B')
            b0, b1 = res.x
        except Exception:
            b0, b1 = init_b0, init_b1

        if not np.isfinite(b0) or not np.isfinite(b1):
            b0, b1 = init_b0, init_b1

        if b1 >= -1e-4:
            b1 = -1e-4

        self.beta0 = float(b0)
        self.beta1 = float(b1)

        # Fisher Information Matrix & Delta Method Variance
        eta = np.clip(b0 + b1 * x_arr, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = n_arr * p * (1.0 - p)

        X = np.column_stack([np.ones_like(x_arr), x_arr])
        I = X.T @ (w[:, None] * X) + 1e-4 * np.eye(2)
        try:
            cov = np.linalg.inv(I)
        except np.linalg.LinAlgError:
            cov = np.eye(2) * 10.0

        x0 = -b0 / b1
        grad_g = np.array([-1.0 / b1, b0 / (b1 ** 2)])
        var_x0 = float(grad_g.T @ cov @ grad_g)
        if not np.isfinite(var_x0) or var_x0 <= 0:
            var_x0 = 1.0
        se_x0 = float(np.sqrt(max(1e-9, var_x0)))

        min_log = math.log(max(1.0, self.min_sims * 0.1))
        max_log = math.log(max(1.0, self.max_sims * 10.0))
        x0_clamped = float(np.clip(x0, min_log, max_log))

        self.x0 = x0_clamped
        self.n50 = float(math.exp(x0_clamped))
        self.se_x0 = se_x0
        self.ci_lower = float(math.exp(max(-5.0, x0_clamped - 1.96 * se_x0)))
        self.ci_upper = float(math.exp(min(25.0, x0_clamped + 1.96 * se_x0)))
        self.ci_ratio = float(self.ci_upper / max(1e-9, self.ci_lower))

    def get_next_anchor(self):
        # 1. Seed / Hunt Phase: Not yet bracketed on both sides (p < 0.40 and p > 0.60)
        if not self.is_bracketed:
            phase_name = "Seed/Hunt"
            if len(self.match_history) == 0:
                self.current_sims = self.start_sims
            else:
                last_match = self.match_history[-1]
                last_wr = last_match["win_rate"]
                if last_wr >= 0.50:
                    next_s = min(self.max_sims, int(self.current_sims * 2.0))
                else:
                    next_s = max(self.min_sims, int(self.current_sims * 0.5))
                self.current_sims = next_s
            return self.current_sims, phase_name

        # 2. Convergence Phase: Online Fisher Information sampling targeting c- and D-optimal points
        phase_name = "Convergence (D-Optimal)"
        delta_x = min(1.543 / max(1e-4, abs(self.beta1)), 1.2)

        cycle = self.fisher_cycle_idx % 4
        self.fisher_cycle_idx += 1

        if cycle == 0 or cycle == 2:
            target_x = self.x0
        elif cycle == 1:
            target_x = self.x0 - delta_x
        else:
            target_x = self.x0 + delta_x

        try:
            next_sims = int(round(math.exp(np.clip(target_x, 0.0, 20.0))))
        except Exception:
            next_sims = self.start_sims

        next_sims = max(self.min_sims, min(self.max_sims, next_sims))
        self.current_sims = next_sims
        return self.current_sims, phase_name

    def project_remaining_time(self, ema_game_time: float, current_step: int, max_steps: int) -> tuple:
        """
        Projects remaining games and estimated time to hit target CI precision based on variance convergence rate.
        """
        total_games_played = sum(d["games"] for d in self.anchor_data.values())
        if not self.is_bracketed or self.ci_ratio >= 10.0 or total_games_played == 0:
            rem_steps = max(0, max_steps - current_step)
            rem_games = rem_steps * 2
            rem_sec = rem_games * max(0.1, ema_game_time)
            return rem_games, rem_sec

        target_ratio = max(1.01, self.target_ci_ratio)
        target_se = math.log(target_ratio) / 3.92
        if self.se_x0 <= target_se:
            return 0, 0.0

        try:
            req_games = total_games_played * ((self.se_x0 / max(1e-6, target_se)) ** 2)
            if not np.isfinite(req_games):
                req_games = max_steps * 2
            rem_games = max(0, int(math.ceil(req_games - total_games_played)))
            max_possible_rem_games = max(0, (max_steps - current_step) * 2)
            rem_games = min(rem_games, max_possible_rem_games)
            rem_sec = rem_games * max(0.1, ema_game_time)
            return rem_games, rem_sec
        except Exception:
            rem_games = max(0, (max_steps - current_step) * 2)
            return rem_games, rem_games * max(0.1, ema_game_time)


def render_board_lines(env, board_size, extra_info):
    lines = []
    lines.append("┌────────── REALTIME GAME STATUS ──────────┐")
    for k, v in extra_info.items():
        lines.append(f" {k:<15}: {v}")
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
                row_symbols.append("\033[91m.\033[0m")
            elif c == 0 or c == board_size - 1:
                row_symbols.append("\033[94m.\033[0m")
            else:
                row_symbols.append(".")

        row_str = " ".join(row_symbols)
        lines.append(f" {r:1d}{indent}{row_str}")

    lines.append("└──────────────────────────────────────────┘")
    return lines


import re

def _visible_len(s: str) -> int:
    """Computes visible length of string ignoring ANSI escape codes."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return len(ansi_escape.sub('', s))

def update_dashboard(optimizer: SequentialCSEOptimizer, current_env, board_size: int, extra_info: dict):
    try:
        plotext.clear_terminal()

        unique_anchors = sorted(optimizer.anchor_data.keys())
        
        # ── Plot 1: Logistic Win-Rate Curve ──
        plotext.clf()
        plotext.plotsize(46, 8)

        if len(unique_anchors) >= 1:
            x_sims = unique_anchors
            y_rates = [optimizer.anchor_data[s]["wins"] / max(1, optimizer.anchor_data[s]["games"]) for s in unique_anchors]
            plotext.scatter(x_sims, y_rates, color="cyan", label="Anchors")
            plotext.plot([min(x_sims), max(x_sims)], [0.5, 0.5], color="red")

            if len(unique_anchors) >= 2 and optimizer.n50 is not None and np.isfinite(optimizer.n50):
                min_s = min(x_sims)
                max_s = max(x_sims)
                min_bound = max(1.0, min(min_s, optimizer.n50 * 0.6))
                max_bound = max(min_bound * 1.2, max(max_s, optimizer.n50 * 1.4))
                if np.isfinite(min_bound) and np.isfinite(max_bound) and max_bound > min_bound:
                    sim_grid = np.geomspace(min_bound, max_bound, 50)
                    x_grid = np.log(sim_grid)
                    p_grid = 1.0 / (1.0 + np.exp(-(optimizer.beta0 + optimizer.beta1 * x_grid)))
                    plotext.plot(sim_grid.tolist(), p_grid.tolist(), color="green", label="Sigmoid")
                    plotext.scatter([optimizer.n50], [0.5], color="yellow", marker="x", label="N50")
                    plotext.title(f"Logistic Curve (N50: {optimizer.n50:.1f} | Ratio: {optimizer.ci_ratio:.2f}x)")
                else:
                    plotext.title("Logistic Win-Rate Curve")
            else:
                plotext.title("Logistic Win-Rate Curve")
            plotext.xscale("log")
        else:
            plotext.title("Logistic Curve (Sampling...)")

        plotext.ylabel("Win Rate")
        p1_lines = [l for l in plotext.build().split("\n") if l]

        # ── Plot 2: Real-time 95% CI Convergence Plot ──
        plotext.clf()
        plotext.plotsize(46, 8)

        if len(optimizer.step_history) >= 2:
            steps = optimizer.step_history
            upper_line = optimizer.ci_upper_history
            n50_line = optimizer.n50_history
            lower_line = optimizer.ci_lower_history

            plotext.plot(steps, upper_line, color="red", label="Upper 95%")
            plotext.plot(steps, n50_line, color="yellow", marker="sd", label="N50")
            plotext.plot(steps, lower_line, color="blue", label="Lower 95%")
            plotext.title(f"95% CI Convergence (Target <= {optimizer.target_ci_ratio:.2f}x)")
            plotext.xlabel("Evaluation Step")
            plotext.ylabel("Sims")
        else:
            plotext.title("CI Convergence Trajectory")
            plotext.xlabel("Evaluation Step")
            plotext.ylabel("Sims")

        p2_lines = [l for l in plotext.build().split("\n") if l]
        chart_lines = p1_lines + p2_lines
        board_lines = render_board_lines(current_env, board_size, extra_info)

        max_l = max(len(chart_lines), len(board_lines))
        output = []
        for i in range(max_l):
            raw_c = chart_lines[i] if i < len(chart_lines) else ""
            v_len = _visible_len(raw_c)
            c_line = raw_c + (" " * max(0, 46 - v_len))
            b_line = board_lines[i] if i < len(board_lines) else ""
            output.append(f"{c_line} │ {b_line}")

        print("\n".join(output))
    except Exception as display_err:
        # Fallback text rendering if terminal/plotext has an error, so the game never stops!
        print(f"[{extra_info.get('Eval Step', '')}] {extra_info.get('Current Game', '')} - Move: {extra_info.get('Current Move', '')} (Display Notice: {display_err})")


async def play_paired_match(
    board_size: int,
    muzero_sims: int,
    classic_sims: int,
    model,
    optimizer: SequentialCSEOptimizer,
    step_num: int,
    max_steps: int,
    phase_str: str,
    input_channels: int,
    ema_game_time: float,
    total_muzero_wins_all: int,
    total_games_all: int
):
    """
    Executes 1 paired match (2 games: Game A MuZero=Red, Game B MuZero=Blue) at classic_sims.
    """
    muzero_wins = 0
    total_games = 2
    stop_event = asyncio.Event()

    muzero_time_total = 0.0
    muzero_moves = 0
    classic_time_total = 0.0
    classic_moves = 0

    rem_games, rem_sec = optimizer.project_remaining_time(ema_game_time, step_num, max_steps)
    eta_min = rem_sec / 60.0
    eta_str = f"{int(rem_sec//60)}:{int(rem_sec%60):02d} min" if rem_sec > 0 else "0:00 min"

    # ── Game A: MuZero = Red (1), Classic = Blue (2) ──
    env_a = HexEnv(board_size=board_size)
    latent_mcts_a = LatentMCTS(model=model)
    classic_mcts_a = ClassicMCTS()
    move_num = 0
    game_a_num = total_games_all + 1

    while env_a.winner == 0:
        move_num += 1
        legal = env_a.legal_actions()
        if not legal:
            break

        info = {
            "Phase": phase_str,
            "Eval Step": f"Step {step_num} / {max_steps} (2 games/step)",
            "Current Anchor": f"{classic_sims:,} Classic Sims",
            "Current Game": f"Game #{game_a_num} (Pair 1/2: MuZero=Red)",
            "Overall Score": f"MuZero {total_muzero_wins_all} - {total_games_all - total_muzero_wins_all} Classic",
            "95% CI Ratio": f"{optimizer.ci_ratio:.2f}x (Target: <={optimizer.target_ci_ratio:.2f}x)",
            "Dynamic ETA": f"{eta_str} (EMA: {ema_game_time:.1f}s/game)",
            "Current Move": f"#{move_num} - {'MuZero (Red)' if env_a.current_player == 1 else 'Classic (Blue)'}"
        }
        update_dashboard(optimizer, env_a, board_size, info)

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

    # ── Game B: Classic = Red (1), MuZero = Blue (2) ──
    env_b = HexEnv(board_size=board_size)
    latent_mcts_b = LatentMCTS(model=model)
    classic_mcts_b = ClassicMCTS()
    move_num = 0
    game_b_num = total_games_all + 2

    while env_b.winner == 0:
        move_num += 1
        legal = env_b.legal_actions()
        if not legal:
            break

        info = {
            "Phase": phase_str,
            "Eval Step": f"Step {step_num} / {max_steps} (2 games/step)",
            "Current Anchor": f"{classic_sims:,} Classic Sims",
            "Current Game": f"Game #{game_b_num} (Pair 2/2: MuZero=Blue)",
            "Overall Score": f"MuZero {total_muzero_wins_all + muzero_wins} - {total_games_all + 1 - (total_muzero_wins_all + muzero_wins)} Classic",
            "95% CI Ratio": f"{optimizer.ci_ratio:.2f}x (Target: <={optimizer.target_ci_ratio:.2f}x)",
            "Dynamic ETA": f"{eta_str} (EMA: {ema_game_time:.1f}s/game)",
            "Current Move": f"#{move_num} - {'Classic (Red)' if env_b.current_player == 1 else 'MuZero (Blue)'}"
        }
        update_dashboard(optimizer, env_b, board_size, info)

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

    win_rate = muzero_wins / total_games
    return win_rate, muzero_wins, total_games, muzero_time_total, muzero_moves, classic_time_total, classic_moves


def print_final_telemetry_report(
    optimizer: SequentialCSEOptimizer,
    total_muzero_wins: int,
    total_games: int,
    total_muzero_time: float,
    total_muzero_moves: int,
    total_classic_time: float,
    total_classic_sims_completed: int,
    t_elapsed_total: float,
    muzero_sims: int,
    target_ci_ratio: float,
    baseline_log_n50: float = None,
    interrupted: bool = False
):
    print(f"\n========================================================================")
    header_prefix = "⚠️ INTERMEDIATE / PARTIAL " if interrupted else "📊 "
    print(f"{header_prefix}ACTIVE SEQUENTIAL EXPERIMENT DESIGN: TELEMETRY REPORT")
    print(f"========================================================================")
    
    print(f"\n📋 MATCH & ROUND WIN/LOSS RECORDS")
    print(f"{'Step':<6} {'Phase':<24} {'Anchor':<14} {'Score (M - C)':<16} {'Win Rate':<10}")
    print(f"────────────────────────────────────────────────────────────────────────")
    for r in optimizer.match_history:
        p_str = f"{r['phase']}"
        a_str = f"{r['anchor']:,} sims"
        s_str = f"{r['wins']}W - {r['losses']}L"
        w_str = f"{r['win_rate']*100:.1f}%"
        print(f"#{r['step']:<5} {p_str:<24} {a_str:<14} {s_str:<16} {w_str:<10}")
    print(f"────────────────────────────────────────────────────────────────────────")
    overall_wr = (total_muzero_wins / max(1, total_games)) * 100
    print(f"{'TOTAL':<6} {'All Evaluated Steps':<24} {'Aggregated':<14} {f'{total_muzero_wins}W - {total_games - total_muzero_wins}L':<16} {f'{overall_wr:.1f}%':<10}")
    print(f"────────────────────────────────────────────────────────────────────────")

    print(f"\n🎯 AGGREGATED ANCHOR SUPPORT POINTS")
    print(f"{'Anchor Sims':<16} {'Games Played':<14} {'MuZero Wins':<14} {'Win Rate':<10}")
    print(f"────────────────────────────────────────────────────────────────────────")
    for sim in sorted(optimizer.anchor_data.keys()):
        d = optimizer.anchor_data[sim]
        wr = (d['wins'] / max(1, d['games'])) * 100
        print(f"{sim:<16,d} {d['games']:<14d} {d['wins']:<14d} {wr:<9.1f}%")
    print(f"────────────────────────────────────────────────────────────────────────")

    # Timing Stats
    t_muzero = total_muzero_time / max(1, total_muzero_moves)
    t_classic_sim = total_classic_time / max(1, total_classic_sims_completed)
    crossover = t_muzero / max(1e-9, t_classic_sim)

    print(f"\n⏱️ EXECUTION SPEED & TIMING METRICS")
    print(f"  • Total Benchmark Time:           {int(t_elapsed_total//60)}m {int(t_elapsed_total%60):02d}s ({total_games} games)")
    print(f"  • MuZero ({muzero_sims} sims):             {t_muzero:.4f} sec/move")
    print(f"  • Classic MCTS Simulation Time:   {t_classic_sim*1000:.4f} ms/simulation")
    print(f"  • Wall-Clock Crossover (N_time):  {crossover:.1f} classic simulations")

    # Convergence & CSE Parity
    cse = optimizer.n50
    ci_low = optimizer.ci_lower
    ci_high = optimizer.ci_upper
    ci_ratio = optimizer.ci_ratio

    print(f"\n🏆 MODEL EFFICIENCY & CONVERGENCE RESULTS")
    print(f"  • Natural Log Parity (ln N_50):   {optimizer.x0:.4f} ± {optimizer.se_x0:.4f} (SE)")
    print(f"  • SIMULATION CSE (N_50):          {cse:.1f}")
    print(f"  • 95% Confidence Interval:        [{ci_low:.1f}, {ci_high:.1f}]")
    print(f"  • 95% Parameter Uncertainty Ratio:{ci_ratio:.3f}x (Target: <={target_ci_ratio:.2f}x)")
    
    speedup = (cse * t_classic_sim) / max(1e-9, t_muzero)
    print(f"  • REALIZED WALL-CLOCK SPEEDUP:    {speedup:.2f}x")

    # Diminishing Returns Index
    if baseline_log_n50 is not None:
        delta_ln_n50 = optimizer.x0 - baseline_log_n50
        multiplier = math.exp(delta_ln_n50)
        print(f"\n📈 MARGINAL RETURN & SEARCH EFFICIENCY GAIN")
        print(f"  • Baseline ln(N_50):              {baseline_log_n50:.4f}")
        print(f"  • Current ln(N_50):               {optimizer.x0:.4f}")
        print(f"  • Delta ln(N_50):                 {delta_ln_n50:+.4f}")
        print(f"  • Equivalence Scaling Multiplier: {multiplier:.2f}x over baseline")

    print(f"========================================================================\n")


async def main():
    parser = argparse.ArgumentParser(description="Active Sequential Logistic Arena for MuZero vs Classic MCTS")
    parser.add_argument("--run-id", type=str, default="v4_clone", help="Run ID for versioned weights")
    parser.add_argument("--board-size", type=int, default=7)
    parser.add_argument("--muzero-sims", type=int, default=400)
    parser.add_argument("--classic-anchors", type=str, default="250,500,1000,2500,5000,10000")
    parser.add_argument("--games-per-anchor", type=int, default=10)
    parser.add_argument("--use-fcn", action="store_true", help="Use Fully Convolutional Prediction Head")
    parser.add_argument("--input-channels", type=int, default=3, help="Number of input observation channels")
    
    # Active Sequential Experiment Design flags
    parser.add_argument("--adaptive", action="store_true", help="Enable Active Sequential Experiment Design (Fisher Information sampling & automated convergence)")
    parser.add_argument("--target-ci-ratio", type=float, default=1.30, help="Target ratio of upper/lower 95% confidence bounds to trigger termination")
    parser.add_argument("--max-eval-steps", type=int, default=30, help="Upper safety cap for paired matches (max games = 2 * steps)")
    parser.add_argument("--start-sims", type=int, default=5000, help="Initial classic simulation anchor for adaptive hunt")
    parser.add_argument("--min-sims", type=int, default=500, help="Minimum simulation floor for adaptive hunt")
    parser.add_argument("--max-sims", type=int, default=150000, help="Maximum simulation ceiling for adaptive hunt")
    parser.add_argument("--baseline-log-n50", type=float, default=None, help="Previous model checkpoint ln(N_50) for marginal return calculation")
    args = parser.parse_args()

    board_size = args.board_size
    action_space_size = board_size ** 2
    latent_channels = 96
    num_res_blocks = 8

    # Load weights
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

    optimizer = SequentialCSEOptimizer(
        start_sims=args.start_sims,
        min_sims=args.min_sims,
        max_sims=args.max_sims,
        target_ci_ratio=args.target_ci_ratio
    )

    total_muzero_time = 0.0
    total_muzero_moves = 0
    total_classic_time = 0.0
    total_classic_sims_completed = 0
    total_muzero_wins = 0
    total_games = 0

    ema_game_time = 5.0  # Initial EMA estimate in seconds
    t_start_all = time.time()
    last_context_info = {}

    try:
        if args.adaptive:
            print(f"🎯 Starting Active Sequential Experiment Design (Target CI Ratio: <={args.target_ci_ratio:.2f}x)...")
            for step in range(1, args.max_eval_steps + 1):
                classic_sims, phase_name = optimizer.get_next_anchor()

                last_context_info = {
                    "Step": step,
                    "Phase": phase_name,
                    "Sims": classic_sims,
                    "Total Games": total_games
                }

                t_match_start = time.time()
                win_rate, muz_w, games_played, muz_t, muz_m, clas_t, clas_m = await play_paired_match(
                    board_size=args.board_size,
                    muzero_sims=args.muzero_sims,
                    classic_sims=classic_sims,
                    model=model,
                    optimizer=optimizer,
                    step_num=step,
                    max_steps=args.max_eval_steps,
                    phase_str=phase_name,
                    input_channels=args.input_channels,
                    ema_game_time=ema_game_time,
                    total_muzero_wins_all=total_muzero_wins,
                    total_games_all=total_games
                )
                t_match_elapsed = time.time() - t_match_start

                # Update rolling EMA of game time
                match_avg_game_time = t_match_elapsed / max(1, games_played)
                ema_game_time = 0.3 * match_avg_game_time + 0.7 * ema_game_time

                total_muzero_wins += muz_w
                total_games += games_played
                total_muzero_time += muz_t
                total_muzero_moves += muz_m
                total_classic_time += clas_t
                total_classic_sims_completed += (clas_m * classic_sims)

                optimizer.add_match_result(classic_sims, muz_w, games_played, phase_name)

                if optimizer.converged:
                    break
        else:
            # Fixed Anchors Mode
            anchors = [int(x) for x in args.classic_anchors.split(',')]
            pairs_per_anchor = max(1, args.games_per_anchor // 2)
            total_steps = len(anchors) * pairs_per_anchor
            step = 0

            for anchor in anchors:
                for p in range(pairs_per_anchor):
                    step += 1
                    last_context_info = {
                        "Step": step,
                        "Anchor": anchor,
                        "Total Games": total_games
                    }
                    t_match_start = time.time()
                    win_rate, muz_w, games_played, muz_t, muz_m, clas_t, clas_m = await play_paired_match(
                        board_size=args.board_size,
                        muzero_sims=args.muzero_sims,
                        classic_sims=anchor,
                        model=model,
                        optimizer=optimizer,
                        step_num=step,
                        max_steps=total_steps,
                        phase_str="Fixed Anchors",
                        input_channels=args.input_channels,
                        ema_game_time=ema_game_time,
                        total_muzero_wins_all=total_muzero_wins,
                        total_games_all=total_games
                    )
                    t_match_elapsed = time.time() - t_match_start
                    match_avg_game_time = t_match_elapsed / max(1, games_played)
                    ema_game_time = 0.3 * match_avg_game_time + 0.7 * ema_game_time

                    total_muzero_wins += muz_w
                    total_games += games_played
                    total_muzero_time += muz_t
                    total_muzero_moves += muz_m
                    total_classic_time += clas_t
                    total_classic_sims_completed += (clas_m * anchor)

                    optimizer.add_match_result(anchor, muz_w, games_played, "Fixed Anchors")

        # ── Final Summary Telemetry Report ──
        t_elapsed_total = time.time() - t_start_all
        print_final_telemetry_report(
            optimizer=optimizer,
            total_muzero_wins=total_muzero_wins,
            total_games=total_games,
            total_muzero_time=total_muzero_time,
            total_muzero_moves=total_muzero_moves,
            total_classic_time=total_classic_time,
            total_classic_sims_completed=total_classic_sims_completed,
            t_elapsed_total=t_elapsed_total,
            muzero_sims=args.muzero_sims,
            target_ci_ratio=args.target_ci_ratio,
            baseline_log_n50=args.baseline_log_n50,
            interrupted=False
        )

    except KeyboardInterrupt:
        print("\n\n⏹️ Evaluation interrupted by user (Ctrl+C). Printing partial results...")
        t_elapsed_total = time.time() - t_start_all
        print_final_telemetry_report(
            optimizer=optimizer,
            total_muzero_wins=total_muzero_wins,
            total_games=total_games,
            total_muzero_time=total_muzero_time,
            total_muzero_moves=total_muzero_moves,
            total_classic_time=total_classic_time,
            total_classic_sims_completed=total_classic_sims_completed,
            t_elapsed_total=t_elapsed_total,
            muzero_sims=args.muzero_sims,
            target_ci_ratio=args.target_ci_ratio,
            baseline_log_n50=args.baseline_log_n50,
            interrupted=True
        )

    except Exception as exc:
        dump_crash_telemetry(exc, optimizer=optimizer, extra_info=last_context_info)
        t_elapsed_total = time.time() - t_start_all
        if total_games > 0:
            print("📊 Printing partial telemetry results from games completed prior to the exception:")
            print_final_telemetry_report(
                optimizer=optimizer,
                total_muzero_wins=total_muzero_wins,
                total_games=total_games,
                total_muzero_time=total_muzero_time,
                total_muzero_moves=total_muzero_moves,
                total_classic_time=total_classic_time,
                total_classic_sims_completed=total_classic_sims_completed,
                t_elapsed_total=t_elapsed_total,
                muzero_sims=args.muzero_sims,
                target_ci_ratio=args.target_ci_ratio,
                baseline_log_n50=args.baseline_log_n50,
                interrupted=True
            )
        raise

if __name__ == "__main__":
    asyncio.run(main())


