import os
import sys
import time
import asyncio
import numpy as np
import torch

# Ensure backend directory is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from muzero_nets import MuZeroModels
from latent_mcts import LatentMCTS
from hex_env import HexEnv

def benchmark_inference(board_size: int, n_calls: int = 1000):
    action_space_size = board_size * board_size
    model = MuZeroModels(board_size=board_size, action_space_size=action_space_size, latent_channels=32, num_res_blocks=2)
    model.eval()

    obs = torch.randn(1, 3, board_size, board_size, dtype=torch.float32)
    action = torch.tensor([0], dtype=torch.long)

    # Warmup
    with torch.no_grad():
        v, r, p, s = model.initial_inference(obs)
        model.recurrent_inference(s, action)

    # Benchmark initial_inference
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_calls):
            model.initial_inference(obs)
    t1 = time.perf_counter()
    init_time_ms = ((t1 - t0) / n_calls) * 1000.0

    # Benchmark recurrent_inference
    with torch.no_grad():
        s = model.representation(obs)
        t0 = time.perf_counter()
        for _ in range(n_calls):
            model.recurrent_inference(s, action)
        t1 = time.perf_counter()
    rec_time_ms = ((t1 - t0) / n_calls) * 1000.0

    return init_time_ms, rec_time_ms

def benchmark_mcts_search(board_size: int, num_simulations: int = 100):
    action_space_size = board_size * board_size
    model = MuZeroModels(board_size=board_size, action_space_size=action_space_size, latent_channels=32, num_res_blocks=2)
    model.eval()

    obs = torch.randn(1, 3, board_size, board_size, dtype=torch.float32)
    legal_actions = list(range(action_space_size))
    mcts = LatentMCTS(model=model, c_puct=1.25)

    t0 = time.perf_counter()
    asyncio.run(mcts.search(obs, legal_actions, num_simulations=num_simulations))
    t1 = time.perf_counter()
    search_time_ms = (t1 - t0) * 1000.0

    return search_time_ms

def benchmark_self_play_games(board_size: int, n_games: int = 3, num_sims: int = 40):
    action_space_size = board_size * board_size
    env = HexEnv(board_size=board_size)
    model = MuZeroModels(board_size=board_size, action_space_size=action_space_size, latent_channels=32, num_res_blocks=2)
    model.eval()

    t0 = time.perf_counter()
    for _ in range(n_games):
        obs = env.reset()
        done = False
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            legal = env.legal_actions()
            mcts = LatentMCTS(model=model, c_puct=1.25)
            root = asyncio.run(mcts.search(obs_tensor, legal, num_simulations=num_sims))
            actions = list(root.children.keys())
            chosen_action = max(actions, key=lambda a: root.children[a].visit_count)
            obs, reward, done = env.step(chosen_action)
    t1 = time.perf_counter()

    avg_game_time_sec = (t1 - t0) / n_games
    return avg_game_time_sec

def main():
    print("=== Running Hardware Performance Benchmarks (5x5 vs 7x7) ===")

    # 1. Measure Inference Speed (1,000 calls)
    print("Benchmarking model inference speed (1,000 calls)...")
    init_5x5, rec_5x5 = benchmark_inference(5, n_calls=1000)
    init_7x7, rec_7x7 = benchmark_inference(7, n_calls=1000)

    # 2. Measure MCTS Throughput (100 simulations)
    print("Benchmarking Latent MCTS 100-simulation search speed...")
    mcts_5x5 = benchmark_mcts_search(5, num_simulations=100)
    mcts_7x7 = benchmark_mcts_search(7, num_simulations=100)

    # 3. Measure Self-Play Game Completion Rate (3 games)
    print("Benchmarking self-play game completion rate (3 games, 40 sims/move)...")
    sp_5x5_sec = benchmark_self_play_games(5, n_games=3, num_sims=40)
    sp_7x7_sec = benchmark_self_play_games(7, n_games=3, num_sims=40)

    # Projections
    proj_50_5x5_min = (sp_5x5_sec * 50) / 60.0
    proj_100_5x5_min = (sp_5x5_sec * 100) / 60.0

    proj_40_7x7_min = (sp_7x7_sec * 40) / 60.0
    proj_50_7x7_min = (sp_7x7_sec * 50) / 60.0

    report = f"""================================================================================
PHASE 5.5 HARDWARE BENCHMARK REPORT
================================================================================

1. RAW PYTORCH MODEL INFERENCE SPEED (Avg Time / Call over 1,000 iterations):
--------------------------------------------------------------------------------
5x5 Board:
  - RepresentationNetwork (Initial Inference): {init_5x5:.4f} ms / call
  - DynamicsNetwork (Recurrent Inference):     {rec_5x5:.4f} ms / call

7x7 Board:
  - RepresentationNetwork (Initial Inference): {init_7x7:.4f} ms / call
  - DynamicsNetwork (Recurrent Inference):     {rec_7x7:.4f} ms / call


2. LATENT MCTS THROUGHPUT (100-Simulation Search Execution Time):
--------------------------------------------------------------------------------
  - 5x5 Board (25 Legal Actions):  {mcts_5x5:.2f} ms / 100 sims
  - 7x7 Board (49 Legal Actions):  {mcts_7x7:.2f} ms / 100 sims


3. SELF-PLAY GAME COMPLETION RATE (40 MCTS Sims / Move):
--------------------------------------------------------------------------------
  - 5x5 Board Avg Time / Game: {sp_5x5_sec:.2f} seconds
  - 7x7 Board Avg Time / Game: {sp_7x7_sec:.2f} seconds


4. TRAINING TIME PROJECTIONS:
--------------------------------------------------------------------------------
  - 5x5 Board (50 Games):   ~{proj_50_5x5_min:.2f} minutes
  - 5x5 Board (100 Games):  ~{proj_100_5x5_min:.2f} minutes
  - 7x7 Board (40 Games):   ~{proj_40_7x7_min:.2f} minutes
  - 7x7 Board (50 Games):   ~{proj_50_7x7_min:.2f} minutes


5. DECISION CRITERIA:
--------------------------------------------------------------------------------
7x7 Avg Game Time Threshold: 15.00 seconds.
Measured 7x7 Game Time:      {sp_7x7_sec:.2f} seconds.
Selected Pipeline: {"7x7 Scaled Training (40 Games)" if sp_7x7_sec < 15.0 else "5x5 Master Model Training (100 Games)"}
================================================================================
"""

    eval_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmark_eval.txt"))
    with open(eval_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✅ Benchmark completed. Results saved to: {eval_path}")
    print(f"7x7 per-game time: {sp_7x7_sec:.2f}s (Threshold: 15s)")

    # Return decision boolean for scaled training runner
    return sp_7x7_sec < 15.0

if __name__ == "__main__":
    main()
