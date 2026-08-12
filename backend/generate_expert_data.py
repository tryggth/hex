import argparse
import os
import sys
import pickle
import numpy as np
import multiprocessing
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hex_env import HexEnv
from classic_mcts import ClassicMCTS

def generate_game(args):
    board_size, sims_per_move, temp_moves = args
    env = HexEnv(board_size=board_size)
    mcts = ClassicMCTS()
    
    trajectory = []
    
    while env.winner == 0:
        if not env.legal_actions():
            break
            
        move_count = env.moves_made
        temp = 1.0 if move_count < temp_moves else 0.0
        
        obs = env.get_observation()
        player = env.current_player
        
        act, pol = mcts.search(env, num_simulations=sims_per_move, temperature=temp)
        
        trajectory.append({
            "obs": obs.copy(),
            "action": act,
            "target_policy": pol.copy(),
            "player": player
        })
        
        env.step(act)
        
    winner = env.winner
    
    # Assign target values and build augmented trajectory
    aug_trajectory = []
    for step in trajectory:
        val = 1.0 if step["player"] == winner else -1.0
        step["target_val"] = val
        
        rot_obs = np.rot90(step["obs"], k=2, axes=(1, 2)).copy()
        rot_act = (board_size * board_size - 1) - step["action"]
        rot_pol = step["target_policy"][::-1].copy()
        
        aug_trajectory.append({
            "obs": rot_obs,
            "action": int(rot_act),
            "target_policy": rot_pol,
            "player": step["player"],
            "target_val": val
        })
        
    return [trajectory, aug_trajectory]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-size", type=int, default=7)
    parser.add_argument("--num-games", type=int, default=500)
    parser.add_argument("--sims-per-move", type=int, default=1000)
    parser.add_argument("--temp-moves", type=int, default=6)
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--output", type=str, default="backend/expert_data.pkl")
    args = parser.parse_args()
    
    print(f"Generating {args.num_games} expert games using {args.workers} workers...")
    
    worker_args = [(args.board_size, args.sims_per_move, args.temp_moves) for _ in range(args.num_games)]
    all_trajectories = []
    
    with multiprocessing.Pool(args.workers) as pool:
        for result in tqdm(pool.imap_unordered(generate_game, worker_args), total=args.num_games):
            all_trajectories.extend(result)
            
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(all_trajectories, f)
        
    print(f"Saved {len(all_trajectories)} augmented trajectories to {args.output}")

if __name__ == "__main__":
    main()
