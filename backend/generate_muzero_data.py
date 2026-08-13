import argparse
import asyncio
import os
import sys
import pickle
import numpy as np
import multiprocessing
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hex_env import HexEnv
from muzero_nets import MuZeroModels
from latent_mcts import LatentMCTS

def generate_games_worker(args):
    worker_idx, num_games, board_size, sims_per_move, temp_moves, run_id = args
    
    # Load Model in each worker
    device = torch.device("cpu")
    weights_path = os.path.join(os.path.dirname(__file__), "runs", run_id, "model_weights.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"No weights found at {weights_path}")
        
    model = MuZeroModels(
        board_size=board_size,
        action_space_size=board_size * board_size,
        latent_channels=96,
        num_res_blocks=8,
        input_channels=3,
        use_fcn=False
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    async def run_game():
        env = HexEnv(board_size=board_size)
        mcts = LatentMCTS(model=model)
        trajectory = []
        
        while env.winner == 0:
            legal = env.legal_actions()
            if not legal:
                break
                
            move_count = env.moves_made
            temp = 1.0 if move_count < temp_moves else 0.0
            
            # Use 3-channel obs for MCTS to act
            obs_3c = env.get_observation(v5_features=False)
            obs_tensor = torch.tensor(obs_3c, dtype=torch.float32).unsqueeze(0)
            
            # CRITICAL: add dirichlet noise to root
            root = await mcts.search(
                initial_state_tensor=obs_tensor,
                legal_actions=legal,
                num_simulations=sims_per_move,
                add_dirichlet_noise=True
            )
            
            actions = list(root.children.keys())
            visits = np.array([root.children[a].visit_count for a in actions], dtype=np.float64)
            
            target_policy = np.zeros(board_size * board_size, dtype=np.float32)
            total_visits = np.sum(visits)
            if total_visits > 0:
                for act, child in root.children.items():
                    target_policy[act] = child.visit_count / total_visits
            else:
                for act in legal:
                    target_policy[act] = 1.0 / len(legal)
                    
            if total_visits > 0:
                if temp > 0.0:
                    visits_norm = visits / np.max(visits)
                    visits_pow = visits_norm ** (1.0 / temp)
                    probs = visits_pow / np.sum(visits_pow)
                    probs = probs / np.sum(probs)
                    best_action = np.random.choice(actions, p=probs)
                else:
                    best_action = max(actions, key=lambda a: root.children[a].visit_count)
            else:
                best_action = np.random.choice(legal)
                
            # CRITICAL: Save 5-channel V5 observation
            obs_5c = env.get_observation(v5_features=True).copy()
            player = env.current_player
            
            trajectory.append({
                "obs": obs_5c,
                "action": int(best_action),
                "target_policy": target_policy,
                "player": player
            })
            
            env.step(best_action)
            
        winner = env.winner
        aug_trajectory = []
        for step in trajectory:
            val = 1.0 if step["player"] == winner else -1.0
            step["target_val"] = val
            
            # 180-degree symmetry
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

    # Generate num_games
    all_trajectories = []
    for _ in range(num_games):
        trajs = asyncio.run(run_game())
        all_trajectories.extend(trajs)
        
    return all_trajectories

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True, help="V4 run ID to load weights from")
    parser.add_argument("--board-size", type=int, default=7)
    parser.add_argument("--num-games", type=int, default=500)
    parser.add_argument("--sims-per-move", type=int, default=400)
    parser.add_argument("--temp-moves", type=int, default=6)
    parser.add_argument("--output", type=str, default="backend/expert_data_v5_7x7.pkl")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    
    print(f"Generating {args.num_games} MuZero games using {args.workers} workers...")
    games_per_worker = [args.num_games // args.workers] * args.workers
    for i in range(args.num_games % args.workers):
        games_per_worker[i] += 1
        
    worker_args = [
        (i, games_per_worker[i], args.board_size, args.sims_per_move, args.temp_moves, args.run_id) 
        for i in range(args.workers)
    ]
    
    all_trajectories = []
    if args.workers == 1:
        # Single worker: generate one game at a time with per-game progress
        pbar = tqdm(total=args.num_games, desc="Generating games", unit="game")
        for w_args in worker_args:
            results = generate_games_worker(w_args)
            all_trajectories.extend(results)
            pbar.update(w_args[1])
        pbar.close()
    else:
        with multiprocessing.Pool(args.workers) as pool:
            pbar = tqdm(total=args.num_games, desc="Generating games", unit="game")
            for result in pool.imap_unordered(generate_games_worker, worker_args):
                games_done = len(result) // 2  # each game produces 2 trajectories (original + augmented)
                all_trajectories.extend(result)
                pbar.update(games_done)
            pbar.close()
            
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(all_trajectories, f)
        
    print(f"Saved {len(all_trajectories)} augmented trajectories to {args.output}")

if __name__ == "__main__":
    main()
