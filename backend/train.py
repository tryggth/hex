import os
import sys
import random
import argparse
import asyncio
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from hex_env import HexEnv
from muzero_nets import MuZeroModels
from latent_mcts import LatentMCTS

class ExperienceReplayBuffer:
    """Experience Replay Buffer for storing and sampling self-play transitions."""
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, observation: np.ndarray, target_policy: np.ndarray, target_value: float):
        self.buffer.append((observation, target_policy, target_value))

    def sample(self, batch_size: int):
        k = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), size=k, replace=False)
        samples = [self.buffer[i] for i in indices]

        batch_obs = torch.tensor(np.array([s[0] for s in samples]), dtype=torch.float32)
        batch_policy = torch.tensor(np.array([s[1] for s in samples]), dtype=torch.float32)
        batch_value = torch.tensor(np.array([s[2] for s in samples]), dtype=torch.float32).unsqueeze(1)

        return batch_obs, batch_policy, batch_value

    def __len__(self) -> int:
        return len(self.buffer)

def train_self_play(args):
    print(f"=== Starting Configurable MuZero Self-Play Training ===")
    print(f"  Board Size:           {args.board_size}x{args.board_size}")
    print(f"  Self-Play Games:      {args.num_games}")
    print(f"  MCTS Sims / Move:     {args.sims_per_move}")
    print(f"  Residual Blocks:      {args.num_blocks}")
    print(f"  Latent Channels:      {args.latent_channels}")
    print(f"  Learning Rate:        {args.lr}")
    print(f"  Batch Size:           {args.batch_size}")
    print(f"  Replay Capacity:      {args.buffer_capacity}")
    print(f"  Temp Moves:           {args.temp_moves}")
    print(f"  Epochs / Pass:        {args.epochs}")
    print(f"  Checkpoint Interval:  {args.checkpoint_interval}")
    print(f"  Output Directory:     {args.output_dir}\n")

    action_space_size = args.board_size * args.board_size
    env = HexEnv(board_size=args.board_size)
    
    model = MuZeroModels(
        board_size=args.board_size,
        action_space_size=action_space_size,
        latent_channels=args.latent_channels,
        num_res_blocks=args.num_blocks
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    replay_buffer = ExperienceReplayBuffer(capacity=args.buffer_capacity)

    # Setup directories
    output_dir = os.path.abspath(args.output_dir)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("--- Starting Self-Play & Neural Optimization Loop ---")
    pbar = tqdm(range(args.num_games), desc="MuZero Training Loop", unit="game")

    for game_idx in pbar:
        obs = env.reset()
        game_history = []
        done = False
        move_count = 0

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            legal = env.legal_actions()

            mcts = LatentMCTS(model=model, c_puct=1.25)
            root = asyncio.run(mcts.search(
                obs_tensor,
                legal,
                num_simulations=args.sims_per_move,
                add_dirichlet_noise=True,
                dirichlet_alpha=0.3,
                dirichlet_fraction=0.25
            ))

            # Build target policy vector from visit counts
            target_policy = np.zeros(action_space_size, dtype=np.float32)
            total_visits = sum(child.visit_count for child in root.children.values())

            if total_visits > 0:
                for act, child in root.children.items():
                    target_policy[act] = child.visit_count / total_visits
            else:
                for act in legal:
                    target_policy[act] = 1.0 / len(legal)

            # Temperature Sampling:
            # First `temp_moves` moves: sample action probabilistically tau = 1.0
            # After `temp_moves`: greedy action selection argmax(N)
            actions = list(root.children.keys())
            visits = [root.children[a].visit_count for a in actions]

            if move_count < args.temp_moves and sum(visits) > 0:
                probs = [v / sum(visits) for v in visits]
                chosen_action = np.random.choice(actions, p=probs)
            else:
                chosen_action = max(actions, key=lambda a: root.children[a].visit_count)

            # Record step
            player_at_step = env.current_player
            game_history.append({
                "obs": obs,
                "action": chosen_action,
                "target_policy": target_policy,
                "player": player_at_step
            })

            # Step Environment
            obs, reward, done = env.step(chosen_action)
            move_count += 1

        winner = env.winner

        # Push game transitions to Replay Buffer with target values
        for sample in game_history:
            target_val = 1.0 if sample["player"] == winner else -1.0
            replay_buffer.push(sample["obs"], sample["target_policy"], target_val)

        pbar.set_postfix({
            "Moves": move_count,
            "Winner": f"P{winner}",
            "Buffer": len(replay_buffer)
        })

        # Train model on sampled mini-batches from Replay Buffer
        if len(replay_buffer) >= min(args.batch_size, 16):
            model.train()
            for epoch in range(args.epochs):
                b_obs, b_policy, b_value = replay_buffer.sample(args.batch_size)
                val_pred, rw_pred, policy_logits, _ = model.initial_inference(b_obs)

                # Policy & Value Losses
                policy_loss = -torch.mean(torch.sum(b_policy * torch.log_softmax(policy_logits, dim=-1), dim=-1))
                value_loss = torch.mean((val_pred - b_value) ** 2)
                total_loss = policy_loss + value_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

        # Save checkpoint every checkpoint_interval games
        if (game_idx + 1) % args.checkpoint_interval == 0:
            chk_path = os.path.join(checkpoint_dir, f"model_checkpoint_{game_idx + 1}.pth")
            torch.save(model.state_dict(), chk_path)
            tqdm.write(f"  💾 Saved checkpoint: {chk_path}")

    # Save final model weights
    save_path = os.path.join(output_dir, "model_weights.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n🏆 Training complete! Master weights saved to: {save_path}")
    return replay_buffer

def parse_args():
    parser = argparse.ArgumentParser(description="MuZero Self-Play Reinforcement Learning Trainer for Hex")
    parser.add_argument("--board-size", type=int, default=7, help="Grid size of Hex board (default: 7)")
    parser.add_argument("--num-games", type=int, default=300, help="Total self-play games (default: 300)")
    parser.add_argument("--sims-per-move", type=int, default=400, help="MCTS simulations per move (default: 400)")
    parser.add_argument("--num-blocks", type=int, default=8, help="Number of Residual Blocks in CNN (default: 8)")
    parser.add_argument("--latent-channels", type=int, default=96, help="Latent feature channels (default: 96)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Optimizer learning rate (default: 0.001)")
    parser.add_argument("--batch-size", type=int, default=64, help="Replay buffer mini-batch size (default: 64)")
    parser.add_argument("--buffer-capacity", type=int, default=10000, help="Replay buffer capacity (default: 10000)")
    parser.add_argument("--temp-moves", type=int, default=6, help="Temperature sampling moves at start (default: 6)")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per game batch (default: 5)")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Game interval for checkpoints (default: 10)")
    parser.add_argument("--output-dir", type=str, default="backend", help="Directory to save weights & checkpoints (default: backend)")
    return parser.parse_args()

if __name__ == "__main__":
    cli_args = parse_args()
    train_self_play(cli_args)
