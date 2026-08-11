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
    """Experience Replay Buffer for storing and sampling full self-play games."""
    def __init__(self, capacity: int = 2000):
        self.buffer = deque(maxlen=capacity)

    def push(self, game_history: list):
        self.buffer.append(game_history)

    def sample(self, batch_size: int, num_unroll_steps: int = 5):
        k = min(batch_size, len(self.buffer))
        games = random.choices(self.buffer, k=k)

        batch_obs = []
        batch_actions = []
        batch_policies = []
        batch_values = []

        for game in games:
            start_idx = random.randint(0, len(game) - 1)
            batch_obs.append(game[start_idx]["obs"])
            
            actions, policies, values = [], [], []
            for i in range(num_unroll_steps + 1):
                if start_idx + i < len(game):
                    step = game[start_idx + i]
                    actions.append(step["action"])
                    policies.append(step["target_policy"])
                    values.append(step["target_val"])
                else:
                    # Pad out of bounds
                    actions.append(0)
                    policies.append(np.zeros_like(game[0]["target_policy"]))
                    values.append(0.0)

            batch_actions.append(actions)
            batch_policies.append(policies)
            batch_values.append(values)

        b_obs = torch.tensor(np.array(batch_obs), dtype=torch.float32)
        b_actions = torch.tensor(np.array(batch_actions), dtype=torch.long)
        b_policies = torch.tensor(np.array(batch_policies), dtype=torch.float32)
        b_values = torch.tensor(np.array(batch_values), dtype=torch.float32)

        return b_obs, b_actions, b_policies, b_values

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
    if args.load_weights:
        model.load_state_dict(torch.load(args.load_weights, map_location="cpu"), strict=False)
        print(f"  Loaded weights from {args.load_weights}")

    if args.freeze_conv:
        for param in model.representation.parameters():
            param.requires_grad = False
        for param in model.dynamics.parameters():
            param.requires_grad = False
        print("  Frozen representation and dynamics layers.")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    replay_buffer = ExperienceReplayBuffer(capacity=args.buffer_capacity)

    # Setup directories
    if args.run_id:
        output_dir = os.path.abspath(os.path.join(args.output_dir, "runs", args.run_id))
    else:
        output_dir = os.path.abspath(args.output_dir)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("--- Starting Self-Play & Neural Optimization Loop ---")
    pbar = tqdm(range(args.num_games), desc="MuZero Training Loop", unit="game")
    
    latest_p_loss = 0.0
    latest_v_loss = 0.0

    for game_idx in pbar:
        obs = env.reset()
        game_history = []
        game_history_sym = []
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

            # Symmetry: 180-degree rotation
            rot_obs = np.rot90(obs, k=2, axes=(1, 2)).copy()
            rot_action = (args.board_size * args.board_size - 1) - chosen_action
            rot_policy = target_policy[::-1].copy()

            game_history_sym.append({
                "obs": rot_obs,
                "action": int(rot_action),
                "target_policy": rot_policy,
                "player": player_at_step
            })

            # Step Environment
            obs, reward, done = env.step(chosen_action)
            move_count += 1

        winner = env.winner

        # Push game transitions to Replay Buffer with target values
        for sample, sample_sym in zip(game_history, game_history_sym):
            val = 1.0 if sample["player"] == winner else -1.0
            sample["target_val"] = val
            sample_sym["target_val"] = val
            
        replay_buffer.push(game_history)
        replay_buffer.push(game_history_sym)

        # Progress bar will be updated at the end of the game loop

        # Train model on sampled mini-batches from Replay Buffer using BPTT
        if len(replay_buffer) >= max(1, args.batch_size // 40):
            model.train()
            for epoch in range(args.epochs):
                b_obs, b_actions, b_policies, b_values = replay_buffer.sample(args.batch_size, num_unroll_steps=5)
                val_pred, rw_pred, policy_logits, hidden_state = model.initial_inference(b_obs)

                # Initial step loss
                p_loss = -torch.mean(torch.sum(b_policies[:, 0] * torch.log_softmax(policy_logits, dim=-1), dim=-1))
                v_loss = torch.mean((val_pred.squeeze(-1) - b_values[:, 0]) ** 2)
                
                latest_p_loss = p_loss.item()
                latest_v_loss = v_loss.item()
                
                total_loss = p_loss + v_loss

                # Unrolled steps loss for Dynamics Network (g_theta)
                for i in range(1, 6):
                    action_step = b_actions[:, i-1]
                    val_pred, rw_pred, policy_logits, hidden_state = model.recurrent_inference(hidden_state, action_step)
                    
                    # Scale gradients for hidden state
                    hidden_state.register_hook(lambda grad: grad * 0.5)
                    
                    # Compute masked loss to ignore padded steps
                    mask = (torch.sum(b_policies[:, i], dim=-1) > 0).float()
                    p_loss_step = -torch.mean(torch.sum(b_policies[:, i] * torch.log_softmax(policy_logits, dim=-1), dim=-1))
                    v_loss_step = torch.mean(mask * (val_pred.squeeze(-1) - b_values[:, i]) ** 2)
                    
                    total_loss += p_loss_step + v_loss_step

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

        pbar.set_postfix({
            "Moves": move_count,
            "Buffer": len(replay_buffer),
            "PLoss": f"{latest_p_loss:.3f}",
            "VLoss": f"{latest_v_loss:.3f}"
        })

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
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per game batch (default: 1)")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Game interval for checkpoints (default: 10)")
    parser.add_argument("--output-dir", type=str, default="backend", help="Directory to save weights & checkpoints (default: backend)")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID for versioned checkpoints and weights")
    parser.add_argument("--load-weights", type=str, default=None, help="Path to weights to load before training")
    parser.add_argument("--freeze-conv", action="store_true", help="Freeze representation and dynamics layers")
    return parser.parse_args()

if __name__ == "__main__":
    cli_args = parse_args()
    train_self_play(cli_args)
