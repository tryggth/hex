import os
import sys
import argparse
import pickle
import random
from collections import deque
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from muzero_nets import MuZeroModels

class ExperienceReplayBuffer:
    """Experience Replay Buffer for storing and sampling full self-play games."""
    def __init__(self, capacity: int = 200000):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-size", type=int, default=7)
    parser.add_argument("--dataset", type=str, default="backend/expert_data.pkl")
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--latent-channels", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--run-id", type=str, default="v4_clone")
    parser.add_argument("--use-fcn", action="store_true", help="Use Fully Convolutional Prediction Head")
    parser.add_argument("--input-channels", type=int, default=3, help="Number of input observation channels")
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset}...")
    with open(args.dataset, "rb") as f:
        all_trajectories = pickle.load(f)
        
    replay_buffer = ExperienceReplayBuffer(capacity=len(all_trajectories) + 10)
    for traj in all_trajectories:
        replay_buffer.push(traj)
        
    print(f"Loaded {len(replay_buffer)} trajectories.")

    model = MuZeroModels(
        board_size=args.board_size,
        action_space_size=args.board_size ** 2,
        latent_channels=args.latent_channels,
        num_res_blocks=args.num_blocks,
        input_channels=args.input_channels,
        use_fcn=args.use_fcn
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    steps_per_epoch = max(1, len(replay_buffer) * 20 // args.batch_size)
    
    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", args.run_id)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    model.train()
    
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}/{args.epochs}")
        
        for step in pbar:
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
            
            pbar.set_postfix({"PLoss": f"{latest_p_loss:.3f}", "VLoss": f"{latest_v_loss:.3f}"})
            
        chk_path = os.path.join(checkpoint_dir, f"model_checkpoint_{epoch}.pth")
        torch.save(model.state_dict(), chk_path)
        
    save_path = os.path.join(run_dir, "model_weights.pth")
    torch.save(model.state_dict(), save_path)
    print(f"🏆 Supervised Training complete! Master weights saved to: {save_path}")

if __name__ == "__main__":
    main()
