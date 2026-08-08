import os
import random
import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from hex_env import HexEnv
from muzero_nets import MuZeroModels
from latent_mcts import LatentMCTS

def train_self_play(num_games: int = 5, num_simulations: int = 25, board_size: int = 5):
    print(f"=== Starting MuZero Self-Play Training ({num_games} games, {board_size}x{board_size} board) ===")
    
    action_space_size = board_size * board_size
    env = HexEnv(board_size=board_size)
    model = MuZeroModels(board_size=board_size, action_space_size=action_space_size, latent_channels=32, num_res_blocks=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    trajectory_data = []  # List of (obs, action, target_policy, player)

    for game_idx in range(num_games):
        obs = env.reset()
        game_history = []
        done = False
        step_count = 0

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            legal = env.legal_actions()

            mcts = LatentMCTS(model=model, c_puct=1.25)
            root = asyncio.run(mcts.search(obs_tensor, legal, num_simulations=num_simulations))

            # Build target policy vector from visit counts
            target_policy = np.zeros(action_space_size, dtype=np.float32)
            total_visits = sum(child.visit_count for child in root.children.values())
            
            if total_visits > 0:
                for act, child in root.children.items():
                    target_policy[act] = child.visit_count / total_visits
            else:
                for act in legal:
                    target_policy[act] = 1.0 / len(legal)

            # Sample action proportional to visit counts (or max visits)
            actions = list(root.children.keys())
            visits = [root.children[a].visit_count for a in actions]
            if sum(visits) > 0:
                probs = [v / sum(visits) for v in visits]
                chosen_action = np.random.choice(actions, p=probs)
            else:
                chosen_action = random.choice(legal)

            # Store tuple for training
            player_at_step = env.current_player
            game_history.append({
                "obs": obs,
                "action": chosen_action,
                "target_policy": target_policy,
                "player": player_at_step
            })

            # Step Environment
            obs, reward, done = env.step(chosen_action)
            step_count += 1

        winner = env.winner
        print(f"  Game {game_idx + 1}/{num_games} finished in {step_count} steps. Winner: Player {winner}")

        # Assign outcome-based target values
        for sample in game_history:
            # Target value is +1.0 if player won, -1.0 if player lost
            target_val = 1.0 if sample["player"] == winner else -1.0
            trajectory_data.append((
                sample["obs"],
                sample["action"],
                sample["target_policy"],
                target_val
            ))

    # --- TRAINING STEP ---
    print(f"\nTraining model on {len(trajectory_data)} self-play states...")
    model.train()
    
    batch_obs = torch.tensor(np.array([d[0] for d in trajectory_data]), dtype=torch.float32)
    batch_target_policy = torch.tensor(np.array([d[2] for d in trajectory_data]), dtype=torch.float32)
    batch_target_value = torch.tensor(np.array([d[3] for d in trajectory_data]), dtype=torch.float32).unsqueeze(1)

    # Initial inference pass
    value, reward, policy_logits, latent_state = model.initial_inference(batch_obs)

    # Policy loss: CrossEntropy / KL Loss
    policy_loss = -torch.mean(torch.sum(batch_target_policy * torch.log_softmax(policy_logits, dim=-1), dim=-1))
    # Value loss: Mean Squared Error
    value_loss = torch.mean((value - batch_target_value) ** 2)

    total_loss = policy_loss + value_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print(f"Loss: {total_loss.item():.4f} (Policy: {policy_loss.item():.4f}, Value: {value_loss.item():.4f})")

    # Save model weights
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "model_weights.pth"))
    torch.save(model.state_dict(), save_path)
    print(f"✅ Saved trained model weights to: {save_path}")

if __name__ == "__main__":
    train_self_play(num_games=3, num_simulations=15, board_size=5)
