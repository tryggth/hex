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

def train_self_play(
    num_games: int = 40,
    mcts_simulations_per_move: int = 40,
    epochs_per_game_batch: int = 5,
    learning_rate: float = 1e-3,
    board_size: int = 7
):
    print(f"=== Starting Scaled MuZero Self-Play Training ===")
    print(f"  Board Size: {board_size}x{board_size}")
    print(f"  Games: {num_games}")
    print(f"  MCTS Sims/Move: {mcts_simulations_per_move}")
    print(f"  Epochs: {epochs_per_game_batch}")
    print(f"  Learning Rate: {learning_rate}\n")

    action_space_size = board_size * board_size
    env = HexEnv(board_size=board_size)
    model = MuZeroModels(board_size=board_size, action_space_size=action_space_size, latent_channels=32, num_res_blocks=2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    trajectory_data = []

    # 1. Self-Play Trajectory Collection
    for game_idx in range(num_games):
        obs = env.reset()
        game_history = []
        done = False
        step_count = 0

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            legal = env.legal_actions()

            mcts = LatentMCTS(model=model, c_puct=1.25)
            root = asyncio.run(mcts.search(obs_tensor, legal, num_simulations=mcts_simulations_per_move))

            # Build target policy vector from visit counts
            target_policy = np.zeros(action_space_size, dtype=np.float32)
            total_visits = sum(child.visit_count for child in root.children.values())

            if total_visits > 0:
                for act, child in root.children.items():
                    target_policy[act] = child.visit_count / total_visits
            else:
                for act in legal:
                    target_policy[act] = 1.0 / len(legal)

            # Sample action proportional to visit counts
            actions = list(root.children.keys())
            visits = [root.children[a].visit_count for a in actions]
            if sum(visits) > 0:
                probs = [v / sum(visits) for v in visits]
                chosen_action = np.random.choice(actions, p=probs)
            else:
                chosen_action = random.choice(legal)

            # Record step history
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
        print(f"  Game {game_idx + 1:2d}/{num_games} finished in {step_count:2d} steps. Winner: Player {winner}")

        # Assign target values (+1 for winner, -1 for loser)
        for sample in game_history:
            target_val = 1.0 if sample["player"] == winner else -1.0
            trajectory_data.append((
                sample["obs"],
                sample["action"],
                sample["target_policy"],
                target_val
            ))

    # 2. Neural Network Optimization Pass
    print(f"\n--- Training 7x7 PyTorch MuZero Model on {len(trajectory_data)} self-play states ---")
    model.train()

    batch_obs = torch.tensor(np.array([d[0] for d in trajectory_data]), dtype=torch.float32)
    batch_target_policy = torch.tensor(np.array([d[2] for d in trajectory_data]), dtype=torch.float32)
    batch_target_value = torch.tensor(np.array([d[3] for d in trajectory_data]), dtype=torch.float32).unsqueeze(1)

    loss_logs = []

    for epoch in range(1, epochs_per_game_batch + 1):
        value, reward, policy_logits, latent_state = model.initial_inference(batch_obs)

        # Policy Loss: CrossEntropy / KL Loss
        policy_loss = -torch.mean(torch.sum(batch_target_policy * torch.log_softmax(policy_logits, dim=-1), dim=-1))
        # Value Loss: Mean Squared Error
        value_loss = torch.mean((value - batch_target_value) ** 2)

        total_loss = policy_loss + value_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        log_str = f"  Epoch {epoch}/{epochs_per_game_batch} - Total Loss: {total_loss.item():.4f} | Policy Loss: {policy_loss.item():.4f} | Value Loss: {value_loss.item():.4f}"
        print(log_str)
        loss_logs.append(log_str)

    # Save trained model weights
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "model_weights.pth"))
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Saved trained 7x7 model weights to: {save_path}")

    # Append final training logs to benchmark_eval.txt
    eval_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmark_eval.txt"))
    append_str = "\n6. SCALED 7x7 TRAINING LOGS (40 Games, 5 Epochs):\n--------------------------------------------------------------------------------\n" + "\n".join(loss_logs) + "\n================================================================================\n"
    with open(eval_path, "a", encoding="utf-8") as f:
        f.write(append_str)

    print(f"✅ Appended training logs to: {eval_path}")

if __name__ == "__main__":
    train_self_play(num_games=40, mcts_simulations_per_move=40, epochs_per_game_batch=5, learning_rate=1e-3, board_size=7)
