import os
import numpy as np

def verify_symmetry():
    board_size = 7
    # 1. Create a mock observation (3, 7, 7)
    obs = np.zeros((3, board_size, board_size), dtype=np.float32)
    # Put a player 1 stone at (0, 1) and player 2 stone at (6, 5)
    obs[0, 0, 1] = 1.0
    obs[1, 6, 5] = 1.0
    obs[2, :, :] = 1.0 # Player 1's turn
    
    # 2. Mock target policy
    target_policy = np.zeros(board_size * board_size, dtype=np.float32)
    target_policy[2] = 0.8
    target_policy[48] = 0.2
    
    # 3. Mock chosen action
    chosen_action = 2

    # --- Apply 180-degree rotation logic from train.py ---
    obs_sym = np.flip(obs, axis=(1, 2)).copy()
    policy_2d = target_policy.reshape((board_size, board_size))
    target_policy_sym = np.flip(policy_2d, axis=(0, 1)).flatten().copy()
    chosen_action_sym = (board_size * board_size - 1) - chosen_action

    # --- Assertions ---
    # The stone at (0, 1) should move to (6, 5)
    assert obs_sym[0, 6, 5] == 1.0, "Player 1 stone didn't rotate correctly"
    # The stone at (6, 5) should move to (0, 1)
    assert obs_sym[1, 0, 1] == 1.0, "Player 2 stone didn't rotate correctly"
    
    # Policy at index 2 (row 0, col 2) should move to index 46 (row 6, col 4)
    assert target_policy_sym[46] == 0.8, f"Policy didn't rotate correctly, got {target_policy_sym[46]} at 46"
    assert target_policy_sym[0] == 0.2, f"Policy didn't rotate correctly, got {target_policy_sym[0]} at 0"
    
    # Action 2 should move to 46
    assert chosen_action_sym == 46, "Action didn't rotate correctly"
    
    # Output success to v2_eval.txt
    output_msg = (
        "180-Degree Symmetry Augmentation Verification SUCCESS\n"
        "-----------------------------------------------------\n"
        f"Original action: {chosen_action} -> Rotated action: {chosen_action_sym}\n"
        "All assertions passed correctly. The math and rotation logic is solid."
    )
    
    out_file = os.path.join(os.path.dirname(__file__), "v2_eval.txt")
    with open(out_file, "w") as f:
        f.write(output_msg)
    
    print(output_msg)

if __name__ == "__main__":
    verify_symmetry()
