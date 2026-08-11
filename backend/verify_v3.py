import os
import sys
import subprocess
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from muzero_nets import MuZeroModels

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(base_dir, '..'))
    eval_file = os.path.join(root_dir, 'v3_eval.txt')
    
    # 1. Initialize 5x5 MuZeroModels and save to dummy_5x5.pth
    model = MuZeroModels(board_size=5, action_space_size=25, latent_channels=96, num_res_blocks=8)
    dummy_path = os.path.join(base_dir, 'dummy_5x5.pth')
    torch.save(model.state_dict(), dummy_path)
    
    # 2. Execute transplant.py
    transplant_script = os.path.join(base_dir, 'transplant.py')
    result = subprocess.run(
        [sys.executable, transplant_script, '--source', dummy_path, '--target-board', '7'],
        capture_output=True, text=True
    )
    
    output_text = "--- Transplant Surgery Output ---\n"
    output_text += result.stdout
    if result.stderr:
        output_text += "\nErrors:\n" + result.stderr
        
    # 4. Verify symmetry math
    board_size = 5
    obs = np.zeros((3, board_size, board_size), dtype=np.float32)
    obs[0, 0, 2] = 1.0
    action = 2
    target_policy = np.zeros(board_size * board_size, dtype=np.float32)
    target_policy[2] = 1.0
    
    rot_obs = np.rot90(obs, k=2, axes=(1, 2)).copy()
    rot_action = (board_size * board_size - 1) - action
    rot_policy = target_policy[::-1].copy()
    
    try:
        assert rot_action == 22, f"Expected action 22, got {rot_action}"
        assert rot_policy[22] == 1.0, f"Expected policy[22] to be 1.0, got {rot_policy[22]}"
        assert rot_obs[0, 4, 2] == 1.0, "Observation rotation failed"
        output_text += "\n--- Symmetry Math Verification ---\nSUCCESS: Math and rotation logic is solid for 5x5."
    except AssertionError as e:
        output_text += f"\n--- Symmetry Math Verification ---\nFAILURE: {e}"
        
    # 3. Output to v3_eval.txt
    with open(eval_file, 'w') as f:
        f.write(output_text)
        
    print(output_text)

if __name__ == "__main__":
    main()
