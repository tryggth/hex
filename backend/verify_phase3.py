import os
import inspect
import sys
import torch

# Ensure backend directory is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from muzero_nets import MuZeroModels, DynamicsNetwork

def main():
    print("=== Testing MuZero PyTorch Neural Networks ===")

    board_size = 5
    action_space_size = 25
    latent_channels = 32
    num_res_blocks = 2

    # 1. Instantiate MuZeroModels
    model = MuZeroModels(
        board_size=board_size,
        action_space_size=action_space_size,
        latent_channels=latent_channels,
        num_res_blocks=num_res_blocks
    )
    model.eval()

    # 2. Count total trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    # 3. Test initial_inference with dummy observation (batch_size=1, 3, 5, 5)
    dummy_obs = torch.randn(1, 3, board_size, board_size, dtype=torch.float32)
    val_init, rw_init, pol_init, latent_state = model.initial_inference(dummy_obs)

    # 4. Test recurrent_inference with action = 12
    action = torch.tensor([12], dtype=torch.long)
    val_rec, rw_rec, pol_rec, next_latent_state = model.recurrent_inference(latent_state, action)

    # Verify tensor shapes
    assert latent_state.shape == (1, latent_channels, board_size, board_size), f"Invalid latent shape: {latent_state.shape}"
    assert pol_init.shape == (1, action_space_size), f"Invalid policy shape: {pol_init.shape}"
    assert val_init.shape == (1, 1), f"Invalid value shape: {val_init.shape}"
    assert rw_init.shape == (1, 1), f"Invalid reward shape: {rw_init.shape}"
    assert next_latent_state.shape == (1, latent_channels, board_size, board_size), f"Invalid next latent shape: {next_latent_state.shape}"

    print("PASS: initial_inference & recurrent_inference executed successfully without dimensionality mismatch!")
    print(f"  Latent State Shape: {latent_state.shape}")
    print(f"  Policy Logits Shape: {pol_init.shape}")
    print(f"  Value Shape: {val_init.shape}")
    print(f"  Reward Shape: {rw_rec.shape}")
    print(f"  Next Latent State Shape: {next_latent_state.shape}")

    # Capture source code of DynamicsNetwork.forward
    dynamics_forward_src = inspect.getsource(DynamicsNetwork.forward)

    eval_content = f"""================================================================================
PHASE 3 VERIFICATION REPORT (MuZero Neural Network Architecture)
================================================================================

1. TOTAL TRAINABLE PARAMETERS:
--------------------------------------------------------------------------------
Total Model Parameters: {total_params:,}

2. DynamicsNetwork.forward() SOURCE CODE (Action Concatenation Logic):
--------------------------------------------------------------------------------
{dynamics_forward_src}

3. TENSOR SHAPE VERIFICATION RESULTS:
--------------------------------------------------------------------------------
Initial Inference:
  - Input Observation Shape:  {tuple(dummy_obs.shape)}
  - Latent State Shape:       {tuple(latent_state.shape)}
  - Policy Logits Shape:      {tuple(pol_init.shape)}
  - Value Shape:              {tuple(val_init.shape)}
  - Initial Reward Shape:     {tuple(rw_init.shape)}

Recurrent Inference (Action = 12):
  - Input Action Tensor:      {tuple(action.shape)} (value=12)
  - Next Latent State Shape:  {tuple(next_latent_state.shape)}
  - Predicted Reward Shape:   {tuple(rw_rec.shape)}
  - Recurrent Policy Shape:   {tuple(pol_rec.shape)}
  - Recurrent Value Shape:    {tuple(val_rec.shape)}

4. EXECUTION STATUS:
--------------------------------------------------------------------------------
ALL TESTS PASSED SUCCESSFULLY! No dimensionality mismatch errors detected.
================================================================================
"""

    root_eval_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase3_eval.txt"))
    with open(root_eval_path, "w", encoding="utf-8") as f:
        f.write(eval_content)

    print(f"\n[Verification] Report written to: {root_eval_path}")

if __name__ == "__main__":
    main()
