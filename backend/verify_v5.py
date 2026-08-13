import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from muzero_nets import MuZeroModels

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(base_dir, '..'))
    eval_file = os.path.join(root_dir, 'v5_eval.txt')
    
    print("Initializing V5 Fully Convolutional Architecture...")
    model = MuZeroModels(
        board_size=7, 
        latent_channels=96, 
        num_res_blocks=8, 
        input_channels=5, 
        use_fcn=True
    )
    
    dummy_obs = torch.randn(1, 5, 7, 7)
    
    print("Running initial inference on dummy observation...")
    value, reward, policy_logits, latent_state = model.initial_inference(dummy_obs)
    
    print(f"policy_logits shape: {policy_logits.shape}")
    print(f"value shape: {value.shape}")
    print(f"latent_state shape: {latent_state.shape}")
    
    assert policy_logits.shape == (1, 49), f"Expected policy shape (1, 49), got {policy_logits.shape}"
    assert value.shape == (1, 1), f"Expected value shape (1, 1), got {value.shape}"
    assert latent_state.shape == (1, 96, 7, 7), f"Expected latent shape (1, 96, 7, 7), got {latent_state.shape}"
    
    output_text = "--- V5 FCN Architecture Verification ---\n"
    output_text += f"Policy Logits Shape: {policy_logits.shape}\n"
    output_text += f"Value Shape: {value.shape}\n"
    output_text += f"Latent State Shape: {latent_state.shape}\n\n"
    output_text += "[Network Summary]\n"
    output_text += str(model) + "\n\n"
    output_text += "SUCCESS: Shapes verified!\n"
    
    with open(eval_file, "w") as f:
        f.write(output_text)
        
    print(f"Saved evaluation to {eval_file}")
    
if __name__ == "__main__":
    main()
