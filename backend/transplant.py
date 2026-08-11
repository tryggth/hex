import argparse
import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from muzero_nets import MuZeroModels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--target-board", type=int, default=7)
    parser.add_argument("--latent-channels", type=int, default=96)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--output", type=str, default="backend/transplanted_weights.pth")
    args = parser.parse_args()

    source_state_dict = torch.load(args.source, map_location="cpu")
    
    target_action_space = args.target_board ** 2
    target_model = MuZeroModels(
        board_size=args.target_board,
        action_space_size=target_action_space,
        latent_channels=args.latent_channels,
        num_res_blocks=args.num_blocks
    )
    
    target_state_dict = target_model.state_dict()
    
    copied = 0
    skipped = []
    
    for key in target_state_dict.keys():
        if key in source_state_dict:
            if target_state_dict[key].shape == source_state_dict[key].shape:
                target_state_dict[key] = source_state_dict[key]
                copied += 1
            else:
                skipped.append(key)
        else:
            skipped.append(key)
            
    print(f"Copied {copied} tensors. Skipped {len(skipped)} tensors due to shape mismatch.")
    if skipped:
        print("Skipped keys:")
        for k in skipped:
            print(f"  - {k}")
            
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(target_state_dict, args.output)

if __name__ == "__main__":
    main()
