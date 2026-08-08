import os
import inspect
import sys
import asyncio
import torch

# Ensure backend directory is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from muzero_nets import MuZeroModels
from latent_mcts import LatentMCTS

def main():
    print("=== Testing Latent MCTS Search Engine ===")

    board_size = 5
    action_space_size = 25
    legal_actions = list(range(action_space_size))

    # 1. Instantiate PyTorch MuZero Model
    model = MuZeroModels(board_size=board_size, action_space_size=action_space_size, latent_channels=32, num_res_blocks=2)
    model.eval()

    # 2. Create Dummy Observation Tensor (1, 3, 5, 5)
    obs = torch.randn(1, 3, board_size, board_size, dtype=torch.float32)

    # 3. Run Latent MCTS Search for 50 simulations
    mcts_engine = LatentMCTS(model=model, c_puct=1.25)
    root = asyncio.run(mcts_engine.search(obs, legal_actions, num_simulations=50))

    # 4. Verify Root Node & Children
    assert root.visit_count == 50, f"Expected 50 root visits, got {root.visit_count}"
    assert len(root.children) == action_space_size, f"Expected {action_space_size} children, got {len(root.children)}"

    print(f"PASS: LatentMCTS search completed successfully!")
    print(f"  Total Root Visits: {root.visit_count}")
    print(f"  Total Expanded Children: {len(root.children)}")

    # Format child visit counts for evaluation report
    child_visits_str = "\n".join([
        f"  - Action {act:2d}: {child.visit_count:2d} visits (prior={child.prior:.4f}, Q={child.value():.4f})"
        for act, child in root.children.items()
    ])

    # Capture source code of LatentMCTS.search
    search_src = inspect.getsource(LatentMCTS.search)

    eval_content = f"""================================================================================
PHASE 4 VERIFICATION REPORT (Latent MCTS & Live Inference Integration)
================================================================================

1. LatentMCTS.search() SOURCE CODE (PUCT & Alternating Value Inversion):
--------------------------------------------------------------------------------
{search_src}

2. ROOT NODE CHILD ACTIONS & VISIT COUNTS (50 Simulations on 5x5 Board):
--------------------------------------------------------------------------------
{child_visits_str}

3. EXECUTION STATUS:
--------------------------------------------------------------------------------
ALL TESTS PASSED SUCCESSFULLY! Real PyTorch Latent MCTS executed cleanly.
================================================================================
"""

    root_eval_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase4_eval.txt"))
    with open(root_eval_path, "w", encoding="utf-8") as f:
        f.write(eval_content)

    print(f"\n[Verification] Report written to: {root_eval_path}")

if __name__ == "__main__":
    main()
