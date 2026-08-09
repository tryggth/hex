from hex_env import HexEnv
from classic_mcts import ClassicMCTS
import sys
import os

def main():
    env = HexEnv(board_size=5)
    mcts = ClassicMCTS()
    
    print("Running Classic MCTS (1000 simulations) on 5x5 board...")
    best_action, root = mcts.search(env, num_simulations=1000)
    
    output = f"Chosen Action: {best_action}\nRoot Children Visit Counts:\n"
    for act, child in sorted(root.children.items()):
        output += f"Action {act}: {child.visit_count} visits, Value Sum: {child.value_sum:.2f}\n"
        
    print(output)
    
    # Write to root directory
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "arena_phase1_eval.txt")
    with open(output_path, "w") as f:
        f.write(output)
        
if __name__ == "__main__":
    main()
