import os
import inspect
import sys

# Ensure backend directory is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hex_env import HexEnv

def verify_game_1():
    print("=== Testing Player 1 (Red - Vertical) Win on 5x5 Board ===")
    env = HexEnv(board_size=5)
    obs = env.reset()
    assert obs.shape == (3, 5, 5), f"Expected obs shape (3, 5, 5), got {obs.shape}"

    # P1 plays col 0 from row 0 to 4 (indices 0, 5, 10, 15, 20)
    # P2 plays col 1..4 in row 0 (indices 1, 2, 3, 4)
    p1_moves = [0, 5, 10, 15, 20]
    p2_moves = [1, 2, 3, 4]

    done = False
    reward = 0.0
    turn = 0

    for i in range(len(p1_moves)):
        # P1 move
        obs, reward, done = env.step(p1_moves[i])
        turn += 1
        if done:
            break
        # P2 move
        obs, reward, done = env.step(p2_moves[i])
        turn += 1
        if done:
            break

    assert done == True, "P1 Win Test Failed: done is False"
    assert reward == 1.0, f"P1 Win Test Failed: reward is {reward}"
    assert env.winner == 1, f"P1 Win Test Failed: winner is {env.winner}"
    assert obs.shape == (3, 5, 5), f"P1 Win Test Failed: obs shape is {obs.shape}"
    print(f"PASS: Player 1 won in {turn} turns with reward={reward}, winner={env.winner}, obs shape={obs.shape}")
    return "PASS"

def verify_game_2():
    print("\n=== Testing Player 2 (Blue - Horizontal) Win on 5x5 Board ===")
    env = HexEnv(board_size=5)
    obs = env.reset()
    assert obs.shape == (3, 5, 5), f"Expected obs shape (3, 5, 5), got {obs.shape}"

    # P1 plays col 0 from row 1 to 4; P2 plays row 0 from col 0 to 4 (indices 0, 1, 2, 3, 4)
    p1_moves = [5, 10, 15, 20, 24]
    p2_moves = [0, 1, 2, 3, 4]

    done = False
    reward = 0.0
    turn = 0

    for i in range(len(p2_moves)):
        # P1 move
        obs, reward, done = env.step(p1_moves[i])
        turn += 1
        if done:
            break
        # P2 move
        obs, reward, done = env.step(p2_moves[i])
        turn += 1
        if done:
            break

    assert done == True, "P2 Win Test Failed: done is False"
    assert reward == 1.0, f"P2 Win Test Failed: reward is {reward}"
    assert env.winner == 2, f"P2 Win Test Failed: winner is {env.winner}"
    assert obs.shape == (3, 5, 5), f"P2 Win Test Failed: obs shape is {obs.shape}"
    print(f"PASS: Player 2 won in {turn} turns with reward={reward}, winner={env.winner}, obs shape={obs.shape}")
    return "PASS"

def main():
    res1 = verify_game_1()
    res2 = verify_game_2()

    # Capture source code of required functions
    step_src = inspect.getsource(HexEnv.step)
    get_obs_src = inspect.getsource(HexEnv.get_observation)

    eval_content = f"""================================================================================
PHASE 2 VERIFICATION REPORT (HexEnv Simulator)
================================================================================

1. HexEnv.step() SOURCE CODE:
--------------------------------------------------------------------------------
{step_src}

2. HexEnv.get_observation() SOURCE CODE:
--------------------------------------------------------------------------------
{get_obs_src}

3. SIMULATED GAME VERIFICATION RESULTS:
--------------------------------------------------------------------------------
Player 1 (Vertical - Red) Win Test: {res1}
Player 2 (Horizontal - Blue) Win Test: {res2}
Observation Tensor Shape: (3, 5, 5) verified.
Overall Status: ALL TESTS PASSED SUCCESSFULLY!
================================================================================
"""

    root_eval_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase2_eval.txt"))
    with open(root_eval_path, "w", encoding="utf-8") as f:
        f.write(eval_content)

    print(f"\n[Verification] Report written to: {root_eval_path}")

if __name__ == "__main__":
    main()
