import os
import sys
import subprocess

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(base_dir, '..'))
    eval_file = os.path.join(root_dir, 'v4_eval.txt')
    
    gen_script = os.path.join(base_dir, "generate_expert_data.py")
    train_script = os.path.join(base_dir, "train_supervised.py")
    
    my_env = os.environ.copy()
    my_env["PYTHONPATH"] = base_dir
    
    print("Generating expert data...")
    res1 = subprocess.run([
        sys.executable, gen_script,
        "--board-size", "5",
        "--num-games", "2",
        "--sims-per-move", "50",
        "--workers", "1",
        "--output", os.path.join(base_dir, "test_data.pkl")
    ], env=my_env, capture_output=True, text=True)
    
    print("Training supervised model...")
    res2 = subprocess.run([
        sys.executable, train_script,
        "--board-size", "5",
        "--dataset", os.path.join(base_dir, "test_data.pkl"),
        "--epochs", "1",
        "--batch-size", "16",
        "--run-id", "v4_test"
    ], env=my_env, capture_output=True, text=True)
    
    output_text = "--- Behavioral Cloning V4 Verification ---\n"
    output_text += "\n[Generation Output]\n" + res1.stdout
    if res1.stderr:
        output_text += "\n[Generation Errors]\n" + res1.stderr
        
    output_text += "\n[Training Output]\n" + res2.stdout
    if res2.stderr:
        output_text += "\n[Training Errors]\n" + res2.stderr
        
    if res1.returncode == 0 and res2.returncode == 0:
        output_text += "\nSUCCESS: V4 Behavioral Cloning Pipeline Verified!"
    else:
        output_text += "\nFAILURE: Pipeline Verification Failed."
        
    with open(eval_file, "w") as f:
        f.write(output_text)
        
    print(output_text)

if __name__ == "__main__":
    main()
