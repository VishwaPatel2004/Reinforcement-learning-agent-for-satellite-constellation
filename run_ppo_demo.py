"""
run_ppo_demo.py
Master script: Train Safe PPO if needed -> Run proof -> Launch dashboard
Usage:
  python run_ppo_demo.py           # Full pipeline
  python run_ppo_demo.py --train   # Force re-train
  python run_ppo_demo.py --proof   # Only run proof
  python run_ppo_demo.py --dash    # Only dashboard
"""
import os, sys, subprocess
import config as cfg

print("=" * 70)
print("  SATELLITE COLLISION AVOIDANCE - SAFE PPO DEMO")
print("  DA-IICT | CMDP + PPO + Live 3D Dashboard")
print("=" * 70)

force_train = "--train" in sys.argv
only_proof  = "--proof" in sys.argv
only_dash   = "--dash"  in sys.argv

model_path = None
run_txt    = os.path.join(cfg.MODELS_DIR, "latest_safe_run.txt")

if os.path.exists(run_txt):
    with open(run_txt) as f:
        lines = f.read().strip().split("\n")
    if len(lines) >= 2 and os.path.exists(lines[1]):
        model_path = lines[1]
        print(f"[OK] Found trained Safe PPO model: {model_path}")
    else:
        print("[WARN] latest_safe_run.txt found but model file missing. Will retrain.")
else:
    print("[WARN] No trained Safe PPO model found.")

if not only_proof and not only_dash:
    if model_path is None or force_train:
        print("\n--- STEP 1: Training Safe PPO model (CMDP + Lagrangian) ---")
        ret = subprocess.run([sys.executable, "train_safe.py"], cwd=os.path.dirname(os.path.abspath(__file__)))
        if ret.returncode != 0:
            print("[ERROR] Training failed!")
            sys.exit(1)
        with open(run_txt) as f:
            lines = f.read().strip().split("\n")
        model_path = lines[1]
    else:
        print("\n[OK] STEP 1: Skipping training - model already exists.")

if not only_dash:
    print("\n--- STEP 2: Running 3-scenario collision avoidance proof ---")
    ret = subprocess.run([sys.executable, "prove_agent_works.py"], cwd=os.path.dirname(os.path.abspath(__file__)))
    if ret.returncode != 0:
        print("[WARN] Proof script had errors. Continuing to dashboard...")

if not only_proof:
    print("\n--- STEP 3: Launching Live Dashboard with AI active ---")
    print("  Open http://localhost:8050 in your browser")
    print("  Press Ctrl+C to stop")
    subprocess.run([sys.executable, "live_dashboard.py"], cwd=os.path.dirname(os.path.abspath(__file__)))
