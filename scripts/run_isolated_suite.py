import os
import sys
import subprocess
import json

pairs_to_eval = [57, 0, 1, 2, 41, 45]

print("Starting isolated multi-process registration benchmark suite...", flush=True)
print(f"Evaluation pairs: {pairs_to_eval}\n", flush=True)

for pair_idx in pairs_to_eval:
    out_file = f"results/pair_{pair_idx:03d}_autograd_gaussian.json"
    if os.path.exists(out_file):
        try:
            with open(out_file, "r") as f:
                d = json.load(f)
                if d.get("status") == "SUCCESS":
                    sym = d["syntx_dice_sym"]
                    fold = d["syntx_fold"]
                    minj = d["syntx_min_jac"]
                    inve = d["syntx_inv_mean"]
                    t = d["syntx_time"]
                    ants_d = d["ants_baseline"].get("dice_sym", 0.0)
                    print(f"CACHED_RESULT: Pair {pair_idx:02d} | Autograd Sym: {sym:.4f} | Fold: {fold:.4f}% | MinJac: {minj:.4f} | InvErr: {inve:.2f}mm | Time: {t:.1f}s | (ANTs C++ Baseline: {ants_d:.4f})", flush=True)
                    continue
        except Exception:
            pass

    cmd = [sys.executable, "-u", "scripts/run_single_pair_eval.py", str(pair_idx)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout, flush=True)

print("\n--- ALL ISOLATED PROCESS RUNS COMPLETED ---", flush=True)
