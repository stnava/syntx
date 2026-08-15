import subprocess
import os

def main():
    print("Starting 90-Pair Benchmark Orchestrator...")
    print("Resuming 90-Pair Benchmark Orchestrator from Pair 4...")
    for pair_idx in range(4, 90):
        for model_name in ["ants_syn", "syn"]:
            print(f"\n[{pair_idx}/90] Launching Pair {pair_idx} with {model_name}...")
            cmd = [
                "python", "scripts/run_single_pair.py", 
                "--pair-idx", str(pair_idx), 
                "--model", model_name, 
                "--device", "mps",
                "--save-artifacts"
            ]
            
            # Run the pair (blocking)
            res = subprocess.run(cmd, capture_output=True, text=True)
            
            if res.returncode != 0:
                print(f"❌ Pair {pair_idx} ({model_name}) crashed!")
                print("STDOUT:", res.stdout)
                print("STDERR:", res.stderr)
            else:
                print(f"✅ Pair {pair_idx} ({model_name}) completed successfully.")
            
        # Regenerate HTML report
        print("Regenerating HTML report...")
        subprocess.run(["python", "scripts/generate_html_report.py"], check=True)
        print(f"[{pair_idx}/90] Finished and report updated.")

if __name__ == "__main__":
    main()
