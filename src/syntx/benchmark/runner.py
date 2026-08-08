"""
Main orchestration engine for syntx benchmark suite.

Manages process-isolated execution, automatic state resumption, progress logging,
and structured JSON provenance output across Phase 1, Phase 2, and Phase 3.
"""

import os
import sys
import json
import time
import tempfile
import subprocess
from typing import List, Dict, Any, Optional

from .state import StateTracker
from .grid import get_phase1_tasks, get_phase2_tasks


def run_single_task_isolated(task_def: Dict[str, Any]) -> Dict[str, Any]:
    """Runs a benchmark task in an isolated worker process."""
    tmp_dir = tempfile.mkdtemp(prefix="syntx_bench_")
    task_json_path = os.path.join(tmp_dir, "task.json")
    out_json_path = os.path.join(tmp_dir, "out.json")

    with open(task_json_path, 'w', encoding='utf-8') as f:
        json.dump(task_def, f, indent=2)

    cmd = [
        sys.executable, "-m", "syntx.benchmark.worker",
        "--task-json", task_json_path,
        "--out-json", out_json_path
    ]

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3600)
        if proc.returncode == 0 and os.path.exists(out_json_path):
            with open(out_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {
                'task_id': task_def.get('task_id'),
                'status': 'FAILED',
                'error': f"Worker exited with code {proc.returncode}: {proc.stderr[:300]}",
                'runtime_seconds': 0.0
            }
    except Exception as e:
        return {
            'task_id': task_def.get('task_id'),
            'status': 'FAILED',
            'error': str(e),
            'runtime_seconds': 0.0
        }
    finally:
        # Cleanup temporary task JSONs
        try:
            if os.path.exists(task_json_path):
                os.remove(task_json_path)
            if os.path.exists(out_json_path):
                os.remove(out_json_path)
            os.rmdir(tmp_dir)
        except Exception:
            pass


def update_progress_report(tracker: StateTracker, report_path: str = "docs/BENCHMARKING_PROGRESS_REPORT.md") -> None:
    """Writes clean Markdown progress report summarizing state."""
    state = tracker.state
    completed = state.get("completed_tasks", {})
    
    total = len(completed)
    successes = [v for v in completed.values() if v.get("status") == "SUCCESS"]
    
    lines = [
        "# syntx Benchmark Suite Progress Report\n",
        f"**Last Updated**: {state.get('updated_at', 'N/A')}\n",
        f"**Total Tasks Recorded**: {total}\n",
        f"**Successfully Completed**: {len(successes)}\n\n",
        "## Recent Results\n\n",
        "| Task ID | Dataset | Model | Regularizer | Fast Smooth | Symmetric Dice | Folding (%) | Runtime (s) |\n",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    ]

    for item in list(successes)[-20:]: # show last 20
        tid = item.get('task_id', 'N/A')
        ds = item.get('dataset', 'N/A')
        model = item.get('model', 'N/A')
        reg = item.get('regularizer', 'N/A')
        fast = item.get('fast_smooth', False)
        dice = item.get('dice_sym', 0.0)
        fold = item.get('folding_pct', 0.0)
        rt = item.get('runtime_seconds', 0.0)
        lines.append(f"| `{tid}` | {ds} | {model} | {reg} | {fast} | {dice:.4f} | {fold:.4f}% | {rt:.2f}s |\n")

    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def run_benchmark_suite(
    output_dir: str = "docs/provenance",
    state_file: str = "docs/provenance/benchmark_state.json",
    phases: Optional[List[int]] = None,
    force_restart: bool = False
) -> Dict[str, Any]:
    """Runs the syntx restartable benchmark suite across specified phases.

    Args:
        output_dir: Directory for storing provenance artifacts and progress reports.
        state_file: JSON file path tracking persistent restart state.
        phases: List of phases to execute (defaults to [1, 2]).
        force_restart: If True, resets existing state and re-runs all tasks.

    Returns:
        Dictionary containing all completed results and benchmark summary statistics.
    """
    if phases is None:
        phases = [1, 2]

    tracker = StateTracker(state_file=state_file)
    if force_restart:
        print("[run_benchmark_suite] Force restart requested. Resetting state.")
        tracker.reset()

    # Collect tasks
    all_tasks = []
    if 1 in phases:
        all_tasks.extend(get_phase1_tasks())
    if 2 in phases:
        all_tasks.extend(get_phase2_tasks())

    total_tasks = len(all_tasks)
    print(f"\n==================================================================")
    print(f" SYNTHESIS & REGISTRATION (syntx) BENCHMARK SUITE")
    print(f" Total Tasks Queued: {total_tasks} across Phases: {phases}")
    print(f" State File: {state_file}")
    print(f"==================================================================\n")

    completed_count = 0
    skipped_count = 0

    for idx, task_def in enumerate(all_tasks, 1):
        task_id = task_def['task_id']
        ds = task_def['dataset']
        cfg_id = task_def['config']['id']

        if tracker.is_completed(task_id):
            skipped_count += 1
            rec = tracker.get_result(task_id)
            dice_sym = rec.get('dice_sym', 0.0) if rec else 0.0
            print(f"[{idx}/{total_tasks}] SKIPPED (Already Completed) | `{task_id}` | Dice_sym: {dice_sym:.4f}")
            continue

        print(f"[{idx}/{total_tasks}] RUNNING | Task: `{task_id}`...", flush=True)
        record = run_single_task_isolated(task_def)

        if record.get('status') == 'SUCCESS':
            tracker.record_success(task_id, record)
            completed_count += 1
            dice_sym = record.get('dice_sym', 0.0)
            folding = record.get('folding_pct', 0.0)
            rt = record.get('runtime_seconds', 0.0)
            print(f"  --> SUCCESS | Dice_sym: {dice_sym:.4f} | Folding: {folding:.4f}% | Time: {rt:.2f}s", flush=True)
        else:
            err = record.get('error', 'Unknown Error')
            tracker.record_failure(task_id, err, record.get('runtime_seconds', 0.0))
            print(f"  --> FAILED | Error: {err}", flush=True)

        update_progress_report(tracker)

    print("\n==================================================================")
    print(f" BENCHMARK SUITE RUN COMPLETE")
    print(f" Newly Executed: {completed_count} | Skipped: {skipped_count} | Total: {total_tasks}")
    print(f"==================================================================\n")

    return tracker.state
