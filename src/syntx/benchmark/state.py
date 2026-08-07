"""
State management and restartability tracking for syntx benchmark suite.

Maintains an atomic JSON state file recording completed benchmark tasks,
allowing interrupted runs to resume seamlessly without repeating work.
"""

import os
import json
import time
import tempfile
from typing import Dict, Any, Optional, List


class StateTracker:
    """Manages persistent benchmark execution state on disk."""

    def __init__(self, state_file: str = "docs/provenance/benchmark_state.json"):
        self.state_file = os.path.abspath(state_file)
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        self.state: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        """Loads state from JSON file if present, else initializes new state."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[StateTracker] Warning: Failed to parse {self.state_file}: {e}. Initializing fresh state.")
        return {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "completed_tasks": {}
        }

    def save(self) -> None:
        """Atomically saves current state to disk."""
        self.state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        target_dir = os.path.dirname(self.state_file)
        os.makedirs(target_dir, exist_ok=True)

        # Write to temporary file first then atomically replace
        fd, tmp_path = tempfile.mkstemp(dir=target_dir, prefix="state_", suffix=".json")
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=2)
        os.replace(tmp_path, self.state_file)

    def is_completed(self, task_id: str) -> bool:
        """Returns True if the task completed successfully."""
        task_info = self.state.get("completed_tasks", {}).get(task_id)
        if task_info and task_info.get("status") == "SUCCESS":
            return True
        return False

    def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Returns result dictionary for a completed task."""
        return self.state.get("completed_tasks", {}).get(task_id)

    def record_success(self, task_id: str, record: Dict[str, Any]) -> None:
        """Records a successfully completed benchmark task."""
        if "completed_tasks" not in self.state:
            self.state["completed_tasks"] = {}
        record["status"] = "SUCCESS"
        record["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.state["completed_tasks"][task_id] = record
        self.save()

    def record_failure(self, task_id: str, error_msg: str, elapsed: float) -> None:
        """Records a failed benchmark task attempt."""
        if "completed_tasks" not in self.state:
            self.state["completed_tasks"] = {}
        self.state["completed_tasks"][task_id] = {
            "status": "FAILED",
            "error": str(error_msg),
            "runtime_seconds": elapsed,
            "failed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        self.save()

    def get_completed_count(self) -> int:
        """Returns count of successfully completed tasks."""
        tasks = self.state.get("completed_tasks", {})
        return sum(1 for t in tasks.values() if t.get("status") == "SUCCESS")

    def reset(self) -> None:
        """Clears state file completely."""
        self.state = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "completed_tasks": {}
        }
        self.save()
