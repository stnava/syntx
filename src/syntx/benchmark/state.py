"""
Restart state for the grid suite (``syntx.benchmark.runner``).

A JSON file holds ``{'created_at', 'updated_at', 'completed_tasks': {task_id: record}}``;
each record carries ``status`` 'SUCCESS' or 'FAILED'. The file is rewritten after every
recorded task via a temporary file + ``os.replace``, so an interrupted run leaves either the
old or the new file, and a rerun skips tasks already recorded as 'SUCCESS'.
"""

import os
import json
import time
import tempfile
from typing import Dict, Any, Optional, List


class StateTracker:
    """Persistent task-completion state backed by one JSON file.

    Parameters
    ----------
    state_file : str, default 'docs/provenance/benchmark_state.json'
        Path of the state file (made absolute; its directory is created). An existing file
        is loaded; an unreadable one is reported (printed) and replaced by a fresh state on
        the next save.

    Attributes
    ----------
    state : dict
        The loaded state (see module docstring).
    """

    def __init__(self, state_file: str = "docs/provenance/benchmark_state.json"):
        self.state_file = os.path.abspath(state_file)
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        self.state: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        """Return the parsed state file, or a fresh empty state if it is missing / unreadable."""
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
        """Stamp ``updated_at`` and write the state atomically (temp file + ``os.replace``)."""
        self.state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        target_dir = os.path.dirname(self.state_file)
        os.makedirs(target_dir, exist_ok=True)

        # Write to temporary file first then atomically replace
        fd, tmp_path = tempfile.mkstemp(dir=target_dir, prefix="state_", suffix=".json")
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=2)
        os.replace(tmp_path, self.state_file)

    def is_completed(self, task_id: str) -> bool:
        """True if ``task_id`` is recorded with status 'SUCCESS' (a 'FAILED' record is rerun)."""
        task_info = self.state.get("completed_tasks", {}).get(task_id)
        if task_info and task_info.get("status") == "SUCCESS":
            return True
        return False

    def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """The stored record for ``task_id`` (successful or failed), or None."""
        return self.state.get("completed_tasks", {}).get(task_id)

    def record_success(self, task_id: str, record: Dict[str, Any]) -> None:
        """Store ``record`` as a success and save.

        Side effect: sets ``record['status'] = 'SUCCESS'`` and ``record['completed_at']`` on
        the caller's dict.
        """
        if "completed_tasks" not in self.state:
            self.state["completed_tasks"] = {}
        record["status"] = "SUCCESS"
        record["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.state["completed_tasks"][task_id] = record
        self.save()

    def record_failure(self, task_id: str, error_msg: str, elapsed: float) -> None:
        """Store ``{'status': 'FAILED', 'error', 'runtime_seconds': elapsed, 'failed_at'}`` for
        ``task_id`` (replacing any earlier record) and save."""
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
        """Number of tasks recorded with status 'SUCCESS'."""
        tasks = self.state.get("completed_tasks", {})
        return sum(1 for t in tasks.values() if t.get("status") == "SUCCESS")

    def reset(self) -> None:
        """Replace the state with an empty one and save it (all task records are lost)."""
        self.state = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "completed_tasks": {}
        }
        self.save()
