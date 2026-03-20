import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List


class BufferedLineWriter:
    def __init__(self, flush_records: int = 200):
        self.flush_records = max(int(flush_records), 1)
        self._buffers: Dict[str, List[str]] = defaultdict(list)
        self._pending_records = 0
        self._metrics = {
            "trace_buffer_flush_cnt": 0,
            "trace_buffer_records_written": 0,
            "trace_buffer_max_pending_records": 0,
            "trace_buffer_dropped_cnt": 0,
            "trace_buffer_flush_wall_time_ms_sum": 0.0,
        }

    def _update_max_pending(self) -> None:
        self._metrics["trace_buffer_max_pending_records"] = max(
            int(self._metrics["trace_buffer_max_pending_records"]),
            int(self._pending_records),
        )

    def append_text(self, path: str, line: str) -> None:
        text = line if line.endswith("\n") else f"{line}\n"
        self._buffers[path].append(text)
        self._pending_records += 1
        self._update_max_pending()
        if len(self._buffers[path]) >= self.flush_records:
            self.flush_path(path)

    def append_json(self, path: str, record: Dict[str, Any]) -> None:
        try:
            line = json.dumps(record, ensure_ascii=False)
        except Exception:
            self._metrics["trace_buffer_dropped_cnt"] += 1
            raise
        self.append_text(path, line)

    def flush_path(self, path: str) -> None:
        lines = list(self._buffers.get(path, []))
        if len(lines) == 0:
            return

        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        t0 = time.perf_counter()
        with open(path, "a", encoding="utf-8") as f:
            f.write("".join(lines))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self._buffers[path].clear()
        self._pending_records -= len(lines)
        if len(self._buffers[path]) == 0:
            self._buffers.pop(path, None)

        self._metrics["trace_buffer_flush_cnt"] += 1
        self._metrics["trace_buffer_records_written"] += len(lines)
        self._metrics["trace_buffer_flush_wall_time_ms_sum"] += elapsed_ms

    def flush_all(self) -> None:
        for path in list(self._buffers.keys()):
            self.flush_path(path)

    def close(self) -> None:
        self.flush_all()

    def get_metrics(self) -> Dict[str, Any]:
        metrics = dict(self._metrics)
        metrics["trace_buffer_pending_records"] = int(self._pending_records)
        metrics["trace_buffer_pending_paths"] = int(len(self._buffers))
        return metrics
