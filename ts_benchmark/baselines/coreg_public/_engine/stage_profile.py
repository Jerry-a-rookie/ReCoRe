from __future__ import annotations

import contextlib
import os
import time
from collections import defaultdict

import torch


_records: dict[str, list[dict]] = defaultdict(list)
_event_records: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)


def enabled() -> bool:
    return os.environ.get("COREG_PROFILE_STAGES") == "1"


def memory_mode() -> bool:
    return os.environ.get("COREG_PROFILE_STAGE_MEMORY") == "1"


def event_mode() -> bool:
    return os.environ.get("COREG_PROFILE_STAGE_EVENTS") == "1"


@contextlib.contextmanager
def stage(name: str):
    if not enabled():
        yield
        return
    if memory_mode() and torch.cuda.is_available():
        device = torch.cuda.current_device()
        torch.cuda.synchronize(device)
        before = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        with torch.profiler.record_function(name):
            yield
        torch.cuda.synchronize(device)
        _records[name].append(
            {
                "wall_ms": (time.perf_counter() - started) * 1000.0,
                "allocated_before_bytes": before,
                "allocated_after_bytes": torch.cuda.memory_allocated(device),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            }
        )
        return
    if event_mode() and torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.profiler.record_function(name):
            yield
        end.record()
        _event_records[name].append((start, end))
        return
    with torch.profiler.record_function(name):
        yield


def drain_records() -> dict[str, list[dict]]:
    output = dict(_records)
    _records.clear()
    return output


def drain_event_records() -> dict[str, list[float]]:
    output = {
        name: [float(start.elapsed_time(end)) for start, end in values]
        for name, values in _event_records.items()
    }
    _event_records.clear()
    return output



