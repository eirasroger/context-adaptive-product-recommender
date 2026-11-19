"""Run independent training processes side by side, as many as this machine fits.

The caller prepares everything that writes to the database (splits, snapshots) and warms
the prepared caches first; the processes started here only read.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

#: Share of free cores, memory and GPU memory that concurrent runs may take.
HEADROOM = 0.8


@dataclass(frozen=True)
class Footprint:
    """What one training run uses once it is in its epoch loop."""

    cores: float
    ram_bytes: int
    gpu_bytes: int | None


@dataclass(frozen=True)
class Machine:
    physical_cores: int
    free_ram_bytes: int
    free_gpu_bytes: int | None


@dataclass(frozen=True)
class Job:
    """One process to run: its name, its command line and the file its output goes to."""

    name: str
    command: list[str]
    log: Path


def capacity(machine: Machine, footprint: Footprint) -> tuple[int, dict[str, int]]:
    """How many runs fit at once, and the limit each resource sets.

    The machine is measured while the measured run is live, so it counts as one of the memory slots.
    """
    limits = {
        "cpu": int(machine.physical_cores * HEADROOM / max(footprint.cores, 1.0)),
        "ram": 1 + int(machine.free_ram_bytes * HEADROOM / max(footprint.ram_bytes, 1)),
    }
    if machine.free_gpu_bytes is not None and footprint.gpu_bytes:
        limits["gpu"] = 1 + int(machine.free_gpu_bytes * HEADROOM / footprint.gpu_bytes)
    return max(1, min(limits.values())), limits


def free_gpu_bytes(device: str) -> int | None:
    """Device-wide free memory; the driver reports other processes' allocations, CUDA's own view on Windows does not."""
    if device != "cuda":
        return None
    try:
        listed = subprocess.run(
            ["nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return int(listed.split()[0]) << 20


def this_machine(device: str) -> Machine:
    import psutil

    return Machine(
        physical_cores=psutil.cpu_count(logical=False) or psutil.cpu_count() or 1,
        free_ram_bytes=psutil.virtual_memory().available,
        free_gpu_bytes=free_gpu_bytes(device),
    )


def _in_epoch_loop(log: Path) -> bool:
    return any(line.lower().startswith("epoch") for line in log.read_text(encoding="utf-8").splitlines())


def measure(process: subprocess.Popen, log: Path, device: str, gpu_before: int | None) -> Footprint | None:
    """The footprint of a run once its first epoch is logged; None if it exits first."""
    import psutil

    while not _in_epoch_loop(log):
        if process.poll() is not None:
            return None
        time.sleep(2)
    probed = psutil.Process(process.pid)
    cores = probed.cpu_percent(interval=10) / 100
    gpu_after = free_gpu_bytes(device)
    return Footprint(
        cores=cores,
        ram_bytes=probed.memory_info().rss,
        gpu_bytes=None if gpu_before is None or gpu_after is None else max(0, gpu_before - gpu_after),
    )


def size_to_machine(process: subprocess.Popen, log: Path, device: str, gpu_before: int | None) -> dict:
    """Measure the first run, then decide how many can train alongside it."""
    footprint = measure(process, log, device, gpu_before)
    if footprint is None:
        return {"parallel": 1, "reason": "the measured run exited before its first epoch"}
    machine = this_machine(device)
    parallel, limits = capacity(machine, footprint)
    return {"parallel": parallel, "limits": limits, "footprint": asdict(footprint), "machine": asdict(machine)}


def sizing_line(sizing: dict) -> str:
    limits = sizing.get("limits")
    if not limits:
        return f"Trained {sizing['parallel']} at once ({sizing.get('reason', '')})."
    return f"Trained {sizing['parallel']} at once; limits " + ", ".join(
        f"{resource} {limit}" for resource, limit in limits.items()
    ) + "."


def _launch(job: Job) -> subprocess.Popen:
    job.log.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(job.command, stdout=job.log.open("w", encoding="utf-8"), stderr=subprocess.STDOUT)


def run_jobs(
    jobs: Sequence[Job],
    device: str,
    parallel: int | None = None,
    on_finish: Callable[[Job, int], None] | None = None,
    poll_seconds: float = 5.0,
) -> tuple[dict, dict[str, int]]:
    """Run every job, sized to the machine from the first one; the sizing and each job's exit code.

    Put the heaviest job first: its footprint sets how many run at once.
    """
    waiting = list(jobs)
    running: dict[str, tuple[Job, subprocess.Popen, float]] = {}
    exits: dict[str, int] = {}
    started_all = time.perf_counter()

    def start(job: Job) -> subprocess.Popen:
        process = _launch(job)
        running[job.name] = (job, process, time.perf_counter())
        print(f"{job.name}: started", flush=True)
        return process

    if parallel is None and waiting:
        gpu_before = free_gpu_bytes(device)
        probe = waiting.pop(0)
        sizing = size_to_machine(start(probe), probe.log, device, gpu_before)
    else:
        sizing = {"parallel": parallel or 1, "reason": "set by the caller"}
    print(f"running {sizing['parallel']} at once ({sizing_line(sizing)})", flush=True)

    while waiting or running:
        while waiting and len(running) < sizing["parallel"]:
            start(waiting.pop(0))
        time.sleep(poll_seconds)
        for name, (job, process, started) in list(running.items()):
            if process.poll() is None:
                continue
            del running[name]
            exits[name] = process.returncode
            elapsed = (time.perf_counter() - started_all) / 60
            done, total = len(exits), len(jobs)
            left = elapsed / done * (total - done)
            print(
                f"[{done}/{total}] {name}: exit {process.returncode}, "
                f"{(time.perf_counter() - started) / 60:.1f} min | {elapsed:.1f} min elapsed"
                + (f" | about {left:.0f} min left" if total > done else ""),
                flush=True,
            )
            if on_finish is not None:
                on_finish(job, process.returncode)
    return sizing, exits
