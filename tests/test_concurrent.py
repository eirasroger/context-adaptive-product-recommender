from __future__ import annotations

import sys

from train import concurrent

GB = 1 << 30


def _capacity(cores, free_ram_gb, free_gpu_gb, run_cores=1.5, run_ram_gb=2.0, run_gpu_gb=0.75):
    return concurrent.capacity(
        concurrent.Machine(
            physical_cores=cores,
            free_ram_bytes=int(free_ram_gb * GB),
            free_gpu_bytes=None if free_gpu_gb is None else int(free_gpu_gb * GB),
        ),
        concurrent.Footprint(
            cores=run_cores, ram_bytes=int(run_ram_gb * GB),
            gpu_bytes=None if free_gpu_gb is None else int(run_gpu_gb * GB),
        ),
    )


def test_a_small_machine_trains_one_run_at_a_time():
    parallel, _ = _capacity(cores=2, free_ram_gb=1, free_gpu_gb=None)
    assert parallel == 1


def test_the_scarcest_resource_sets_the_parallelism():
    parallel, limits = _capacity(cores=32, free_ram_gb=100, free_gpu_gb=2)
    assert limits["gpu"] == parallel == 3
    assert limits["cpu"] > parallel and limits["ram"] > parallel


def test_a_large_machine_trains_many_runs_at_once():
    parallel, _ = _capacity(cores=32, free_ram_gb=100, free_gpu_gb=22)
    assert parallel >= 16


def test_without_a_gpu_only_cores_and_memory_count():
    _, limits = _capacity(cores=8, free_ram_gb=16, free_gpu_gb=None)
    assert set(limits) == {"cpu", "ram"}


def test_every_job_runs_and_reports_its_exit_code(tmp_path):
    jobs = [
        concurrent.Job(name=f"job{i}", command=[sys.executable, "-c", f"print('epoch 1'); raise SystemExit({i})"],
                       log=tmp_path / f"job{i}.log")
        for i in range(3)
    ]
    finished = []
    _, exits = concurrent.run_jobs(jobs, "cpu", parallel=2, on_finish=lambda job, code: finished.append(job.name))

    assert exits == {"job0": 0, "job1": 1, "job2": 2}
    assert sorted(finished) == ["job0", "job1", "job2"]
    assert (tmp_path / "job1.log").read_text(encoding="utf-8").strip() == "epoch 1"
