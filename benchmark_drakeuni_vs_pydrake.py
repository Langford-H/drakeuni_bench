#!/usr/bin/env python3
"""Benchmark raw pydrake loops against DrakeUni batched stepping.

This mirrors the spirit of ``unilabsim/mujoco_uni_bench``: compare a Python
per-env simulator loop with a batched runtime on the same model. The benchmark
intentionally runs each backend in a separate subprocess because DrakeUni owns
Drake symbol loading and rejects processes that already imported ``pydrake``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_UNILAB_ROOT = ROOT_DIR.parent / "unilab" / "UniLab"
DEFAULT_DRAKEUNI_SRC = ROOT_DIR.parent / "drakeuni" / "src"
DEFAULT_OUTPUT = ROOT_DIR / "outputs" / "results.json"

MODEL_PATHS = {
    "go1": Path("src/unilab/assets/robots/go1/scene_flat.xml"),
    "go2": Path("src/unilab/assets/robots/go2/scene_flat.xml"),
}


@dataclass(frozen=True)
class BenchRecord:
    model: str
    backend: str
    num_envs: int
    nthread: int | str
    nstep: int
    warmup: int
    repeat: int
    nq: int
    nv: int
    nu: int
    nsensordata: int
    setup_ms: float
    step_mean_ms: float
    step_median_ms: float
    reset_mean_ms: float
    reset_median_ms: float
    steps_per_second: float
    reset_envs_per_second: float


def _parse_csv(text: str) -> list[str]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError(f"Expected at least one value in {text!r}")
    return values


def _parse_int_csv(text: str) -> list[int]:
    return [int(value) for value in _parse_csv(text)]


def _effective_drakeuni_threads(num_envs: int, requested: Sequence[int]) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for value in requested:
        effective = min(num_envs, int(value)) if int(value) > 0 else 0
        if effective in seen:
            continue
        seen.add(effective)
        values.append(effective)
    return values


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _time_call(func: Callable[[], Any], *, warmup: int, repeat: int) -> tuple[float, float]:
    for _ in range(warmup):
        func()
    samples: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        func()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return _mean(samples), _median(samples)


def _worker_env(drakeuni_src: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(ROOT_DIR)]
    if drakeuni_src is not None:
        paths.insert(0, str(drakeuni_src))
    existing = env.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def _run_worker(
    *,
    backend: str,
    model_name: str,
    model_file: Path,
    num_envs: int,
    nthread: int,
    sim_dt: float,
    nstep: int,
    warmup: int,
    repeat: int,
    drakeuni_src: Path | None,
) -> BenchRecord:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker-backend",
        backend,
        "--_worker-model-name",
        model_name,
        "--_worker-model-file",
        str(model_file),
        "--num-envs",
        str(num_envs),
        "--nthread",
        str(nthread),
        "--sim-dt",
        str(sim_dt),
        "--nstep",
        str(nstep),
        "--warmup",
        str(warmup),
        "--repeat",
        str(repeat),
    ]
    completed = subprocess.run(
        cmd,
        cwd=ROOT_DIR,
        env=_worker_env(drakeuni_src),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{backend} worker failed for {model_name} envs={num_envs} nthread={nthread}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{backend} worker did not emit JSON\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        ) from exc
    return BenchRecord(**payload)


def _run_drakeuni_worker(args: argparse.Namespace) -> BenchRecord:
    from drakeuni.runtime import DrakeBatchConfig, create_runtime

    t0 = time.perf_counter()
    runtime = create_runtime(
        DrakeBatchConfig(
            model_file=str(args._worker_model_file),
            num_envs=args.num_envs,
            sim_dt=args.sim_dt,
            nthread=args.nthread,
        )
    )
    setup_ms = (time.perf_counter() - t0) * 1000.0

    info = runtime.model_info()
    env_ids = np.arange(args.num_envs, dtype=np.int32)
    qpos = np.broadcast_to(info.home_qpos, (args.num_envs, info.nq)).copy()
    qvel = np.broadcast_to(info.home_qvel, (args.num_envs, info.nv)).copy()
    ctrl = np.zeros((args.num_envs, info.nu), dtype=np.float64)

    reset_mean_ms, reset_median_ms = _time_call(
        lambda: runtime.reset(env_ids, qpos, qvel),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    step_mean_ms, step_median_ms = _time_call(
        lambda: runtime.step(ctrl, args.nstep),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    runtime.close()

    step_seconds = step_mean_ms / 1000.0
    reset_seconds = reset_mean_ms / 1000.0
    return BenchRecord(
        model=args._worker_model_name,
        backend="drakeuni",
        num_envs=args.num_envs,
        nthread=int(runtime.nthread),
        nstep=args.nstep,
        warmup=args.warmup,
        repeat=args.repeat,
        nq=info.nq,
        nv=info.nv,
        nu=info.nu,
        nsensordata=info.nsensordata,
        setup_ms=setup_ms,
        step_mean_ms=step_mean_ms,
        step_median_ms=step_median_ms,
        reset_mean_ms=reset_mean_ms,
        reset_median_ms=reset_median_ms,
        steps_per_second=args.num_envs * args.nstep / step_seconds if step_seconds > 0 else 0.0,
        reset_envs_per_second=args.num_envs / reset_seconds if reset_seconds > 0 else 0.0,
    )


def _run_pydrake_worker(args: argparse.Namespace) -> BenchRecord:
    from drakeuni.runtime.mjcf_model_parser import (
        materialize_drake_compatible_mjcf,
        parse_mjcf_model_contract,
        read_keyframe_qpos,
    )
    from pydrake.all import AddMultibodyPlantSceneGraph, DiagramBuilder, Parser, Simulator
    from pydrake.multibody.plant import ContactModel, DiscreteContactApproximation

    class PydrakeEnv:
        def __init__(self, model_file: str, sim_dt: float, home_qpos: np.ndarray | None) -> None:
            builder = DiagramBuilder()
            self.plant, _scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=sim_dt)
            self.plant.set_discrete_contact_approximation(DiscreteContactApproximation.kSap)
            self.plant.set_contact_model(ContactModel.kHydroelasticWithFallback)
            Parser(self.plant).AddModels(model_file)
            self.plant.Finalize()
            self.diagram = builder.Build()
            self.context = self.diagram.CreateDefaultContext()
            self.plant_context = self.plant.GetMyMutableContextFromRoot(self.context)
            self.simulator = Simulator(self.diagram, self.context)
            if home_qpos is None:
                self.home_qpos = np.asarray(
                    self.plant.GetPositions(self.plant_context),
                    dtype=np.float64,
                )
            else:
                self.home_qpos = home_qpos.copy()
            self.home_qvel = np.zeros(self.plant.num_velocities(), dtype=np.float64)
            self.ctrl = np.zeros(self.plant.num_actuators(), dtype=np.float64)
            if self.plant.num_actuators() > 0:
                self.plant.get_actuation_input_port().FixValue(self.plant_context, self.ctrl)
            self.reset()

        def reset(self) -> np.ndarray:
            self.context.SetTime(0.0)
            self.plant.SetPositions(self.plant_context, self.home_qpos)
            self.plant.SetVelocities(self.plant_context, self.home_qvel)
            self.simulator.Initialize()
            return self.state()

        def step(self, nstep: int, sim_dt: float) -> np.ndarray:
            self.simulator.AdvanceTo(self.context.get_time() + nstep * sim_dt)
            return self.state()

        def state(self) -> np.ndarray:
            return np.concatenate(
                [
                    np.asarray([self.context.get_time()], dtype=np.float64),
                    np.asarray(self.plant.GetPositions(self.plant_context), dtype=np.float64),
                    np.asarray(self.plant.GetVelocities(self.plant_context), dtype=np.float64),
                ]
            )

    drake_model = materialize_drake_compatible_mjcf(args._worker_model_file)
    try:
        contract = parse_mjcf_model_contract(drake_model.model_file)
        home_qpos = read_keyframe_qpos(args._worker_model_file, "home")
        t0 = time.perf_counter()
        first_env = PydrakeEnv(drake_model.model_file, args.sim_dt, home_qpos)
        home_qpos = first_env.home_qpos
        if home_qpos.shape != (first_env.plant.num_positions(),):
            raise ValueError(
                "home qpos shape mismatch: "
                f"expected {(first_env.plant.num_positions(),)}, got {home_qpos.shape}"
            )
        envs = [first_env]
        envs.extend(
            PydrakeEnv(drake_model.model_file, args.sim_dt, home_qpos)
            for _ in range(args.num_envs - 1)
        )
        setup_ms = (time.perf_counter() - t0) * 1000.0

        def reset_all() -> np.ndarray:
            return np.asarray([env.reset() for env in envs], dtype=np.float64)

        def step_all() -> np.ndarray:
            return np.asarray([env.step(args.nstep, args.sim_dt) for env in envs], dtype=np.float64)

        reset_mean_ms, reset_median_ms = _time_call(
            reset_all,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        step_mean_ms, step_median_ms = _time_call(
            step_all,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        nq = int(envs[0].plant.num_positions())
        nv = int(envs[0].plant.num_velocities())
        nu = int(envs[0].plant.num_actuators())
    finally:
        drake_model.close()

    step_seconds = step_mean_ms / 1000.0
    reset_seconds = reset_mean_ms / 1000.0
    return BenchRecord(
        model=args._worker_model_name,
        backend="pydrake-loop",
        num_envs=args.num_envs,
        nthread="serial",
        nstep=args.nstep,
        warmup=args.warmup,
        repeat=args.repeat,
        nq=nq,
        nv=nv,
        nu=nu,
        nsensordata=int(contract.nsensordata),
        setup_ms=setup_ms,
        step_mean_ms=step_mean_ms,
        step_median_ms=step_median_ms,
        reset_mean_ms=reset_mean_ms,
        reset_median_ms=reset_median_ms,
        steps_per_second=args.num_envs * args.nstep / step_seconds if step_seconds > 0 else 0.0,
        reset_envs_per_second=args.num_envs / reset_seconds if reset_seconds > 0 else 0.0,
    )


def _print_table(records: Sequence[BenchRecord]) -> None:
    columns = [
        "model",
        "backend",
        "envs",
        "nthread",
        "setup_ms",
        "step_ms",
        "steps/s",
        "reset_ms",
        "reset_envs/s",
    ]
    print(" | ".join(columns))
    print(" | ".join("---" for _ in columns))
    for record in records:
        row = [
            record.model,
            record.backend,
            str(record.num_envs),
            str(record.nthread),
            f"{record.setup_ms:.2f}",
            f"{record.step_median_ms:.3f}",
            f"{record.steps_per_second:.0f}",
            f"{record.reset_median_ms:.3f}",
            f"{record.reset_envs_per_second:.0f}",
        ]
        print(" | ".join(row))


def _worker_main(args: argparse.Namespace) -> None:
    args.num_envs = int(args.num_envs)
    if args._worker_backend == "drakeuni":
        record = _run_drakeuni_worker(args)
    elif args._worker_backend == "pydrake":
        record = _run_pydrake_worker(args)
    else:
        raise ValueError(f"Unknown worker backend {args._worker_backend!r}")
    print(json.dumps(asdict(record)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="go1,go2", help="Comma-separated model keys.")
    parser.add_argument(
        "--unilab-root",
        type=Path,
        default=DEFAULT_UNILAB_ROOT,
        help="UniLab checkout used for default Go1/Go2 MJCF model paths.",
    )
    parser.add_argument(
        "--model-path",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Add or override a model path.",
    )
    parser.add_argument(
        "--backends",
        default="pydrake,drakeuni",
        help="Comma-separated backends: pydrake,drakeuni.",
    )
    parser.add_argument("--num-envs", default="1,4,16", help="Comma-separated env counts.")
    parser.add_argument("--nthreads", default=str(min(cpu_count(), 16)))
    parser.add_argument("--nthread", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--sim-dt", type=float, default=0.005)
    parser.add_argument("--nstep", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--drakeuni-src", type=Path, default=DEFAULT_DRAKEUNI_SRC)

    # Private subprocess mode.
    parser.add_argument("--_worker-backend", choices=("pydrake", "drakeuni"))
    parser.add_argument("--_worker-model-name", default="")
    parser.add_argument("--_worker-model-file", type=Path)
    args = parser.parse_args()

    if args._worker_backend:
        _worker_main(args)
        return

    unilab_root = args.unilab_root.expanduser().resolve()
    model_paths = {
        name: (unilab_root / relative_path).resolve() for name, relative_path in MODEL_PATHS.items()
    }
    for item in args.model_path:
        if "=" not in item:
            raise ValueError(f"--model-path must be NAME=PATH, got {item!r}")
        name, value = item.split("=", 1)
        model_paths[name.strip()] = Path(value).expanduser()

    records: list[BenchRecord] = []
    for model_name in _parse_csv(args.models):
        model_file = model_paths.get(model_name)
        if model_file is None:
            raise ValueError(f"Unknown model {model_name!r}; available: {sorted(model_paths)}")
        if not model_file.exists():
            raise FileNotFoundError(f"Model {model_name!r} does not exist: {model_file}")
        for num_envs in _parse_int_csv(args.num_envs):
            for backend in _parse_csv(args.backends):
                requested_threads = _parse_int_csv(args.nthreads)
                thread_values = (
                    _effective_drakeuni_threads(num_envs, requested_threads)
                    if backend == "drakeuni"
                    else [0]
                )
                for nthread in thread_values:
                    print(
                        f"Running model={model_name} backend={backend} "
                        f"envs={num_envs} nthread={nthread if backend == 'drakeuni' else 'serial'}",
                        flush=True,
                    )
                    records.append(
                        _run_worker(
                            backend=backend,
                            model_name=model_name,
                            model_file=model_file,
                            num_envs=num_envs,
                            nthread=nthread,
                            sim_dt=args.sim_dt,
                            nstep=args.nstep,
                            warmup=args.warmup,
                            repeat=args.repeat,
                            drakeuni_src=args.drakeuni_src,
                        )
                    )

    _print_table(records)
    payload = {
        "records": [asdict(record) for record in records],
        "args": {
            "models": _parse_csv(args.models),
            "backends": _parse_csv(args.backends),
            "num_envs": _parse_int_csv(args.num_envs),
            "nthreads": _parse_int_csv(args.nthreads),
            "sim_dt": args.sim_dt,
            "nstep": args.nstep,
            "warmup": args.warmup,
            "repeat": args.repeat,
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
