# DrakeUni Bench

Standalone benchmark for comparing a raw `pydrake` per-environment simulator loop
against DrakeUni batched stepping.

The benchmark follows the same idea as
[`unilabsim/mujoco_uni_bench`](https://github.com/unilabsim/mujoco_uni_bench):
hold the model, timestep, action, and step count fixed, then compare the Python
loop baseline with the batched runtime.

## What Is Compared

Two backends are measured in separate subprocesses:

- `pydrake-loop`: one `pydrake` `Simulator` per environment, stepped serially in
  Python.
- `drakeuni`: DrakeUni compact-state batch runtime with configurable worker
  threads.

Each benchmark row measures:

- batched step latency
- aggregate physics steps per second, the main rollout metric
- speedup of DrakeUni over the serial `pydrake-loop`

The benchmark script also records setup and reset diagnostics in JSON, but the
README report focuses on steady rollout speed. Setup includes one-time model
preprocessing and is not a fair runtime-speed headline.

The default model paths point to a sibling UniLab checkout:

```text
../unilab/UniLab/src/unilab/assets/robots/go1/scene_flat.xml
../unilab/UniLab/src/unilab/assets/robots/go2/scene_flat.xml
```

These are the original UniLab MJCF scenes. DrakeUni materializes temporary
Drake-compatible copies during setup; the benchmark does not require
pre-generated `*_drake.xml` files or committed converted mesh assets.

Custom model paths can be supplied with `--model-path NAME=/path/to/model.xml`.

## Reproduce

Use a Python environment where `pydrake`, `numpy`, and local `drakeuni` are
available. In the current workspace, the UniLab virtual environment provides
`pydrake`:

```bash
cd /Users/huanghaochen/solver/drakeuni_bench

PYTHONPATH=/Users/huanghaochen/solver/drakeuni/src \
  /Users/huanghaochen/solver/unilab/UniLab/.venv/bin/python \
  benchmark_drakeuni_vs_pydrake.py \
  --models go1,go2 \
  --num-envs 1,4,16 \
  --nthreads 1,4,8 \
  --nstep 10 \
  --warmup 2 \
  --repeat 5 \
  --out-json outputs/go1_go2_small_clean.json
```

## Current Result

Run date: 2026-06-21  
Machine: local macOS arm64 workstation  
Timestep: `0.005` seconds  
Rollout chunk: `10` simulator steps per benchmark step  
Control input: zero control  
Models: UniLab Go1 and Go2 flat MJCF scenes  

### Throughput Summary

Higher is better. The speedup column compares the best DrakeUni result for that
model/env count against the serial `pydrake-loop` baseline.

| model | envs | pydrake-loop steps/s | best DrakeUni steps/s | best DrakeUni workers | speedup |
|---|---:|---:|---:|---:|---:|
| Go1 | 1 | 4,190 | 16,798 | 1 | **4.0x** |
| Go1 | 4 | 4,180 | 39,575 | 4 | **9.5x** |
| Go1 | 16 | 4,103 | 51,352 | 8 | **12.5x** |
| Go2 | 1 | 6,361 | 15,469 | 1 | **2.4x** |
| Go2 | 4 | 6,110 | 37,893 | 4 | **6.2x** |
| Go2 | 16 | 6,178 | 45,884 | 8 | **7.4x** |

### DrakeUni Scaling Detail

This table shows how DrakeUni improves as worker count increases. A dash means
the worker count is not distinct for that env count because DrakeUni caps
workers at `num_envs`.

| model | envs | pydrake-loop steps/s | DrakeUni 1 worker | DrakeUni 4 workers | DrakeUni 8 workers |
|---|---:|---:|---:|---:|---:|
| Go1 | 1 | 4,190 | 16,798 | - | - |
| Go1 | 4 | 4,180 | 16,915 | 39,575 | - |
| Go1 | 16 | 4,103 | 17,344 | 40,096 | 51,352 |
| Go2 | 1 | 6,361 | 15,469 | - | - |
| Go2 | 4 | 6,110 | 15,862 | 37,893 | - |
| Go2 | 16 | 6,178 | 16,154 | 38,216 | 45,884 |

## Reading The Result

The key result is steady rollout throughput:

- Go1 at 16 envs: DrakeUni reaches about `51k` physics steps/s with 8 workers,
  versus about `4k` physics steps/s for the serial pydrake loop, a **12.5x**
  speedup.
- Go2 at 16 envs: DrakeUni reaches about `46k` physics steps/s with 8 workers,
  versus about `6k` physics steps/s for the serial pydrake loop, a **7.4x**
  speedup.

This means DrakeUni's current advantage is exactly where it was intended to be:
batched rollout stepping.

The setup gap is real, but it is mostly not a physics-step cost. It comes from:

- materializing a Drake-compatible MJCF copy;
- parsing the MJCF contract for actuators, sensors, joints, contacts, and body
  indices;
- constructing the DrakeUni batch pool; and
- constructing reusable Drake worker contexts.

Those operations happen once before rollout. They should be optimized, but they
should not be read as the steady-state simulation speed.

The reset gap is also real and has a different cause. The raw pydrake baseline
resets already-constructed simulator contexts in place. DrakeUni reset restores
compact state through the batch runtime and refreshes raw sensor data. That
matches UniLab's training data path more closely, but it currently does more
copy/sensor work than the minimal pydrake reset loop. Reset-heavy tasks will
therefore expose this overhead more strongly than long rollout segments.

The next optimization targets for DrakeUni are setup decomposition and reset
state/sensor refresh. The current benchmark already shows that the core batched
physics stepping path is working.

## Notes

- The benchmark intentionally uses subprocesses. This avoids mixing `pydrake`
  and DrakeUni's compiled Drake extension in one Python process.
- Drake warnings from MJCF parsing are expected and are emitted by Drake's MJCF
  parser.
- The benchmark is simulator-level. It does not measure UniLab observation,
  reward, policy, replay buffer, or training overhead.
