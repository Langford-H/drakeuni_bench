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

- setup time, reported as a diagnostic only
- reset latency after the runtime has already been built
- batched step latency
- aggregate physics steps per second, the main rollout metric

Important fairness note: `setup ms` is **not** the headline runtime comparison.
For DrakeUni, setup includes MJCF materialization, MJCF contract parsing, batch
pool creation, and Drake worker-context construction. Those are one-time
preprocessing/runtime-construction costs. For `pydrake-loop`, setup is only
direct simulator construction. The step-throughput columns are the fairer
measure of rollout performance after both backends are ready.

The default model paths point to a sibling UniLab checkout:

```text
../unilab/UniLab/src/unilab/assets/robots/go1/scene_flat.xml
../unilab/UniLab/src/unilab/assets/robots/go2/scene_flat.xml
```

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

| model | backend | envs | workers | setup ms | step median ms | physics steps/s | reset median ms | reset envs/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Go1 | pydrake-loop | 1 | serial | 3.15 | 2.259 | 4,190 | 0.009 | 107,527 |
| Go1 | drakeuni | 1 | 1 | 492.71 | 0.578 | 16,798 | 0.087 | 11,120 |
| Go1 | pydrake-loop | 4 | serial | 9.97 | 9.089 | 4,180 | 0.038 | 104,529 |
| Go1 | drakeuni | 4 | 1 | 490.96 | 2.322 | 16,915 | 0.269 | 14,781 |
| Go1 | drakeuni | 4 | 4 | 489.45 | 1.018 | 39,575 | 0.165 | 23,705 |
| Go1 | pydrake-loop | 16 | serial | 35.26 | 38.758 | 4,103 | 0.151 | 105,814 |
| Go1 | drakeuni | 16 | 1 | 503.27 | 9.106 | 17,344 | 1.024 | 15,031 |
| Go1 | drakeuni | 16 | 4 | 493.50 | 3.873 | 40,096 | 0.496 | 32,228 |
| Go1 | drakeuni | 16 | 8 | 493.43 | 3.064 | 51,352 | 0.476 | 33,727 |
| Go2 | pydrake-loop | 1 | serial | 2.60 | 1.475 | 6,361 | 0.010 | 93,240 |
| Go2 | drakeuni | 1 | 1 | 66.19 | 0.638 | 15,469 | 0.073 | 13,462 |
| Go2 | pydrake-loop | 4 | serial | 7.52 | 6.197 | 6,110 | 0.039 | 94,470 |
| Go2 | drakeuni | 4 | 1 | 67.11 | 2.451 | 15,862 | 0.256 | 15,608 |
| Go2 | drakeuni | 4 | 4 | 79.50 | 1.008 | 37,893 | 0.140 | 27,347 |
| Go2 | pydrake-loop | 16 | serial | 25.61 | 24.916 | 6,178 | 0.150 | 106,289 |
| Go2 | drakeuni | 16 | 1 | 70.06 | 9.496 | 16,154 | 0.963 | 16,616 |
| Go2 | drakeuni | 16 | 4 | 67.59 | 4.337 | 38,216 | 0.555 | 28,353 |
| Go2 | drakeuni | 16 | 8 | 67.21 | 3.391 | 45,884 | 0.459 | 34,873 |

## Reading The Result

The key result is step throughput:

- Go1 at 16 envs: DrakeUni reaches about `51k` physics steps/s with 8 workers,
  versus about `4k` physics steps/s for the serial pydrake loop.
- Go2 at 16 envs: DrakeUni reaches about `46k` physics steps/s with 8 workers,
  versus about `6k` physics steps/s for the serial pydrake loop.

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

In short:

| path | current result |
|---|---|
| setup/preprocess | pydrake-loop is lighter in this benchmark |
| reset | pydrake-loop is lighter in this benchmark |
| steady rollout stepping | DrakeUni is much faster |

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
