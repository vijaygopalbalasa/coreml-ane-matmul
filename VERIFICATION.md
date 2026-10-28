# M5 public-CoreML verification

## Environment and scope

- Apple M5, arm64, macOS 27.0.1
- Python 3.12.12, coremltools 9.0, NumPy 2.5.4
- Signed-int8 weights with one fp16 scale per output channel; fp16 activations
- Public CoreML/coremltools APIs only
- Standalone matmul microbenchmark; no llama.cpp backend, QVAC integration, energy measurement, thermal test, or iOS test

## Commands executed

```bash
uv pip check --python .venv/bin/python
.venv/bin/python -m compileall -q spike tests
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python spike/ane_matmul.py --suite smoke --output results/m5-smoke.json
.venv/bin/python spike/ane_matmul.py --suite full --output results/m5-full.json
.venv/bin/python spike/ane_matmul.py --suite full --seed 107 --repetitions 11 --output results/m5-full-reproduction.json
```

Dependency check passed, compilation completed, and all four unit tests passed. Each timed full-suite prediction followed three warmups. The first full run used seven timed predictions per configuration; the reproduction used eleven and a different data seed.

## Full-suite evidence

K=4096 and N=4096 for every row. `Best alternative` is the lower median of CoreML `CPU_AND_GPU` and `CPU_ONLY`. The ratio is best alternative divided by `CPU_AND_NE`; values above 1.00 favor the Neural Engine configuration.

| M | Run | CPU_AND_NE median | Best alternative | Ratio | NE relative L2 |
|---|---|---|---:|---:|---:|
| 1 | 1 | 0.9695 ms | 0.4398 ms | 0.454x | 0.000289 |
| 1 | 2 | 0.9579 ms | 0.4395 ms | 0.459x | 0.000294 |
| 32 | 1 | 1.0409 ms | 0.5497 ms | 0.528x | 0.000295 |
| 32 | 2 | 1.0359 ms | 0.5383 ms | 0.520x | 0.000292 |
| 128 | 1 | 1.2818 ms | 1.5979 ms | 1.247x | 0.000294 |
| 128 | 2 | 1.2780 ms | 1.6113 ms | 1.261x | 0.000293 |
| 512 | 1 | 2.9316 ms | 5.7723 ms | 1.969x | 0.000294 |
| 512 | 2 | 2.9092 ms | 5.7557 ms | 1.978x | 0.000293 |

For both runs:

- Every shape and every compute-unit configuration passed the 2% relative-L2 correctness check against the independent NumPy reference.
- Public `MLComputePlan` inspection reported `neural_engine` as the preferred device for the matmul at every M value under `CPU_AND_NE`.
- M=128 and M=512 passed the exploratory 1.20x headroom gate. M=1 and M=32 were substantially slower under `CPU_AND_NE`.

Raw inputs, per-prediction timings, operation placements, errors, conversion/load timings, and artifact sizes are in `results/m5-full.json` and `results/m5-full-reproduction.json`.

## Decision and limits

The standalone technical spike passed: there is enough reproducible M5 prefill headroom to justify discussing a minimal ggml experiment after Tether confirms the bounty scope.

This does not establish actual llama.cpp speedup, parity with ggml Metal, Q8_0 block-scale compatibility, IOSurface behavior, cache correctness, energy efficiency, thermal behavior, or bounty acceptance. The scope gate remains pending Tether's written confirmation. The next engineering stage must preserve Metal/CPU fallback and must not infer ordinary decode acceleration from these prefill-oriented results.
