# Public-CoreML M5 matmul spike

This workspace tests one narrow question: can a public-CoreML ML Program with int8 weights and per-output-channel scales produce correct matmul results and show Neural Engine placement and performance headroom on this M5 Mac?

This is a microbenchmark, not an llama.cpp backend or proof of the Tether bounty acceptance criteria. No private ANE APIs are used.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## Verify

Run the correctness tests and smoke benchmark:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python spike/ane_matmul.py --suite smoke --output results/m5-smoke.json
```

Run the larger K=N=4096 benchmark only after the smoke suite passes:

```bash
.venv/bin/python spike/ane_matmul.py --suite full --output results/m5-full.json
```

Each shape uses deterministic int8 weights, one fp16 scale per output channel, and fp16 activations. The script compares `CPU_AND_NE`, `CPU_AND_GPU`, and `CPU_ONLY`, inspects the public `MLComputePlan`, checks against a NumPy reference, separates model conversion/loading from warm prediction, and writes all timings and errors to JSON.

Generated model packages and compiled models are confined to a temporary directory under `results/work` and removed after each shape. Timing gates are for deciding whether a ggml experiment is worth attempting; they are not claimed as end-to-end speedups.

## Verified M5 results

Environment: Apple M5, macOS 27.0.1, Python 3.12.12, coremltools 9.0, NumPy 2.5.4. Two independent full-suite runs used different data seeds, three warmups, and seven or eleven timed predictions per configuration.

At K=N=4096, `CPU_AND_NE` showed the following ratios against the lower median of `CPU_AND_GPU` and `CPU_ONLY`:

| M | First run | Reproduction |
|---|---:|---:|
| 1 | 0.454x | 0.459x |
| 32 | 0.528x | 0.520x |
| 128 | 1.247x | 1.261x |
| 512 | 1.969x | 1.978x |

At M=512, `CPU_AND_NE` took 2.9316 ms versus 5.7723 ms in the first run and 2.9092 ms versus 5.7557 ms in the reproduction. Every shape and compute unit passed the 2% relative-L2 correctness check. Public `MLComputePlan` inspection reported Neural Engine placement for the matmul under `CPU_AND_NE`.

The results support further prefill experiments. They do not establish llama.cpp/Metal end-to-end speedup, Q8_0 block-scale compatibility, IOSurface behavior, energy efficiency, thermal behavior, or bounty acceptance. See `VERIFICATION.md` and the raw JSON under `results/` for exact evidence.

## Reproduce the second full run

```bash
.venv/bin/python spike/ane_matmul.py --suite full --seed 107 --repetitions 11 --output results/m5-full-reproduction.json
```
