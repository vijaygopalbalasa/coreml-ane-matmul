import argparse
import json
import math
import platform
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil.mil import Builder as mb, types
from coremltools.models.compute_plan import MLComputePlan

REL_TOLERANCE = 0.02
HEADROOM_GATE = 1.20
COMPUTE_UNITS = (
    ("cpu_and_ne", ct.ComputeUnit.CPU_AND_NE),
    ("cpu_and_gpu", ct.ComputeUnit.CPU_AND_GPU),
    ("cpu_only", ct.ComputeUnit.CPU_ONLY),
)
SUITES = {
    "smoke": ((1, 256, 384), (32, 256, 384), (128, 256, 384), (512, 256, 384)),
    "full": ((1, 4096, 4096), (32, 4096, 4096), (128, 4096, 4096), (512, 4096, 4096)),
}


@dataclass(frozen=True)
class Shape:
    m: int
    k: int
    n: int

    def as_dict(self):
        return {"m": self.m, "k": self.k, "n": self.n}


def quantize_per_output_channel(weights):
    scales = np.maximum(np.max(np.abs(weights), axis=1) / 127.0, 1e-8)
    qweight = np.clip(np.rint(weights / scales[:, None]), -127, 127).astype(np.int8)
    scales_fp16 = scales.astype(np.float16)
    dequantized = qweight.astype(np.float32) * scales_fp16.astype(np.float32)[:, None]
    return qweight, scales_fp16, dequantized


def reference_matmul(activations, qweight, scales):
    if activations.shape[1] != qweight.shape[1]:
        raise ValueError("activation K must match weight K")
    weights = qweight.astype(np.float32) * scales.astype(np.float32)[:, None]
    return np.matmul(activations.astype(np.float32), weights.T)


def relative_l2(actual, expected):
    actual_f32 = np.asarray(actual, dtype=np.float32)
    expected_f32 = np.asarray(expected, dtype=np.float32)
    denominator = float(np.linalg.norm(expected_f32))
    if denominator == 0.0:
        return float(np.linalg.norm(actual_f32 - expected_f32))
    return float(np.linalg.norm(actual_f32 - expected_f32) / denominator)


def convert_model(shape, qweight, scales, package_path):
    @mb.program(
        input_specs=[mb.TensorSpec(shape=(shape.m, shape.k), dtype=types.fp16)],
        opset_version=ct.target.macOS26,
    )
    def program(x):
        weight = mb.const(val=qweight, name="qweight")
        scale = mb.const(val=scales, name="scales")
        dequantized = mb.dequantize(input=weight, scale=scale, axis=0, name="dequantize")
        return mb.matmul(x=x, y=dequantized, transpose_y=True, name="output")

    started = time.perf_counter()
    model = ct.convert(
        program,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS26,
    )
    conversion_seconds = time.perf_counter() - started
    model.save(str(package_path))
    return conversion_seconds


def device_name(device):
    mapping = {
        "MLCPUComputeDevice": "cpu",
        "MLGPUComputeDevice": "gpu",
        "MLNeuralEngineComputeDevice": "neural_engine",
    }
    return mapping.get(type(device).__name__, type(device).__name__)


def inspect_compute_plan(compiled_path, compute_unit):
    plan = MLComputePlan.load_from_path(compiled_path, compute_units=compute_unit)
    operations = plan.model_structure.program.functions["main"].block.operations
    result = []
    for operation in operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(operation=operation)
        result.append(
            {
                "operation": operation.operator_name,
                "preferred_device": device_name(usage.preferred_compute_device) if usage else "not_reported",
                "supported_devices": [device_name(device) for device in usage.supported_compute_devices] if usage else [],
            }
        )
    return result


def directory_size(path):
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def time_predict(model, activations, warmup, repetitions):
    output = None
    for _ in range(warmup):
        output = model.predict({"x": activations})
    timings_ms = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        output = model.predict({"x": activations})
        timings_ms.append((time.perf_counter_ns() - started) / 1_000_000)
    output_array = np.asarray(next(iter(output.values())))
    return output_array, timings_ms


def benchmark_shape(shape, seed, warmup, repetitions, work_root):
    rng = np.random.default_rng(seed)
    original_weights = rng.standard_normal((shape.n, shape.k), dtype=np.float32) * 0.02
    qweight, scales, _ = quantize_per_output_channel(original_weights)
    activations = rng.standard_normal((shape.m, shape.k), dtype=np.float32).astype(np.float16)
    expected = reference_matmul(activations, qweight, scales)

    with tempfile.TemporaryDirectory(prefix=f"ane-{shape.m}-{shape.k}-{shape.n}-", dir=work_root) as temp_dir:
        package_path = Path(temp_dir) / "matmul.mlpackage"
        conversion_seconds = convert_model(shape, qweight, scales, package_path)
        result = {
            "shape": shape.as_dict(),
            "conversion_seconds": conversion_seconds,
            "compute_units": {},
        }
        for name, compute_unit in COMPUTE_UNITS:
            load_started = time.perf_counter()
            model = ct.models.MLModel(str(package_path), compute_units=compute_unit)
            load_seconds = time.perf_counter() - load_started
            operations = inspect_compute_plan(model.get_compiled_model_path(), compute_unit)
            actual, timings_ms = time_predict(model, activations, warmup, repetitions)
            median_ms = float(np.median(timings_ms))
            relative_error = relative_l2(actual, expected)
            result["compute_units"][name] = {
                "load_seconds": load_seconds,
                "operations": operations,
                "output_dtype": str(actual.dtype),
                "max_abs_error": float(np.max(np.abs(actual.astype(np.float32) - expected))),
                "relative_l2_error": relative_error,
                "correct_within_tolerance": math.isfinite(relative_error) and relative_error <= REL_TOLERANCE,
                "timings_ms": timings_ms,
                "median_ms": median_ms,
                "gflops": float(2 * shape.m * shape.k * shape.n / (median_ms * 1e9)),
                "artifact_bytes": directory_size(package_path) + directory_size(model.get_compiled_model_path()),
            }
            print(
                f"{shape.m}x{shape.k}x{shape.n} {name}: "
                f"{median_ms:.3f} ms, rel_l2={relative_error:.6f}, "
                f"devices={[op['preferred_device'] for op in operations]}",
                flush=True,
            )
        return result


def add_summary(report):
    runs = [
        (shape_result, name, run)
        for shape_result in report["results"]
        for name, run in shape_result["compute_units"].items()
    ]
    correctness = all(run["correct_within_tolerance"] for _, _, run in runs)
    ne_shapes = set()
    for shape_result, name, run in runs:
        if name != "cpu_and_ne":
            continue
        if any(
            "matmul" in operation["operation"] and operation["preferred_device"] == "neural_engine"
            for operation in run["operations"]
        ):
            ne_shapes.add(tuple(shape_result["shape"][key] for key in ("m", "k", "n")))

    candidates = []
    for shape_result in report["results"]:
        m = shape_result["shape"]["m"]
        if m < 32:
            continue
        shape_tuple = tuple(shape_result["shape"][key] for key in ("m", "k", "n"))
        if shape_tuple not in ne_shapes:
            continue
        units = shape_result["compute_units"]
        ne_ms = units["cpu_and_ne"]["median_ms"]
        alternative_ms = min(units["cpu_and_gpu"]["median_ms"], units["cpu_only"]["median_ms"])
        candidates.append({"shape": shape_result["shape"], "speedup": alternative_ms / ne_ms})

    best = max(candidates, key=lambda candidate: candidate["speedup"], default=None)
    performance = best is not None and best["speedup"] >= HEADROOM_GATE
    report["summary"] = {
        "relative_l2_tolerance": REL_TOLERANCE,
        "headroom_gate": HEADROOM_GATE,
        "correctness_gate": correctness,
        "plan_reported_ne_prefill_gate": bool(ne_shapes),
        "performance_gate": performance,
        "ne_prefill_shapes": [list(shape) for shape in sorted(ne_shapes)],
        "best_prefill_candidate": best,
        "spike_gate": correctness and bool(ne_shapes) and performance,
        "scope_gate": "pending_tether_written_confirmation",
        "interpretation": "Microbenchmark only. Not an end-to-end llama.cpp, energy, thermal, or bounty-acceptance result.",
    }


def environment_info():
    try:
        chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        chip = platform.processor() or platform.machine()
    return {
        "chip": chip,
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "coremltools": ct.__version__,
        "numpy": np.__version__,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Public-CoreML int8-weight matmul microbenchmark")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("results/work"))
    parser.add_argument("--suite", choices=sorted(SUITES), default="smoke")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=7)
    args = parser.parse_args()
    if args.warmup < 0 or args.repetitions < 1:
        parser.error("warmup must be >= 0 and repetitions must be >= 1")
    return args


def main():
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "environment": environment_info(),
        "configuration": {
            "suite": args.suite,
            "seed": args.seed,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "quantization": "signed int8 weights, one fp16 scale per output channel, fp16 activations",
        },
        "results": [],
    }
    for index, shape_values in enumerate(SUITES[args.suite]):
        shape = Shape(*shape_values)
        report["results"].append(benchmark_shape(shape, args.seed + index, args.warmup, args.repetitions, args.work_dir))
        add_summary(report)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
