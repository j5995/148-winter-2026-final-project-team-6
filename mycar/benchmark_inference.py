"""
benchmark_inference.py
======================
Benchmarks donkeycar model inference on RPi 5 with and without the AI HAT
(Hailo-8L / Hailo-8 via HailoRT).

Metrics captured
----------------
- Inference latency (ms) per frame  – mean, median, p95, p99
- Frames per second (FPS)
- CPU utilisation (%) during inference
- CPU temperature (°C)
- Memory usage (MB RSS)
- Whether the Hailo NPU is active

Usage
-----
# CPU-only (TFLite)
python benchmark_inference.py --mode cpu --model models/pilot.tflite --runs 500

# AI HAT (Hailo .hef model)
python benchmark_inference.py --mode hailo --model models/pilot.hef --runs 500

# Auto-detect + compare both
python benchmark_inference.py --mode compare --tflite models/pilot.tflite \\
       --hef models/pilot.hef --runs 500 --output results/inference_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports – guarded so the benchmark still runs without hardware
# ---------------------------------------------------------------------------
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logging.warning("psutil not found – CPU/memory metrics will be unavailable. "
                    "Install with: pip install psutil")

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    from hailo_platform import (HEF, ConfigureParams, FormatType,
                                 HailoStreamInterface, InferVStreams,
                                 InputVStreamParams, OutputVStreamParams,
                                 VDevice)
    HAS_HAILO = True
except ImportError:
    HAS_HAILO = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    mode: str                          # 'cpu' | 'hailo'
    model_path: str
    num_runs: int
    input_shape: Tuple[int, ...]
    latencies_ms: List[float] = field(default_factory=list)

    # Derived stats (populated by .summarise())
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    fps: float = 0.0
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    temperature_c: float = 0.0
    hailo_detected: bool = False
    platform: str = ""
    timestamp: str = ""

    def summarise(self) -> None:
        arr = np.array(self.latencies_ms)
        self.mean_ms   = float(np.mean(arr))
        self.median_ms = float(np.median(arr))
        self.p95_ms    = float(np.percentile(arr, 95))
        self.p99_ms    = float(np.percentile(arr, 99))
        self.min_ms    = float(np.min(arr))
        self.max_ms    = float(np.max(arr))
        self.fps       = 1000.0 / self.mean_ms if self.mean_ms > 0 else 0.0
        self.platform  = platform.platform()
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("latencies_ms")          # too verbose for JSON output
        return d


# ---------------------------------------------------------------------------
# System helpers
# ---------------------------------------------------------------------------

def get_cpu_temperature() -> float:
    """Return CPU temperature in °C for Raspberry Pi 5 (or 0.0 if unavailable)."""
    paths = [
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/class/hwmon/hwmon0/temp1_input",
    ]
    for p in paths:
        try:
            raw = Path(p).read_text().strip()
            return float(raw) / 1000.0
        except (FileNotFoundError, ValueError):
            pass
    # Fallback via vcgencmd (Raspberry Pi OS)
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"],
                                      stderr=subprocess.DEVNULL,
                                      timeout=2).decode()
        return float(out.strip().replace("temp=", "").replace("'C", ""))
    except Exception:
        return 0.0


def check_hailo_device() -> bool:
    """Return True if a Hailo NPU is visible on this system."""
    try:
        result = subprocess.run(
            ["hailortcli", "fw-control", "identify"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except FileNotFoundError:
        pass
    # Fallback: check PCIe
    try:
        result = subprocess.run(
            ["lspci"], capture_output=True, text=True, timeout=3,
        )
        return "hailo" in result.stdout.lower()
    except Exception:
        return False


def get_process_memory_mb() -> float:
    if not HAS_PSUTIL:
        return 0.0
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 * 1024)


def get_cpu_percent(interval: float = 0.5) -> float:
    if not HAS_PSUTIL:
        return 0.0
    return psutil.cpu_percent(interval=interval)


# ---------------------------------------------------------------------------
# Dummy image generator (replaces real camera during benchmarking)
# ---------------------------------------------------------------------------

def make_dummy_frame(h: int = 120, w: int = 160, d: int = 3) -> np.ndarray:
    """Return a random uint8 image array matching donkeycar's default resolution."""
    return np.random.randint(0, 256, (h, w, d), dtype=np.uint8)


# ---------------------------------------------------------------------------
# TFLite CPU inference
# ---------------------------------------------------------------------------

def run_tflite_benchmark(
    model_path: str,
    num_runs: int = 300,
    warmup: int = 20,
) -> InferenceResult:
    if not HAS_TF:
        raise RuntimeError("TensorFlow is not installed. "
                           "Install with: pip install tensorflow")

    log.info("Loading TFLite model: %s", model_path)
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Infer input shape from model
    shape = tuple(input_details[0]["shape"])
    _, h, w, d = shape  # (1, H, W, C)

    result = InferenceResult(
        mode="cpu",
        model_path=model_path,
        num_runs=num_runs,
        input_shape=(h, w, d),
    )
    result.hailo_detected = False

    log.info("Warming up TFLite interpreter (%d frames)…", warmup)
    for _ in range(warmup):
        frame = make_dummy_frame(h, w, d).astype(np.float32) / 255.0
        interpreter.set_tensor(input_details[0]["index"],
                               frame.reshape(shape))
        interpreter.invoke()

    log.info("Running %d inference passes (CPU)…", num_runs)
    cpu_samples: List[float] = []

    for i in range(num_runs):
        frame = make_dummy_frame(h, w, d).astype(np.float32) / 255.0
        interpreter.set_tensor(input_details[0]["index"],
                               frame.reshape(shape))

        t0 = time.perf_counter()
        interpreter.invoke()
        t1 = time.perf_counter()

        result.latencies_ms.append((t1 - t0) * 1000.0)

        if i % 50 == 0:
            cpu_samples.append(get_cpu_percent(0.1))
            log.debug("  [%d/%d] %.2f ms", i + 1, num_runs,
                      result.latencies_ms[-1])

    result.cpu_percent   = float(np.mean(cpu_samples)) if cpu_samples else 0.0
    result.memory_mb     = get_process_memory_mb()
    result.temperature_c = get_cpu_temperature()
    result.summarise()
    return result


# ---------------------------------------------------------------------------
# Hailo (AI HAT) inference
# ---------------------------------------------------------------------------

def run_hailo_benchmark(
    model_path: str,
    num_runs: int = 300,
    warmup: int = 20,
) -> InferenceResult:
    if not HAS_HAILO:
        raise RuntimeError(
            "hailo_platform package not found.\n"
            "Install HailoRT from https://hailo.ai/developer-zone/ "
            "and ensure the AI HAT is physically connected via PCIe."
        )

    log.info("Loading HEF model: %s", model_path)
    hef = HEF(model_path)
    target = VDevice()
    configure_params = ConfigureParams.create_from_hef(hef,
                        interface=HailoStreamInterface.PCIe)
    network_groups = target.configure(hef, configure_params)
    network_group = network_groups[0]
    network_group_params = network_group.create_params()

    input_vstreams_params  = InputVStreamParams.make(
        network_group, quantized=False,
        format_type=FormatType.FLOAT32)
    output_vstreams_params = OutputVStreamParams.make(
        network_group, quantized=True,
        format_type=FormatType.FLOAT32)

    input_info  = hef.get_input_vstream_infos()[0]
    shape = input_info.shape               # (H, W, C)
    h, w, d = shape

    result = InferenceResult(
        mode="hailo",
        model_path=model_path,
        num_runs=num_runs,
        input_shape=(h, w, d),
    )
    result.hailo_detected = True

    with InferVStreams(network_group,
                      input_vstreams_params,
                      output_vstreams_params) as infer_pipeline:

        with network_group.activate(network_group_params):

            log.info("Warming up Hailo NPU (%d frames)…", warmup)
            for _ in range(warmup):
                frame = make_dummy_frame(h, w, d).astype(np.float32) / 255.0
                infer_pipeline.infer({input_info.name: frame[np.newaxis, ...]})

            log.info("Running %d inference passes (Hailo NPU)…", num_runs)
            cpu_samples: List[float] = []

            for i in range(num_runs):
                frame = make_dummy_frame(h, w, d).astype(np.float32) / 255.0
                t0 = time.perf_counter()
                infer_pipeline.infer({input_info.name: frame[np.newaxis, ...]})
                t1 = time.perf_counter()

                result.latencies_ms.append((t1 - t0) * 1000.0)

                if i % 50 == 0:
                    cpu_samples.append(get_cpu_percent(0.1))
                    log.debug("  [%d/%d] %.2f ms", i + 1, num_runs,
                              result.latencies_ms[-1])

    result.cpu_percent   = float(np.mean(cpu_samples)) if cpu_samples else 0.0
    result.memory_mb     = get_process_memory_mb()
    result.temperature_c = get_cpu_temperature()
    result.summarise()
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_result(r: InferenceResult) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Mode          : {r.mode.upper()}")
    print(f"  Model         : {r.model_path}")
    print(f"  Runs          : {r.num_runs}")
    print(f"  Input shape   : {r.input_shape}")
    print(sep)
    print(f"  Mean latency  : {r.mean_ms:>8.2f} ms")
    print(f"  Median latency: {r.median_ms:>8.2f} ms")
    print(f"  p95 latency   : {r.p95_ms:>8.2f} ms")
    print(f"  p99 latency   : {r.p99_ms:>8.2f} ms")
    print(f"  Min latency   : {r.min_ms:>8.2f} ms")
    print(f"  Max latency   : {r.max_ms:>8.2f} ms")
    print(f"  FPS (mean)    : {r.fps:>8.1f}")
    print(f"  CPU usage     : {r.cpu_percent:>8.1f} %")
    print(f"  Memory (RSS)  : {r.memory_mb:>8.1f} MB")
    print(f"  CPU temp      : {r.temperature_c:>8.1f} °C")
    print(f"  Hailo detected: {r.hailo_detected}")
    print(sep)


def save_results(results: List[InferenceResult], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data = [r.to_dict() for r in results]

    # JSON
    json_path = output_path if output_path.endswith(".json") else output_path + ".json"
    with open(json_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    log.info("Results saved → %s", json_path)

    # CSV summary
    csv_path = json_path.replace(".json", ".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    log.info("CSV summary  → %s", csv_path)


def print_comparison(cpu_r: InferenceResult, hailo_r: InferenceResult) -> None:
    speedup = cpu_r.mean_ms / hailo_r.mean_ms if hailo_r.mean_ms > 0 else 0
    cpu_relief = cpu_r.cpu_percent - hailo_r.cpu_percent
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY: CPU vs AI HAT (Hailo NPU)")
    print("=" * 60)
    print(f"  FPS  CPU-only : {cpu_r.fps:>8.1f}")
    print(f"  FPS  AI HAT  : {hailo_r.fps:>8.1f}   ({speedup:.2f}× faster)")
    print(f"  Mean latency CPU  : {cpu_r.mean_ms:>6.2f} ms")
    print(f"  Mean latency HAT  : {hailo_r.mean_ms:>6.2f} ms")
    print(f"  CPU load CPU-only : {cpu_r.cpu_percent:>5.1f}%")
    print(f"  CPU load AI HAT  : {hailo_r.cpu_percent:>5.1f}%   ({cpu_relief:+.1f}% relief)")
    print(f"  Temp CPU-only     : {cpu_r.temperature_c:>5.1f} °C")
    print(f"  Temp AI HAT       : {hailo_r.temperature_c:>5.1f} °C")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark donkeycar inference: CPU vs Hailo AI HAT"
    )
    p.add_argument("--mode", choices=["cpu", "hailo", "compare"],
                   default="cpu",
                   help="Which backend(s) to benchmark")
    p.add_argument("--model",   default="",
                   help="Path to .tflite model (cpu mode)")
    p.add_argument("--tflite",  default="",
                   help="Path to .tflite model (compare mode)")
    p.add_argument("--hef",     default="",
                   help="Path to .hef model (hailo / compare mode)")
    p.add_argument("--runs",    type=int, default=300,
                   help="Number of inference runs (default: 300)")
    p.add_argument("--warmup",  type=int, default=20,
                   help="Warmup runs before timing (default: 20)")
    p.add_argument("--output",  default="results/inference_report",
                   help="Output path prefix for JSON + CSV results")
    p.add_argument("--image-size", default="120x160",
                   help="HxW for synthetic frames when no model provided, "
                        "e.g. 120x160")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results: List[InferenceResult] = []

    log.info("Hailo device present: %s", check_hailo_device())

    if args.mode == "cpu":
        if not args.model:
            log.error("--model is required for --mode cpu")
            sys.exit(1)
        r = run_tflite_benchmark(args.model, args.runs, args.warmup)
        print_result(r)
        results.append(r)

    elif args.mode == "hailo":
        if not args.hef and not args.model:
            log.error("--hef (or --model) is required for --mode hailo")
            sys.exit(1)
        hef_path = args.hef or args.model
        r = run_hailo_benchmark(hef_path, args.runs, args.warmup)
        print_result(r)
        results.append(r)

    elif args.mode == "compare":
        if not args.tflite or not args.hef:
            log.error("Both --tflite and --hef are required for --mode compare")
            sys.exit(1)
        cpu_r   = run_tflite_benchmark(args.tflite, args.runs, args.warmup)
        hailo_r = run_hailo_benchmark(args.hef,    args.runs, args.warmup)
        print_result(cpu_r)
        print_result(hailo_r)
        print_comparison(cpu_r, hailo_r)
        results = [cpu_r, hailo_r]

    if results:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
