"""
benchmark_pipeline.py
=====================
Benchmarks the full donkeycar vehicle pipeline end-to-end:
  Camera → Normalise → Model Inference → Steering/Throttle output

Designed to work with donkeycar's 'parts' architecture.
Falls back to a pure-synthetic pipeline if donkeycar is not importable
(useful for CI / development machines).

Key metrics
-----------
- End-to-end loop latency (ms)
- Per-part timing breakdown
- Sustained FPS at different loop rates (10 Hz, 20 Hz, 30 Hz)
- CPU headroom remaining at each rate
- Memory drift over time (detect leaks)

Usage
-----
# With real donkeycar installed:
python benchmark_pipeline.py --mode tflite --model models/pilot.tflite

# With AI HAT:
python benchmark_pipeline.py --mode hailo --model models/pilot.hef

# Synthetic (no donkeycar required):
python benchmark_pipeline.py --mode synthetic --runs 500
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import donkeycar as dk
    from donkeycar.vehicle import Vehicle
    HAS_DONKEYCAR = True
except ImportError:
    HAS_DONKEYCAR = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------

@contextmanager
def timer(label: str, store: Dict[str, List[float]]):
    t0 = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    store.setdefault(label, []).append(elapsed_ms)


# ---------------------------------------------------------------------------
# Part stubs (used when donkeycar is not installed)
# ---------------------------------------------------------------------------

class SyntheticCamera:
    """Generates random frames at a fixed resolution."""
    def __init__(self, h: int = 120, w: int = 160, d: int = 3):
        self.h, self.w, self.d = h, w, d

    def run(self) -> np.ndarray:
        return np.random.randint(0, 256, (self.h, self.w, self.d),
                                 dtype=np.uint8)


class NormalisePart:
    """Normalise uint8 image to float32 [0,1]."""
    def run(self, img: np.ndarray) -> np.ndarray:
        return img.astype(np.float32) / 255.0


class SyntheticPilot:
    """Mock steering/throttle pilot using a random linear operation."""
    def __init__(self, h: int = 120, w: int = 160):
        self.weights = np.random.randn(h * 160 * 3).astype(np.float32)

    def run(self, img: np.ndarray) -> Tuple[float, float]:
        flat = img.flatten()
        steering  = float(np.tanh(np.dot(flat[:len(self.weights)], self.weights[:len(flat)])))
        throttle  = float(np.clip(abs(steering), 0.0, 1.0))
        return steering, throttle


class TFLitePilot:
    """Wraps a TFLite model for donkeycar-style run(img) → (angle, throttle)."""
    def __init__(self, model_path: str):
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        shape = self.input_details[0]["shape"]  # (1, H, W, C)
        self.h, self.w, self.d = int(shape[1]), int(shape[2]), int(shape[3])

    def run(self, img: np.ndarray) -> Tuple[float, float]:
        inp = img.astype(np.float32) / 255.0
        self.interpreter.set_tensor(
            self.input_details[0]["index"],
            inp.reshape(1, self.h, self.w, self.d),
        )
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]["index"])
        angle    = float(out[0][0])
        throttle = float(out[0][1]) if out.shape[-1] > 1 else 0.3
        return angle, throttle


class HailoPilot:
    """Wraps a Hailo .hef model for donkeycar-style run(img) → (angle, throttle)."""
    def __init__(self, model_path: str):
        from hailo_platform import (HEF, ConfigureParams, FormatType,
                                     HailoStreamInterface, InferVStreams,
                                     InputVStreamParams, OutputVStreamParams,
                                     VDevice)
        self.hef     = HEF(model_path)
        self.target  = VDevice()
        cfg_params   = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe)
        groups       = self.target.configure(self.hef, cfg_params)
        self.ng      = groups[0]
        self.ng_params = self.ng.create_params()
        self.in_params  = InputVStreamParams.make(
            self.ng, quantized=False, format_type=FormatType.FLOAT32)
        self.out_params = OutputVStreamParams.make(
            self.ng, quantized=True,  format_type=FormatType.FLOAT32)
        info = self.hef.get_input_vstream_infos()[0]
        self.h, self.w, self.d = info.shape
        self.input_name = info.name
        self._activated = self.ng.activate(self.ng_params)
        self._activated.__enter__()
        self._pipeline = InferVStreams(self.ng, self.in_params, self.out_params)
        self._pipeline.__enter__()

    def run(self, img: np.ndarray) -> Tuple[float, float]:
        frame = img.astype(np.float32) / 255.0
        results = self._pipeline.infer(
            {self.input_name: frame[np.newaxis, ...]})
        out = list(results.values())[0][0]
        angle    = float(out[0])
        throttle = float(out[1]) if len(out) > 1 else 0.3
        return angle, throttle


class MockActuator:
    """Accepts steering/throttle and records them (no hardware needed)."""
    def __init__(self):
        self.last_angle    = 0.0
        self.last_throttle = 0.0

    def run(self, angle: float, throttle: float) -> None:
        self.last_angle    = angle
        self.last_throttle = throttle


# ---------------------------------------------------------------------------
# Pipeline benchmark
# ---------------------------------------------------------------------------

@dataclass
class PartTimings:
    camera_ms:    List[float] = field(default_factory=list)
    normalise_ms: List[float] = field(default_factory=list)
    inference_ms: List[float] = field(default_factory=list)
    actuator_ms:  List[float] = field(default_factory=list)
    loop_total_ms:List[float] = field(default_factory=list)

    def stats(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, vals in asdict(self).items():
            if vals:
                arr = np.array(vals)
                out[name] = {
                    "mean": float(np.mean(arr)),
                    "p95":  float(np.percentile(arr, 95)),
                    "max":  float(np.max(arr)),
                    "fps":  1000.0 / float(np.mean(arr)) if np.mean(arr) > 0 else 0,
                }
        return out


def build_pilot(mode: str, model_path: str):
    if mode == "synthetic":
        return SyntheticPilot()
    elif mode == "tflite":
        return TFLitePilot(model_path)
    elif mode == "hailo":
        return HailoPilot(model_path)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def run_pipeline_benchmark(
    mode: str = "synthetic",
    model_path: str = "",
    num_runs: int = 300,
    warmup: int = 20,
    target_hz: int = 30,
    h: int = 120, w: int = 160, d: int = 3,
) -> Dict[str, Any]:

    log.info("Setting up pipeline (mode=%s)…", mode)
    camera    = SyntheticCamera(h, w, d)
    normalise = NormalisePart()
    pilot     = build_pilot(mode, model_path)
    actuator  = MockActuator()
    timings   = PartTimings()

    frame_period = 1.0 / target_hz

    # --- Warmup ---
    log.info("Warming up (%d frames)…", warmup)
    for _ in range(warmup):
        img = camera.run()
        img = normalise.run(img)
        pilot.run(img)

    # --- Benchmark loop ---
    log.info("Running %d frames at target %d Hz…", num_runs, target_hz)
    memory_start = psutil.Process(os.getpid()).memory_info().rss if HAS_PSUTIL else 0

    for i in range(num_runs):
        loop_t0 = time.perf_counter()

        with timer("camera",    timings.__dict__): img = camera.run()
        with timer("normalise", timings.__dict__): img = normalise.run(img)
        with timer("inference", timings.__dict__): angle, throttle = pilot.run(img)
        with timer("actuator",  timings.__dict__): actuator.run(angle, throttle)

        loop_elapsed = (time.perf_counter() - loop_t0) * 1000.0
        timings.loop_total_ms.append(loop_elapsed)

        # Pace the loop if there's headroom
        remaining = frame_period - (loop_elapsed / 1000.0)
        if remaining > 0:
            time.sleep(remaining)

        if i % 100 == 0:
            log.debug("  [%d/%d] loop=%.1f ms  inference=%.1f ms",
                      i + 1, num_runs,
                      timings.loop_total_ms[-1],
                      timings.__dict__["inference"][-1] if "inference" in timings.__dict__ else 0)

    memory_end = psutil.Process(os.getpid()).memory_info().rss if HAS_PSUTIL else 0
    memory_delta_mb = (memory_end - memory_start) / (1024 ** 2)

    stats = timings.stats()
    result = {
        "mode":           mode,
        "model_path":     model_path,
        "num_runs":       num_runs,
        "target_hz":      target_hz,
        "timings":        stats,
        "memory_delta_mb": memory_delta_mb,
        "sustained_fps":  1000.0 / stats.get("loop_total_ms", {}).get("mean", 1),
        "headroom_pct":   max(0.0, 100.0 * (
            1 - stats.get("loop_total_ms", {}).get("mean", 0) / (1000.0 / target_hz)
        )),
        "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    print(f"\n{'='*60}")
    print(f"  Pipeline benchmark — {mode.upper()}")
    print(f"{'='*60}")
    for part, s in stats.items():
        print(f"  {part:<18} mean={s['mean']:>7.2f} ms  "
              f"p95={s['p95']:>7.2f} ms  fps={s['fps']:>7.1f}")
    print(f"\n  Sustained FPS   : {result['sustained_fps']:.1f}")
    print(f"  CPU headroom    : {result['headroom_pct']:.1f}%")
    print(f"  Memory drift    : {memory_delta_mb:+.2f} MB")
    print(f"{'='*60}\n")

    return result


# ---------------------------------------------------------------------------
# Multi-rate sweep
# ---------------------------------------------------------------------------

def hz_sweep(
    mode: str,
    model_path: str,
    rates: List[int],
    runs: int = 100,
) -> List[Dict[str, Any]]:
    """Run pipeline at several Hz targets to find the sustainable maximum."""
    results = []
    for hz in rates:
        log.info("=== Testing at %d Hz ===", hz)
        r = run_pipeline_benchmark(
            mode=mode,
            model_path=model_path,
            num_runs=runs,
            target_hz=hz,
        )
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end pipeline benchmarks for donkeycar on RPi 5"
    )
    p.add_argument("--mode",    choices=["synthetic", "tflite", "hailo"],
                   default="synthetic")
    p.add_argument("--model",   default="",
                   help="Path to .tflite or .hef model")
    p.add_argument("--runs",    type=int, default=300)
    p.add_argument("--warmup",  type=int, default=20)
    p.add_argument("--hz",      type=int, default=30,
                   help="Target loop rate in Hz")
    p.add_argument("--hz-sweep", action="store_true",
                   help="Sweep across 10/20/30/40 Hz to find max sustainable rate")
    p.add_argument("--output",  default="results/pipeline_report.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.hz_sweep:
        results = hz_sweep(args.mode, args.model,
                           rates=[10, 20, 30, 40],
                           runs=args.runs)
    else:
        results = [run_pipeline_benchmark(
            mode=args.mode,
            model_path=args.model,
            num_runs=args.runs,
            warmup=args.warmup,
            target_hz=args.hz,
        )]

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved → %s", args.output)


if __name__ == "__main__":
    main()
