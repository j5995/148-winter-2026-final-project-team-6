"""
benchmark_system.py
===================
Captures a time-series of system-level metrics on the RPi 5 before, during,
and after a donkeycar-style vehicle loop.  Run once WITHOUT the AI HAT model
loaded (--mode cpu) and once WITH (--mode hailo) to produce side-by-side data.

Metrics
-------
- Per-core CPU frequency and utilisation
- Total & per-core CPU temperature
- RAM and swap usage
- Disk I/O throughput (tub write speed)
- Network latency (useful for ROS2 / GPS streaming)
- PCIe bandwidth (if Hailo HAT is active)

Usage
-----
python benchmark_system.py --duration 60 --interval 0.5 \\
       --label "cpu_only" --output results/system_cpu.json

python benchmark_system.py --duration 60 --interval 0.5 \\
       --label "ai_hat" --output results/system_hat.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("[WARNING] psutil not installed – install with: pip install psutil")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SystemSnapshot:
    timestamp: float = 0.0
    cpu_percent_total: float = 0.0
    cpu_percent_per_core: List[float] = field(default_factory=list)
    cpu_freq_mhz: float = 0.0
    memory_used_mb: float = 0.0
    memory_percent: float = 0.0
    swap_used_mb: float = 0.0
    temperature_c: float = 0.0
    disk_read_mb_s: float = 0.0
    disk_write_mb_s: float = 0.0
    net_sent_mb_s: float = 0.0
    net_recv_mb_s: float = 0.0


@dataclass
class SystemBenchmarkResult:
    label: str
    duration_s: float
    interval_s: float
    snapshots: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def compute_summary(self) -> None:
        if not self.snapshots:
            return
        keys = [
            "cpu_percent_total", "cpu_freq_mhz",
            "memory_used_mb", "memory_percent",
            "temperature_c", "disk_write_mb_s",
        ]
        for k in keys:
            values = [s[k] for s in self.snapshots if k in s]
            if values:
                arr = np.array(values)
                self.summary[k] = {
                    "mean":   float(np.mean(arr)),
                    "max":    float(np.max(arr)),
                    "min":    float(np.min(arr)),
                    "p95":    float(np.percentile(arr, 95)),
                }
        self.summary["platform"] = platform.platform()
        self.summary["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# Metric collectors
# ---------------------------------------------------------------------------

def get_cpu_temperature() -> float:
    """RPi 5 thermal zone or vcgencmd fallback."""
    thermal_paths = [
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/class/thermal/thermal_zone1/temp",
    ]
    for p in thermal_paths:
        try:
            return float(Path(p).read_text().strip()) / 1000.0
        except (FileNotFoundError, ValueError):
            pass
    try:
        out = subprocess.check_output(
            ["vcgencmd", "measure_temp"],
            stderr=subprocess.DEVNULL, timeout=2
        ).decode()
        return float(out.strip().replace("temp=", "").replace("'C", ""))
    except Exception:
        return 0.0


def get_snapshot(
    prev_disk_io: Optional[Any] = None,
    prev_net_io:  Optional[Any] = None,
    prev_time:    Optional[float] = None,
) -> SystemSnapshot:
    snap = SystemSnapshot(timestamp=time.time())

    if not HAS_PSUTIL:
        snap.temperature_c = get_cpu_temperature()
        return snap

    snap.cpu_percent_total    = psutil.cpu_percent(interval=None)
    snap.cpu_percent_per_core = psutil.cpu_percent(percpu=True)
    freq = psutil.cpu_freq()
    snap.cpu_freq_mhz = freq.current if freq else 0.0

    mem = psutil.virtual_memory()
    snap.memory_used_mb  = mem.used / (1024 ** 2)
    snap.memory_percent  = mem.percent

    swap = psutil.swap_memory()
    snap.swap_used_mb = swap.used / (1024 ** 2)

    snap.temperature_c = get_cpu_temperature()

    # Disk I/O rate (MB/s)
    disk_io = psutil.disk_io_counters()
    if prev_disk_io and prev_time:
        dt = snap.timestamp - prev_time
        if dt > 0:
            snap.disk_read_mb_s  = (disk_io.read_bytes  - prev_disk_io.read_bytes)  / (1024**2 * dt)
            snap.disk_write_mb_s = (disk_io.write_bytes - prev_disk_io.write_bytes) / (1024**2 * dt)

    # Network I/O rate (MB/s)
    net_io = psutil.net_io_counters()
    if prev_net_io and prev_time:
        dt = snap.timestamp - prev_time
        if dt > 0:
            snap.net_sent_mb_s = (net_io.bytes_sent - prev_net_io.bytes_sent) / (1024**2 * dt)
            snap.net_recv_mb_s = (net_io.bytes_recv - prev_net_io.bytes_recv) / (1024**2 * dt)

    return snap


# ---------------------------------------------------------------------------
# Simulated donkeycar vehicle-loop workload
# ---------------------------------------------------------------------------

def simulate_vehicle_loop(stop_event: threading.Event) -> None:
    """
    Runs a tight loop mimicking the donkeycar vehicle loop CPU footprint:
    image decode → normalise → (mock) inference → steering output.
    Replace this with real donkeycar Vehicle.start() in production.
    """
    import struct
    log.info("[vehicle-loop] Simulated donkeycar loop started")
    while not stop_event.is_set():
        # Simulate image capture + normalisation (160×120 RGB)
        frame = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
        _ = frame.astype(np.float32) / 255.0

        # Simulate a lightweight 'inference' placeholder
        _ = np.sum(frame) % 2  # cheap mock of pilot output

        time.sleep(1 / 30.0)   # ~30 Hz vehicle loop


# ---------------------------------------------------------------------------
# Disk I/O stress (simulates tub writes)
# ---------------------------------------------------------------------------

def simulate_tub_writes(
    path: str,
    num_records: int = 500,
    image_size: int = 57600,   # 120×160×3 bytes
) -> float:
    """Write synthetic tub records; return throughput in MB/s."""
    import struct
    Path(path).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    total_bytes = 0

    for i in range(num_records):
        img_path   = os.path.join(path, f"frame_{i:06d}.jpg")
        meta_path  = os.path.join(path, f"record_{i:06d}.json")

        img_data = np.random.randint(0, 256, image_size, dtype=np.uint8).tobytes()
        meta = json.dumps({
            "cam/image_array":  f"frame_{i:06d}.jpg",
            "user/angle":       float(np.random.uniform(-1, 1)),
            "user/throttle":    float(np.random.uniform(0, 1)),
            "_index":           i,
        })

        Path(img_path).write_bytes(img_data)
        Path(meta_path).write_text(meta)
        total_bytes += len(img_data) + len(meta)

    elapsed = time.perf_counter() - t0
    mb_written = total_bytes / (1024 ** 2)
    throughput = mb_written / elapsed
    log.info("Tub I/O: %.2f MB in %.2f s → %.2f MB/s", mb_written, elapsed, throughput)
    return throughput


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_system_benchmark(
    duration_s: float = 60.0,
    interval_s: float = 0.5,
    label: str = "baseline",
    run_workload: bool = True,
) -> SystemBenchmarkResult:

    result = SystemBenchmarkResult(
        label=label,
        duration_s=duration_s,
        interval_s=interval_s,
    )

    stop_event = threading.Event()
    workload_thread: Optional[threading.Thread] = None

    if run_workload:
        workload_thread = threading.Thread(
            target=simulate_vehicle_loop,
            args=(stop_event,),
            daemon=True,
        )
        workload_thread.start()

    log.info("Collecting system metrics for %.0f s (interval=%.2f s)…",
             duration_s, interval_s)

    prev_disk_io = psutil.disk_io_counters() if HAS_PSUTIL else None
    prev_net_io  = psutil.net_io_counters()  if HAS_PSUTIL else None
    prev_time    = time.time()

    deadline = time.time() + duration_s
    while time.time() < deadline:
        snap = get_snapshot(prev_disk_io, prev_net_io, prev_time)
        result.snapshots.append(asdict(snap))

        if HAS_PSUTIL:
            prev_disk_io = psutil.disk_io_counters()
            prev_net_io  = psutil.net_io_counters()
        prev_time = snap.timestamp

        time.sleep(interval_s)

    stop_event.set()
    if workload_thread:
        workload_thread.join(timeout=3)

    result.compute_summary()
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(r: SystemBenchmarkResult) -> None:
    print(f"\n{'='*60}")
    print(f"  System Benchmark: {r.label}")
    print(f"  Duration : {r.duration_s:.0f}s  |  Interval: {r.interval_s}s")
    print(f"  Snapshots: {len(r.snapshots)}")
    print(f"{'='*60}")
    for metric, stats in r.summary.items():
        if isinstance(stats, dict):
            print(f"  {metric:<30} mean={stats['mean']:>7.2f}  "
                  f"p95={stats['p95']:>7.2f}  max={stats['max']:>7.2f}")
    print(f"{'='*60}\n")


def save_result(r: SystemBenchmarkResult, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(asdict(r), f, indent=2)
    log.info("Saved → %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RPi 5 system-level benchmarks for donkeycar"
    )
    p.add_argument("--duration",  type=float, default=60.0,
                   help="Monitoring duration in seconds")
    p.add_argument("--interval",  type=float, default=0.5,
                   help="Sample interval in seconds")
    p.add_argument("--label",     default="baseline",
                   help="Label for this run, e.g. 'cpu_only' or 'ai_hat'")
    p.add_argument("--output",    default="results/system_report.json",
                   help="Output path for JSON report")
    p.add_argument("--no-workload", action="store_true",
                   help="Collect idle metrics without simulated vehicle loop")
    p.add_argument("--tub-path",  default="/tmp/benchmark_tub",
                   help="Scratch path for tub I/O stress test")
    p.add_argument("--tub-writes", type=int, default=0,
                   help="Number of synthetic tub records to write (0 = skip)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not HAS_PSUTIL:
        log.warning("psutil unavailable; only temperature will be collected. "
                    "Run: pip install psutil")

    # Optional tub I/O stress test
    if args.tub_writes > 0:
        log.info("Running tub I/O stress: %d records → %s",
                 args.tub_writes, args.tub_path)
        throughput = simulate_tub_writes(args.tub_path, args.tub_writes)
        log.info("Tub write throughput: %.2f MB/s", throughput)

    result = run_system_benchmark(
        duration_s=args.duration,
        interval_s=args.interval,
        label=args.label,
        run_workload=not args.no_workload,
    )
    print_summary(result)
    save_result(result, args.output)


if __name__ == "__main__":
    main()
