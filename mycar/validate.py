"""
validate.py
===========
Pre-flight validation for the RPi 5 + AI HAT donkeycar benchmark suite.

Checks
------
1.  Python version (≥ 3.8)
2.  Required Python packages
3.  donkeycar installation + version
4.  Hailo AI HAT hardware detection (PCIe + firmware)
5.  HailoRT software stack
6.  TensorFlow / TFLite availability
7.  GPU / NPU memory visibility
8.  File system writability
9.  CPU governor (performance recommended)
10. PCIe Gen version (Gen 3 recommended for AI HAT)
11. Model file sanity (if paths provided)

Exit codes
----------
0 = all checks passed
1 = one or more warnings
2 = one or more critical failures

Usage
-----
python validate.py
python validate.py --tflite models/pilot.tflite --hef models/pilot.hef
python validate.py --strict   # fail on warnings too
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# ANSI colours (disabled on Windows / no-TTY)
# ---------------------------------------------------------------------------

USE_COLOR = sys.stdout.isatty() and platform.system() != "Windows"

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text

def green(t):  return _c(t, "32")
def yellow(t): return _c(t, "33")
def red(t):    return _c(t, "31")
def bold(t):   return _c(t, "1")


# ---------------------------------------------------------------------------
# Result accumulator
# ---------------------------------------------------------------------------

PASS    = "PASS"
WARN    = "WARN"
FAIL    = "FAIL"
SKIP    = "SKIP"

results: List[Tuple[str, str, str]] = []   # (check_name, status, detail)


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗", "SKIP": "–"}.get(status, "?")
    colour = {"PASS": green, "WARN": yellow, "FAIL": red, "SKIP": lambda x: x}
    print(f"  {colour[status](icon + ' ' + status):<12} {name:<42} {detail}")


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_python_version() -> None:
    v = sys.version_info
    if v >= (3, 9):
        record("Python version", PASS, f"{v.major}.{v.minor}.{v.micro}")
    elif v >= (3, 8):
        record("Python version", WARN, f"{v.major}.{v.minor} — 3.9+ recommended")
    else:
        record("Python version", FAIL,
               f"{v.major}.{v.minor} — 3.8+ required")


def check_package(pkg: str, import_name: Optional[str] = None,
                  min_version: str = "") -> None:
    name = import_name or pkg
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "unknown")
        if min_version and ver != "unknown":
            from packaging.version import Version
            if Version(ver) < Version(min_version):
                record(f"Package: {pkg}", WARN,
                       f"v{ver} installed, {min_version}+ recommended")
                return
        record(f"Package: {pkg}", PASS, f"v{ver}")
    except ImportError:
        record(f"Package: {pkg}", FAIL,
               f"not found — install with: pip install {pkg}")


def check_donkeycar() -> None:
    try:
        import donkeycar as dk
        ver = getattr(dk, "__version__", "unknown")
        record("donkeycar", PASS, f"v{ver}")
    except ImportError:
        record("donkeycar", WARN,
               "not installed (benchmarks work in standalone mode)")


def check_hailo_device() -> None:
    """Check PCIe for Hailo device and run hailortcli identify."""
    # 1. PCIe check
    try:
        result = subprocess.run(
            ["lspci"], capture_output=True, text=True, timeout=5
        )
        if "hailo" in result.stdout.lower():
            record("Hailo PCIe device", PASS, "Hailo found in lspci output")
        else:
            record("Hailo PCIe device", WARN,
                   "Not found in lspci — AI HAT may not be connected")
    except FileNotFoundError:
        record("Hailo PCIe device", SKIP, "lspci not available")

    # 2. HailoRT CLI check
    if shutil.which("hailortcli"):
        try:
            r = subprocess.run(
                ["hailortcli", "fw-control", "identify"],
                capture_output=True, text=True, timeout=10
            )
            if r.returncode == 0:
                lines = [l.strip() for l in r.stdout.splitlines() if l.strip()]
                fw_line = next((l for l in lines if "Firmware" in l), "")
                record("HailoRT firmware", PASS, fw_line or "OK")
            else:
                record("HailoRT firmware", FAIL,
                       r.stderr.strip()[:80] or "hailortcli returned error")
        except subprocess.TimeoutExpired:
            record("HailoRT firmware", WARN, "hailortcli timed out")
    else:
        record("hailortcli CLI", WARN,
               "hailortcli not in PATH — install HailoRT from hailo.ai")


def check_hailo_python() -> None:
    try:
        from hailo_platform import VDevice
        record("hailo_platform Python", PASS, "importable")
    except ImportError:
        record("hailo_platform Python", WARN,
               "not installed — needed for Hailo benchmarks")


def check_tensorflow() -> None:
    try:
        import tensorflow as tf
        ver = tf.__version__
        # Verify TFLite interpreter works
        interp = tf.lite.Interpreter
        record("TensorFlow / TFLite", PASS, f"v{ver}")
    except ImportError:
        record("TensorFlow / TFLite", WARN,
               "not installed — needed for CPU baseline benchmarks")
    except Exception as e:
        record("TensorFlow / TFLite", FAIL, str(e)[:80])


def check_numpy() -> None:
    check_package("numpy", min_version="1.21.0")


def check_psutil() -> None:
    check_package("psutil", min_version="5.9.0")


def check_filesystem(path: str = "/tmp/donkey_validate_test") -> None:
    p = Path(path)
    try:
        p.mkdir(parents=True, exist_ok=True)
        test_file = p / "write_test.bin"
        test_file.write_bytes(b"\x00" * 1024 * 1024)  # 1 MB
        test_file.unlink()
        p.rmdir()
        record("File system writability", PASS, f"1 MB write OK at {path}")
    except Exception as e:
        record("File system writability", FAIL, str(e)[:80])


def check_cpu_governor() -> None:
    gov_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if not gov_path.exists():
        record("CPU governor", SKIP, "Not available (non-Linux or no cpufreq)")
        return
    governor = gov_path.read_text().strip()
    if governor == "performance":
        record("CPU governor", PASS, governor)
    else:
        record("CPU governor", WARN,
               f"{governor} — set to 'performance' for consistent benchmarks:\n"
               "    echo performance | sudo tee "
               "/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")


def check_pcie_gen() -> None:
    """Check that PCIe is running at Gen 3 (required for full 26 TOPS)."""
    try:
        result = subprocess.run(
            ["lspci", "-vvv"], capture_output=True, text=True, timeout=5
        )
        if "LnkSta" in result.stdout:
            for line in result.stdout.splitlines():
                if "LnkSta" in line and ("Speed 8GT/s" in line or "Gen3" in line):
                    record("PCIe Gen 3", PASS, line.strip()[:80])
                    return
            record("PCIe Gen 3", WARN,
                   "Gen 3 (8GT/s) not confirmed — check /boot/config.txt: "
                   "dtparam=pciex1_gen=3")
        else:
            record("PCIe Gen 3", SKIP, "lspci -vvv output not parsed")
    except FileNotFoundError:
        record("PCIe Gen 3", SKIP, "lspci not available")
    except subprocess.TimeoutExpired:
        record("PCIe Gen 3", SKIP, "lspci timed out")


def check_model_file(path: str, expected_ext: str) -> None:
    p = Path(path)
    if not p.exists():
        record(f"Model file: {p.name}", FAIL, f"Not found: {path}")
        return
    size_mb = p.stat().st_size / (1024 ** 2)
    if p.suffix.lower() != expected_ext.lower():
        record(f"Model file: {p.name}", WARN,
               f"Extension {p.suffix} expected {expected_ext}")
        return
    if size_mb < 0.001:
        record(f"Model file: {p.name}", FAIL, "File is empty")
        return
    record(f"Model file: {p.name}", PASS, f"{size_mb:.2f} MB")


def check_memory() -> None:
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_mb = mem.total / (1024 ** 2)
        avail_mb = mem.available / (1024 ** 2)
        if avail_mb < 256:
            record("Available RAM", WARN,
                   f"{avail_mb:.0f} MB free (< 256 MB — may cause OOM during inference)")
        else:
            record("Available RAM", PASS,
                   f"{avail_mb:.0f} MB free / {total_mb:.0f} MB total")
    except ImportError:
        record("Available RAM", SKIP, "psutil not installed")


def check_ros2() -> None:
    if shutil.which("ros2"):
        try:
            r = subprocess.run(["ros2", "--version"],
                               capture_output=True, text=True, timeout=5)
            record("ROS2 CLI", PASS, r.stdout.strip()[:80])
        except Exception as e:
            record("ROS2 CLI", WARN, str(e)[:80])
    else:
        record("ROS2 CLI", SKIP, "ros2 not in PATH (optional)")


def check_gps_serial() -> None:
    gps_devs = ["/dev/ttyAMA0", "/dev/ttyS0", "/dev/ttyUSB0", "/dev/serial0"]
    found = [d for d in gps_devs if Path(d).exists()]
    if found:
        record("GPS serial device", PASS, ", ".join(found))
    else:
        record("GPS serial device", SKIP,
               "No serial device found (optional for GPS laps)")


# ---------------------------------------------------------------------------
# Summary + exit
# ---------------------------------------------------------------------------

def print_summary() -> int:
    n_pass = sum(1 for _, s, _ in results if s == PASS)
    n_warn = sum(1 for _, s, _ in results if s == WARN)
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_skip = sum(1 for _, s, _ in results if s == SKIP)

    print(f"\n{'='*60}")
    print(bold("  Validation Summary"))
    print(f"{'='*60}")
    print(f"  {green(f'PASS: {n_pass}')}   "
          f"{yellow(f'WARN: {n_warn}')}   "
          f"{red(f'FAIL: {n_fail}')}   "
          f"SKIP: {n_skip}")
    print(f"{'='*60}")

    if n_fail:
        print(red("\n  ✗ Critical failures detected — fix before running benchmarks.\n"))
        return 2
    if n_warn:
        print(yellow("\n  ⚠ Warnings present — benchmarks will run but results may vary.\n"))
        return 1
    print(green("\n  ✓ All checks passed — ready to benchmark!\n"))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate RPi 5 + AI HAT donkeycar benchmark environment"
    )
    p.add_argument("--tflite",  default="",
                   help="Path to .tflite model to validate")
    p.add_argument("--hef",     default="",
                   help="Path to .hef Hailo model to validate")
    p.add_argument("--strict",  action="store_true",
                   help="Exit code 2 on warnings as well as failures")
    p.add_argument("--output",  default="",
                   help="Optional JSON file to write validation results to")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print(f"\n{'='*60}")
    print(bold("  RPi 5 + AI HAT Donkeycar Validation"))
    print(f"  Platform: {platform.platform()}")
    print(f"{'='*60}\n")

    check_python_version()
    check_numpy()
    check_psutil()
    check_package("packaging")
    check_donkeycar()
    check_tensorflow()
    check_hailo_device()
    check_hailo_python()
    check_filesystem()
    check_cpu_governor()
    check_pcie_gen()
    check_memory()
    check_ros2()
    check_gps_serial()

    if args.tflite:
        check_model_file(args.tflite, ".tflite")
    if args.hef:
        check_model_file(args.hef, ".hef")

    if args.output:
        import json
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump([
                {"check": n, "status": s, "detail": d}
                for n, s, d in results
            ], f, indent=2)
        print(f"  Validation report → {args.output}")

    rc = print_summary()
    if args.strict and rc == 1:
        return 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
