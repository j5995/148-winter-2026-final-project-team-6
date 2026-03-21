"""
test_benchmarks.py
==================
Pytest test suite for the RPi 5 + AI HAT donkeycar benchmark suite.

Test categories
---------------
Unit tests  – test individual functions with synthetic data
Integration – test full benchmark flows end-to-end (no hardware needed)
Validation  – test that validate.py logic is internally consistent

Run
---
pip install pytest pytest-timeout
pytest test_benchmarks.py -v
pytest test_benchmarks.py -v -k "not slow"  # skip long tests
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Make sure the benchmark package is importable
sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_tub(tmp_path):
    """Temporary tub directory, cleaned up automatically."""
    tub = tmp_path / "benchmark_tub"
    tub.mkdir()
    yield str(tub)
    shutil.rmtree(str(tub), ignore_errors=True)


@pytest.fixture
def tmp_results(tmp_path):
    """Temporary results directory."""
    results = tmp_path / "results"
    results.mkdir()
    return results


@pytest.fixture
def dummy_tflite_model(tmp_path):
    """Create a minimal valid TFLite flatbuffer for testing."""
    try:
        import tensorflow as tf
        # Build a tiny model: (1, 120, 160, 3) → (1, 2)
        inp = tf.keras.Input(shape=(120, 160, 3))
        x   = tf.keras.layers.GlobalAveragePooling2D()(inp)
        out = tf.keras.layers.Dense(2)(x)
        model = tf.keras.Model(inp, out)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        path = str(tmp_path / "pilot.tflite")
        Path(path).write_bytes(tflite_model)
        return path
    except ImportError:
        pytest.skip("TensorFlow not installed")


# ---------------------------------------------------------------------------
# Tests: benchmark_tub
# ---------------------------------------------------------------------------

class TestTubBenchmark:

    def test_write_creates_files(self, tmp_tub):
        from benchmark_tub import benchmark_write_fs
        result = benchmark_write_fs(tmp_tub, num_records=20, with_images=False)
        assert result.num_records == 20
        assert result.errors == 0
        assert result.records_per_s > 0
        assert result.total_time_s > 0

    def test_write_with_images_creates_both_files(self, tmp_tub):
        from benchmark_tub import benchmark_write_fs
        result = benchmark_write_fs(tmp_tub, num_records=5, with_images=True)
        json_files = list(Path(tmp_tub).glob("record_*.json"))
        img_files  = list(Path(tmp_tub).glob("cam_image_array_*.npy"))
        assert len(json_files) == 5
        assert len(img_files)  == 5
        assert result.errors == 0

    def test_read_after_write(self, tmp_tub):
        from benchmark_tub import benchmark_write_fs, benchmark_read_fs
        benchmark_write_fs(tmp_tub, num_records=30, with_images=False)
        result = benchmark_read_fs(tmp_tub)
        assert result.num_records == 30
        assert result.errors == 0
        assert result.records_per_s > 0

    def test_delete_reduces_file_count(self, tmp_tub):
        from benchmark_tub import benchmark_write_fs, benchmark_delete_fs
        benchmark_write_fs(tmp_tub, num_records=20, with_images=False)
        result = benchmark_delete_fs(tmp_tub, num_deletes=5)
        remaining = list(Path(tmp_tub).glob("record_*.json"))
        assert len(remaining) <= 15
        assert result.errors == 0

    def test_delete_on_empty_tub(self, tmp_tub):
        from benchmark_tub import benchmark_delete_fs
        # Should not raise
        result = benchmark_delete_fs(tmp_tub, num_deletes=10)
        assert result.num_records == 0

    def test_latency_stats_computed(self, tmp_tub):
        from benchmark_tub import benchmark_write_fs
        result = benchmark_write_fs(tmp_tub, num_records=50, with_images=False)
        result.compute()
        assert result.mean_ms > 0
        assert result.p99_ms >= result.p95_ms   # p99 always ≥ p95
        assert result.p95_ms > 0
        assert result.timestamp != ""

    def test_concurrent_benchmark(self, tmp_tub):
        from benchmark_tub import benchmark_concurrent
        result = benchmark_concurrent(tmp_tub + "_conc", num_records=30, hz=30.0)
        assert result["num_records"] == 30
        assert result["actual_hz"] > 0
        assert "write_stats" in result
        assert "read_stats" in result

    def test_mb_per_s_positive_with_images(self, tmp_tub):
        from benchmark_tub import benchmark_write_fs
        result = benchmark_write_fs(tmp_tub, num_records=10, with_images=True)
        assert result.mb_per_s > 0

    def test_json_records_are_valid(self, tmp_tub):
        from benchmark_tub import benchmark_write_fs
        benchmark_write_fs(tmp_tub, num_records=10, with_images=False)
        for f in Path(tmp_tub).glob("record_*.json"):
            data = json.loads(f.read_text())
            assert "_index" in data
            assert "user/angle" in data
            assert -1.0 <= data["user/angle"] <= 1.0

    def test_output_json_saved(self, tmp_tub, tmp_results):
        """End-to-end test: run main and check output file created."""
        from benchmark_tub import (
            benchmark_write_fs, benchmark_read_fs,
            benchmark_delete_fs, TubBenchmarkResult,
        )
        import dataclasses
        w = benchmark_write_fs(tmp_tub, 20, False)
        r = benchmark_read_fs(tmp_tub)
        d = benchmark_delete_fs(tmp_tub, 5)
        out = str(tmp_results / "tub_report.json")
        results = [dataclasses.asdict(w), dataclasses.asdict(r), dataclasses.asdict(d)]
        for item in results:
            item.pop("latencies_ms", None)
        with open(out, "w") as f:
            json.dump(results, f)
        assert Path(out).exists()
        loaded = json.loads(Path(out).read_text())
        assert len(loaded) == 3


# ---------------------------------------------------------------------------
# Tests: benchmark_system
# ---------------------------------------------------------------------------

class TestSystemBenchmark:

    def test_get_snapshot_returns_object(self):
        from benchmark_system import get_snapshot, SystemSnapshot
        snap = get_snapshot()
        assert isinstance(snap, SystemSnapshot)
        assert snap.timestamp > 0

    def test_temperature_function_returns_float(self):
        from benchmark_system import get_cpu_temperature
        temp = get_cpu_temperature()
        assert isinstance(temp, float)
        # On non-Pi hardware, returns 0.0 — that's OK
        assert temp >= 0.0

    def test_tub_writes_throughput(self, tmp_tub):
        from benchmark_system import simulate_tub_writes
        throughput = simulate_tub_writes(tmp_tub, num_records=20)
        assert throughput > 0.0

    def test_system_benchmark_short(self, tmp_results):
        from benchmark_system import run_system_benchmark
        result = run_system_benchmark(
            duration_s=2.0,
            interval_s=0.2,
            label="test_run",
            run_workload=True,
        )
        assert result.label == "test_run"
        assert len(result.snapshots) > 0
        assert result.summary  # non-empty dict

    def test_summary_keys_present(self, tmp_results):
        from benchmark_system import run_system_benchmark
        result = run_system_benchmark(
            duration_s=1.0, interval_s=0.25,
            label="key_check", run_workload=False,
        )
        assert "timestamp" in result.summary

    def test_save_and_reload(self, tmp_results):
        from benchmark_system import run_system_benchmark, save_result
        import dataclasses
        result = run_system_benchmark(
            duration_s=1.0, interval_s=0.5,
            label="save_test", run_workload=False,
        )
        out = str(tmp_results / "system_test.json")
        save_result(result, out)
        assert Path(out).exists()
        data = json.loads(Path(out).read_text())
        assert data["label"] == "save_test"


# ---------------------------------------------------------------------------
# Tests: benchmark_pipeline
# ---------------------------------------------------------------------------

class TestPipelineBenchmark:

    def test_synthetic_camera(self):
        from benchmark_pipeline import SyntheticCamera
        cam = SyntheticCamera(120, 160, 3)
        frame = cam.run()
        assert frame.shape == (120, 160, 3)
        assert frame.dtype == np.uint8

    def test_normalise_part(self):
        from benchmark_pipeline import NormalisePart
        norm = NormalisePart()
        img  = np.ones((120, 160, 3), dtype=np.uint8) * 128
        out  = norm.run(img)
        assert out.dtype == np.float32
        assert np.allclose(out, 128 / 255.0)
        assert out.max() <= 1.0
        assert out.min() >= 0.0

    def test_synthetic_pilot_output_range(self):
        from benchmark_pipeline import SyntheticPilot
        pilot = SyntheticPilot(120, 160)
        img = np.random.rand(120, 160, 3).astype(np.float32)
        angle, throttle = pilot.run(img)
        assert -1.0 <= angle    <= 1.0
        assert  0.0 <= throttle <= 1.0

    def test_mock_actuator(self):
        from benchmark_pipeline import MockActuator
        act = MockActuator()
        act.run(0.5, 0.3)
        assert act.last_angle    == 0.5
        assert act.last_throttle == 0.3

    def test_synthetic_pipeline_runs(self):
        from benchmark_pipeline import run_pipeline_benchmark
        result = run_pipeline_benchmark(
            mode="synthetic", num_runs=50, warmup=5, target_hz=30,
        )
        assert result["sustained_fps"] > 0
        assert "timings" in result
        assert "loop_total_ms" in result["timings"]

    def test_pipeline_fps_exceeds_30hz(self):
        """Synthetic pipeline (no model) should easily sustain > 30 FPS."""
        from benchmark_pipeline import run_pipeline_benchmark
        result = run_pipeline_benchmark(
            mode="synthetic", num_runs=100, warmup=10, target_hz=30,
        )
        # Synthetic pilot is trivially fast; expect well above 30 FPS
        assert result["sustained_fps"] > 30

    def test_pipeline_memory_drift_is_small(self):
        """Memory should not grow by more than 50 MB over 200 frames."""
        from benchmark_pipeline import run_pipeline_benchmark
        result = run_pipeline_benchmark(
            mode="synthetic", num_runs=200, warmup=10, target_hz=30,
        )
        assert abs(result["memory_delta_mb"]) < 50

    def test_hz_sweep_returns_multiple_results(self):
        from benchmark_pipeline import hz_sweep
        results = hz_sweep("synthetic", "", rates=[10, 30], runs=30)
        assert len(results) == 2
        assert results[0]["target_hz"] == 10
        assert results[1]["target_hz"] == 30

    def test_tflite_pilot_with_real_model(self, dummy_tflite_model):
        from benchmark_pipeline import TFLitePilot
        pilot = TFLitePilot(dummy_tflite_model)
        img = np.random.rand(pilot.h, pilot.w, pilot.d).astype(np.float32)
        angle, throttle = pilot.run(img * 255)
        assert isinstance(angle, float)
        assert isinstance(throttle, float)

    def test_pipeline_with_tflite(self, dummy_tflite_model):
        from benchmark_pipeline import run_pipeline_benchmark
        result = run_pipeline_benchmark(
            mode="tflite",
            model_path=dummy_tflite_model,
            num_runs=30,
            warmup=5,
            target_hz=30,
        )
        assert result["sustained_fps"] > 0
        assert result["timings"]["inference_ms"]["mean"] > 0


# ---------------------------------------------------------------------------
# Tests: benchmark_inference
# ---------------------------------------------------------------------------

class TestInferenceBenchmark:

    def test_make_dummy_frame_shape(self):
        from benchmark_inference import make_dummy_frame
        frame = make_dummy_frame(120, 160, 3)
        assert frame.shape == (120, 160, 3)
        assert frame.dtype == np.uint8

    def test_inference_result_summarise(self):
        from benchmark_inference import InferenceResult
        r = InferenceResult(
            mode="cpu",
            model_path="test.tflite",
            num_runs=100,
            input_shape=(120, 160, 3),
            latencies_ms=list(np.random.uniform(10, 30, 100)),
        )
        r.summarise()
        assert r.mean_ms   > 0
        assert r.median_ms > 0
        assert r.p95_ms   >= r.median_ms
        assert r.p99_ms   >= r.p95_ms
        assert r.fps       > 0
        assert r.platform != ""

    def test_inference_result_to_dict(self):
        from benchmark_inference import InferenceResult
        r = InferenceResult(
            mode="cpu",
            model_path="m.tflite",
            num_runs=10,
            input_shape=(120, 160, 3),
            latencies_ms=[15.0] * 10,
        )
        r.summarise()
        d = r.to_dict()
        assert "mode" in d
        assert "fps" in d
        assert "latencies_ms" not in d   # should be stripped

    def test_check_hailo_returns_bool(self):
        from benchmark_inference import check_hailo_device
        result = check_hailo_device()
        assert isinstance(result, bool)

    def test_get_cpu_temperature_returns_float(self):
        from benchmark_inference import get_cpu_temperature
        temp = get_cpu_temperature()
        assert isinstance(temp, float)
        assert temp >= 0.0

    def test_save_results(self, tmp_results):
        from benchmark_inference import InferenceResult, save_results
        r = InferenceResult(
            mode="cpu",
            model_path="m.tflite",
            num_runs=50,
            input_shape=(120, 160, 3),
            latencies_ms=list(np.random.uniform(10, 30, 50)),
        )
        r.summarise()
        out = str(tmp_results / "inference_test")
        save_results([r], out)
        assert Path(out + ".json").exists()
        assert Path(out + ".csv").exists()
        data = json.loads(Path(out + ".json").read_text())
        assert data[0]["mode"] == "cpu"

    def test_tflite_benchmark(self, dummy_tflite_model):
        from benchmark_inference import run_tflite_benchmark
        result = run_tflite_benchmark(
            model_path=dummy_tflite_model,
            num_runs=30,
            warmup=5,
        )
        assert result.mode == "cpu"
        assert result.mean_ms > 0
        assert result.fps     > 0
        assert result.num_runs == 30


# ---------------------------------------------------------------------------
# Tests: validate
# ---------------------------------------------------------------------------

class TestValidate:

    def test_check_python_version_passes(self):
        """validate.py should always pass on Python 3.8+."""
        import validate
        validate.results.clear()
        validate.check_python_version()
        statuses = [s for _, s, _ in validate.results]
        assert "FAIL" not in statuses

    def test_check_filesystem_passes(self, tmp_path):
        import validate
        validate.results.clear()
        validate.check_filesystem(str(tmp_path / "test"))
        statuses = [s for _, s, _ in validate.results]
        assert "FAIL" not in statuses

    def test_check_model_file_fail_missing(self, tmp_path):
        import validate
        validate.results.clear()
        validate.check_model_file(str(tmp_path / "missing.tflite"), ".tflite")
        statuses = [s for _, s, _ in validate.results]
        assert "FAIL" in statuses

    def test_check_model_file_pass(self, tmp_path):
        import validate
        model_path = tmp_path / "pilot.tflite"
        model_path.write_bytes(b"\x00" * 2048)  # 2 KB — above 0.001 MB threshold
        validate.results.clear()
        validate.check_model_file(str(model_path), ".tflite")
        statuses = [s for _, s, _ in validate.results]
        assert "FAIL" not in statuses

    def test_check_model_file_wrong_extension(self, tmp_path):
        import validate
        model_path = tmp_path / "pilot.h5"
        model_path.write_bytes(b"\x00" * 1024)
        validate.results.clear()
        validate.check_model_file(str(model_path), ".tflite")
        statuses = [s for _, s, _ in validate.results]
        assert "WARN" in statuses

    def test_validate_summary_returns_int(self):
        import validate
        validate.results.clear()
        validate.record("test", validate.PASS, "ok")
        rc = validate.print_summary()
        assert rc == 0

    def test_validate_summary_returns_1_on_warnings(self):
        import validate
        validate.results.clear()
        validate.record("test", validate.WARN, "something")
        rc = validate.print_summary()
        assert rc == 1

    def test_validate_summary_returns_2_on_failures(self):
        import validate
        validate.results.clear()
        validate.record("test", validate.FAIL, "something bad")
        rc = validate.print_summary()
        assert rc == 2


# ---------------------------------------------------------------------------
# Integration: end-to-end
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_full_tub_pipeline(self, tmp_path):
        """Write → read → delete cycle should complete without errors."""
        from benchmark_tub import (
            benchmark_write_fs, benchmark_read_fs, benchmark_delete_fs
        )
        tub = str(tmp_path / "e2e_tub")
        w = benchmark_write_fs(tub, 50, False)
        r = benchmark_read_fs(tub)
        d = benchmark_delete_fs(tub, 10)

        assert w.errors == 0
        assert r.errors == 0
        assert d.errors == 0
        assert r.num_records == w.num_records

    def test_system_then_pipeline(self, tmp_results):
        from benchmark_system import run_system_benchmark
        from benchmark_pipeline import run_pipeline_benchmark

        sys_result = run_system_benchmark(1.0, 0.5, "e2e", False)
        pipe_result = run_pipeline_benchmark(
            mode="synthetic", num_runs=50, warmup=5, target_hz=30
        )

        assert len(sys_result.snapshots) > 0
        assert pipe_result["sustained_fps"] > 0

    @pytest.mark.slow
    def test_concurrent_tub_high_load(self, tmp_path):
        """300 records at 30 Hz — simulates a real driving session."""
        from benchmark_tub import benchmark_concurrent
        result = benchmark_concurrent(
            str(tmp_path / "conc"), num_records=300, hz=30.0
        )
        assert result["actual_hz"] > 15  # at least half speed on slow CI
        assert result["write_stats"]["mean_ms"] < 100  # < 100ms per write


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
