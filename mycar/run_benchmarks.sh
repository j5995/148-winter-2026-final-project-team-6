#!/usr/bin/env bash
# run_benchmarks.sh
# =================
# Orchestrates the full benchmark suite on an RPi 5 with or without AI HAT.
# Produces a timestamped results directory and a combined JSON summary.
#
# Usage
# -----
#   chmod +x run_benchmarks.sh
#   ./run_benchmarks.sh                           # synthetic (no models needed)
#   ./run_benchmarks.sh --tflite models/p.tflite  # CPU-only TFLite
#   ./run_benchmarks.sh --tflite models/p.tflite \
#                       --hef    models/p.hef     # full compare (requires AI HAT)

set -euo pipefail

# ────────────────────────────────────────────────────────────────────────────
# Defaults
# ────────────────────────────────────────────────────────────────────────────
TFLITE_MODEL=""
HEF_MODEL=""
RUNS=300
WARMUP=20
TUB_RECORDS=1000
SYSTEM_DURATION=30
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="results/${TIMESTAMP}"
PYTHON="${PYTHON:-python3}"
STRICT_VALIDATE=0

# ────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ────────────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tflite)  TFLITE_MODEL="$2"; shift 2 ;;
    --hef)     HEF_MODEL="$2";    shift 2 ;;
    --runs)    RUNS="$2";         shift 2 ;;
    --warmup)  WARMUP="$2";       shift 2 ;;
    --strict)  STRICT_VALIDATE=1; shift   ;;
    --out)     OUTDIR="$2";       shift 2 ;;
    *)         echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ────────────────────────────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────────────────────────────
BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

log()  { echo -e "${BOLD}[$(date '+%H:%M:%S')]${RESET} $*"; }
ok()   { echo -e "${GREEN}  ✓ $*${RESET}"; }
warn() { echo -e "${YELLOW}  ⚠ $*${RESET}"; }
fail() { echo -e "${RED}  ✗ $*${RESET}"; }

section() {
  echo ""
  echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
  echo -e "${BOLD}  $*${RESET}"
  echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
}

run_step() {
  local label="$1"
  shift
  log "Running: $label"
  if "$@"; then
    ok "$label — done"
  else
    warn "$label — completed with non-zero exit (see log)"
  fi
}

# ────────────────────────────────────────────────────────────────────────────
# Setup
# ────────────────────────────────────────────────────────────────────────────
mkdir -p "${OUTDIR}"
LOG_FILE="${OUTDIR}/run.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo ""
echo -e "${BOLD}════════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  RPi 5 + AI HAT · Donkeycar Benchmark Suite${RESET}"
echo -e "${BOLD}  Timestamp : ${TIMESTAMP}${RESET}"
echo -e "${BOLD}  Output    : ${OUTDIR}${RESET}"
echo -e "${BOLD}  TFLite    : ${TFLITE_MODEL:-'(none)'}${RESET}"
echo -e "${BOLD}  HEF       : ${HEF_MODEL:-'(none)'}${RESET}"
echo -e "${BOLD}════════════════════════════════════════════════════════════${RESET}"

# Set CPU governor to performance (if sudo available)
if command -v sudo &>/dev/null && [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
  echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null 2>&1 \
    && ok "CPU governor set to performance" \
    || warn "Could not set CPU governor (sudo required)"
fi

# ────────────────────────────────────────────────────────────────────────────
# Step 1: Validation
# ────────────────────────────────────────────────────────────────────────────
section "Step 1/6 · Pre-flight Validation"

VALIDATE_ARGS=""
[[ -n "$TFLITE_MODEL" ]] && VALIDATE_ARGS+=" --tflite ${TFLITE_MODEL}"
[[ -n "$HEF_MODEL"    ]] && VALIDATE_ARGS+=" --hef ${HEF_MODEL}"
[[ $STRICT_VALIDATE -eq 1 ]] && VALIDATE_ARGS+=" --strict"

${PYTHON} validate.py ${VALIDATE_ARGS} \
  --output "${OUTDIR}/validation.json" \
  || { warn "Validation returned non-zero — continuing anyway (use --strict to abort)"; }

# ────────────────────────────────────────────────────────────────────────────
# Step 2: Run Tests
# ────────────────────────────────────────────────────────────────────────────
section "Step 2/6 · Unit + Integration Tests"

if command -v pytest &>/dev/null || ${PYTHON} -m pytest --version &>/dev/null 2>&1; then
  run_step "pytest test suite" \
    ${PYTHON} -m pytest test_benchmarks.py -v \
      --tb=short \
      -k "not slow" \
      --junit-xml="${OUTDIR}/test_results.xml" \
    || true   # Don't abort if tests fail on CI
else
  warn "pytest not found — skipping test suite. Install with: pip install pytest"
fi

# ────────────────────────────────────────────────────────────────────────────
# Step 3: Tub Benchmark
# ────────────────────────────────────────────────────────────────────────────
section "Step 3/6 · Tub I/O Benchmark"

run_step "Tub read/write/delete benchmark" \
  ${PYTHON} benchmark_tub.py \
    --records "${TUB_RECORDS}" \
    --deletes 100 \
    --hz 30 \
    --output "${OUTDIR}/tub_report.json"

# ────────────────────────────────────────────────────────────────────────────
# Step 4: System Baseline (idle + workload)
# ────────────────────────────────────────────────────────────────────────────
section "Step 4/6 · System Metrics (CPU / Memory / Temperature)"

# Idle baseline (no vehicle loop)
run_step "System idle baseline" \
  ${PYTHON} benchmark_system.py \
    --duration "${SYSTEM_DURATION}" \
    --interval 0.5 \
    --label "idle_baseline" \
    --no-workload \
    --output "${OUTDIR}/system_idle.json"

# Workload baseline (simulated vehicle loop)
run_step "System under vehicle-loop workload" \
  ${PYTHON} benchmark_system.py \
    --duration "${SYSTEM_DURATION}" \
    --interval 0.5 \
    --label "vehicle_loop_workload" \
    --output "${OUTDIR}/system_workload.json"

# ────────────────────────────────────────────────────────────────────────────
# Step 5: Inference Benchmark
# ────────────────────────────────────────────────────────────────────────────
section "Step 5/6 · Model Inference Benchmark"

if [[ -n "$TFLITE_MODEL" && -n "$HEF_MODEL" ]]; then
  log "Running COMPARE mode (CPU vs AI HAT)"
  run_step "Inference compare: CPU vs Hailo" \
    ${PYTHON} benchmark_inference.py \
      --mode compare \
      --tflite "${TFLITE_MODEL}" \
      --hef    "${HEF_MODEL}" \
      --runs   "${RUNS}" \
      --warmup "${WARMUP}" \
      --output "${OUTDIR}/inference_compare"

elif [[ -n "$TFLITE_MODEL" ]]; then
  log "Running CPU-only TFLite benchmark"
  run_step "Inference: CPU TFLite" \
    ${PYTHON} benchmark_inference.py \
      --mode  cpu \
      --model "${TFLITE_MODEL}" \
      --runs  "${RUNS}" \
      --warmup "${WARMUP}" \
      --output "${OUTDIR}/inference_cpu"

elif [[ -n "$HEF_MODEL" ]]; then
  log "Running AI HAT (Hailo) benchmark"
  run_step "Inference: Hailo AI HAT" \
    ${PYTHON} benchmark_inference.py \
      --mode  hailo \
      --model "${HEF_MODEL}" \
      --runs  "${RUNS}" \
      --warmup "${WARMUP}" \
      --output "${OUTDIR}/inference_hailo"

else
  warn "No model specified — skipping inference benchmark."
  warn "Use --tflite and/or --hef flags to provide models."
fi

# ────────────────────────────────────────────────────────────────────────────
# Step 6: Pipeline End-to-End Benchmark
# ────────────────────────────────────────────────────────────────────────────
section "Step 6/6 · End-to-End Pipeline Benchmark"

if [[ -n "$TFLITE_MODEL" ]]; then
  run_step "Pipeline (TFLite) · Hz sweep 10/20/30/40" \
    ${PYTHON} benchmark_pipeline.py \
      --mode tflite \
      --model "${TFLITE_MODEL}" \
      --runs  "${RUNS}" \
      --warmup "${WARMUP}" \
      --hz-sweep \
      --output "${OUTDIR}/pipeline_tflite.json"
fi

if [[ -n "$HEF_MODEL" ]]; then
  run_step "Pipeline (Hailo) · Hz sweep 10/20/30/40" \
    ${PYTHON} benchmark_pipeline.py \
      --mode hailo \
      --model "${HEF_MODEL}" \
      --runs  "${RUNS}" \
      --warmup "${WARMUP}" \
      --hz-sweep \
      --output "${OUTDIR}/pipeline_hailo.json"
fi

# Always run synthetic (needs no hardware)
run_step "Pipeline (synthetic) · Hz sweep 10/20/30/40" \
  ${PYTHON} benchmark_pipeline.py \
    --mode synthetic \
    --runs  100 \
    --warmup 10 \
    --hz-sweep \
    --output "${OUTDIR}/pipeline_synthetic.json"

# ────────────────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────────────────
section "Summary"

echo ""
log "All steps complete. Results written to: ${OUTDIR}/"
ls -lh "${OUTDIR}/"

# Write a manifest
python3 - <<EOF
import json, os, glob, time
files = sorted(glob.glob("${OUTDIR}/**", recursive=True))
manifest = {
    "timestamp": "${TIMESTAMP}",
    "tflite_model": "${TFLITE_MODEL}",
    "hef_model":    "${HEF_MODEL}",
    "runs":         ${RUNS},
    "files":        [f for f in files if os.path.isfile(f)],
}
with open("${OUTDIR}/manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print("  manifest.json written")
EOF

echo ""
ok "Benchmark suite complete. See ${OUTDIR}/ for all results."
echo ""
