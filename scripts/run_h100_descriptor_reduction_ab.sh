#!/usr/bin/env bash
set -euo pipefail

PATCH_BRANCH="codex/tma-descriptor-reduction-fix"
PATCH_COMMIT="ff9f9d2bc013ca8e17d20215b26d6f312322c8c5"
BASE_COMMIT="800558f67437035e682fdf49fae229f59294a75b"
PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
MAX_JOBS="${MAX_JOBS:-8}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
SOURCE_DIR="/workspace/triton-descriptor-ab-${RUN_ID}"
RESULTS_DIR="/workspace/h100-ab-results-${RUN_ID}"
TRITON_HOME="/workspace/triton-ab-home"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
if [[ "${GPU_NAME}" != *H100* ]]; then
  echo "Expected an H100, found: ${GPU_NAME}" >&2
  exit 1
fi

mkdir -p "${RESULTS_DIR}"
nvidia-smi | tee "${RESULTS_DIR}/nvidia-smi.txt"

git clone --single-branch --branch "${PATCH_BRANCH}" \
  https://github.com/aserputov/triton.git "${SOURCE_DIR}"
cd "${SOURCE_DIR}"

if [[ "$(git rev-parse HEAD)" != "${PATCH_COMMIT}" ]]; then
  echo "Patch branch no longer points at ${PATCH_COMMIT}" >&2
  exit 1
fi
if [[ "$(git rev-parse HEAD^)" != "${BASE_COMMIT}" ]]; then
  echo "Unexpected baseline commit" >&2
  exit 1
fi

BENCH="/workspace/triton_tma_reduction_bench/bench_desc_load_reduce.py"
if [[ ! -f "${BENCH}" ]]; then
  REPRO_DIR="/workspace/triton-reproducer-${RUN_ID}"
  git clone --depth 1 --single-branch \
    --branch codex/tma-descriptor-reduction-reproducer \
    https://github.com/aserputov/triton.git "${REPRO_DIR}"
  BENCH="${REPRO_DIR}/python/test/microbenchmark/tma_descriptor_load_reduce/bench_desc_load_reduce.py"
fi

export TRITON_BUILD_PROTON=OFF
export TRITON_HOME
export MAX_JOBS

"${PYTHON_BIN}" -m pip install -r python/requirements.txt

run_benchmark() {
  local label="$1"
  local cache_dir="/workspace/triton-cache-${RUN_ID}-${label}"
  local dump_dir="${RESULTS_DIR}/compiler-dump-${label}"
  git rev-parse HEAD > "${RESULTS_DIR}/${label}.commit"
  TRITON_CACHE_DIR="${cache_dir}" \
  TRITON_DUMP_DIR="${dump_dir}" \
  TRITON_KERNEL_DUMP=1 \
    "${PYTHON_BIN}" "${BENCH}" \
      --m 8192 \
      --n 8192 \
      --shapes "16x128,32x128,64x128,32x64,32x256" \
      --dtypes "bf16,fp16,fp32" \
      --warps "4" \
      --warmup 10 \
      --repeat 9 \
      --check | tee "${RESULTS_DIR}/${label}.jsonl"
}

# A: exact parent of the patch.
git checkout --detach "${BASE_COMMIT}"
"${PYTHON_BIN}" -m pip install -e . --no-build-isolation
make PYTHON="${PYTHON_BIN}"
run_benchmark baseline_a1

# B: the one-commit layout guard. The existing build is reused, so this is an
# incremental compiler rebuild rather than a second full build.
git checkout --detach "${PATCH_COMMIT}"
make PYTHON="${PYTHON_BIN}"
BUILD_DIR="$(PYTHONPATH="${SOURCE_DIR}/python" "${PYTHON_BIN}" -c \
  'from build_helpers import get_cmake_dir; print(get_cmake_dir())')"
(cd "${BUILD_DIR}" && lit -v "${SOURCE_DIR}/test/TritonGPU/coalesce.mlir") \
  | tee "${RESULTS_DIR}/patched-lit.txt"
run_benchmark patched_b

# A again: rebuild the parent and repeat to detect clock/thermal drift.
git checkout --detach "${BASE_COMMIT}"
make PYTHON="${PYTHON_BIN}"
run_benchmark baseline_a2

"${PYTHON_BIN}" - "${RESULTS_DIR}" <<'PY' | tee "${RESULTS_DIR}/summary.txt"
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
labels = ("baseline_a1", "patched_b", "baseline_a2")
rows = {}
for label in labels:
    parsed = []
    for line in (root / f"{label}.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record.get("status") == "ok":
            parsed.append(record)
    rows[label] = {
        (r["BLOCK_M"], r["BLOCK_N"], r["dtype"], r["num_warps"]): r["median_ms"]
        for r in parsed
    }

print("BLOCK_MxBLOCK_N dtype  A1_ms      B_ms       A2_ms      B/mean(A)")
for key in sorted(rows["patched_b"]):
    a1 = rows["baseline_a1"][key]
    b = rows["patched_b"][key]
    a2 = rows["baseline_a2"][key]
    mean_a = statistics.mean((a1, a2))
    print(f"{key[0]}x{key[1]:<4} {key[2]:<5} {a1:10.6f} {b:10.6f} {a2:10.6f} {b / mean_a:10.3f}x")
PY

ARCHIVE="/workspace/$(basename "${RESULTS_DIR}").tgz"
tar -C "${RESULTS_DIR}" -czf "${ARCHIVE}" .
echo "H100 A/B complete"
echo "Results directory: ${RESULTS_DIR}"
echo "Download archive: ${ARCHIVE}"
