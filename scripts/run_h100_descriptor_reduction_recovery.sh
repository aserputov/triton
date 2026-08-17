#!/usr/bin/env bash
set -euo pipefail

# Benchmark prebuilt wheels only.  Do not add source builds to this script:
# every minute here is billed at the H100 rate.

PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
WORKSPACE="${WORKSPACE:-/workspace}"
WHEEL_ROOT="${WHEEL_ROOT:-${WORKSPACE}/triton-recovery-wheels}"
MANIFEST="${MANIFEST:-${WHEEL_ROOT}/manifest.tsv}"
BENCH="${BENCH:-${WORKSPACE}/triton-recovery-benchmark/bench_desc_load_reduce.py}"
BENCH_SHA256="be0867a4937ecda465067c75a55b01c045afc1c23a0340428a383691f0fbe4e3"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="${WORKSPACE}/triton-recovery-results-${RUN_ID}"
TRITON_HOME="${WORKSPACE}/triton-recovery-h100-home"

EXPECTED_LABELS=(
  official_3_6_0
  pre_9219
  post_9219
  post_9220
  post_9221
  official_3_7_0
)

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -f "${MANIFEST}" ]] || die "missing ${MANIFEST}; run the CPU build first"
[[ -f "${BENCH}" ]] || die "missing ${BENCH}; run the CPU build first"
printf '%s  %s\n' "${BENCH_SHA256}" "${BENCH}" | sha256sum -c -

declare -A COMMITS
declare -A WHEELS
declare -A CHECKSUMS
while IFS=$'\t' read -r label commit wheel checksum; do
  [[ "${label}" == "label" ]] && continue
  [[ -n "${label}" ]] || continue
  COMMITS["${label}"]="${commit}"
  WHEELS["${label}"]="${wheel}"
  CHECKSUMS["${label}"]="${checksum}"
done <"${MANIFEST}"

# Finish every non-GPU preflight before checking the GPU or starting timings.
for label in "${EXPECTED_LABELS[@]}"; do
  [[ -n "${WHEELS[${label}]:-}" ]] || die "manifest has no ${label}"
  [[ -f "${WHEELS[${label}]}" ]] || die "missing wheel for ${label}"
  printf '%s  %s\n' "${CHECKSUMS[${label}]}" "${WHEELS[${label}]}" |
    sha256sum -c - >/dev/null
done

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
[[ "${GPU_NAME}" == *H100* ]] || die "expected H100, found ${GPU_NAME}"

mkdir -p "${RESULTS_DIR}" "${TRITON_HOME}"
nvidia-smi | tee "${RESULTS_DIR}/nvidia-smi.txt"
cp "${MANIFEST}" "${RESULTS_DIR}/manifest.tsv"

export TRITON_HOME
export TRITON_BUILD_PROTON=OFF

echo
echo "All prebuilt wheels passed checksum verification."
echo "Running six wheel variants; there is no Triton source build in this script."

for label in "${EXPECTED_LABELS[@]}"; do
  wheel="${WHEELS[${label}]}"
  commit="${COMMITS[${label}]}"
  echo
  echo "=== ${label} ${commit} ==="

  "${PYTHON_BIN}" -m pip install \
    --no-index \
    --no-deps \
    --force-reinstall \
    "${wheel}"

  git_commit_file="${RESULTS_DIR}/${label}.commit"
  printf '%s\n' "${commit}" >"${git_commit_file}"

  TRITON_CACHE_DIR="${WORKSPACE}/triton-recovery-cache-${label}" \
  TRITON_DUMP_DIR="${RESULTS_DIR}/compiler-dump-${label}" \
  TRITON_KERNEL_DUMP=1 \
    "${PYTHON_BIN}" "${BENCH}" \
      --m 8192 \
      --n 8192 \
      --shapes "16x128,32x128,64x128" \
      --dtypes "bf16,fp32" \
      --warps 4 \
      --warmup 10 \
      --repeat 9 \
      --check |
    tee "${RESULTS_DIR}/${label}.jsonl"
done

"${PYTHON_BIN}" - "${RESULTS_DIR}" "${EXPECTED_LABELS[@]}" <<'PY' |
  tee "${RESULTS_DIR}/summary.txt"
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
labels = sys.argv[2:]
rows = {}

for label in labels:
    records = []
    for line in (root / f"{label}.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record.get("status") == "ok":
            records.append(record)
    if len(records) != 6:
        raise SystemExit(f"{label}: expected 6 successful records, found {len(records)}")
    rows[label] = {
        (r["BLOCK_M"], r["BLOCK_N"], r["dtype"]): r["median_ms"]
        for r in records
    }

baseline = rows["official_3_6_0"]
print("shape     dtype  " + "  ".join(f"{label:>16}" for label in labels))
for key in sorted(baseline):
    values = []
    for label in labels:
        timing = rows[label][key]
        ratio = timing / baseline[key]
        values.append(f"{timing:7.5f} ({ratio:5.2f}x)")
    shape = f"{key[0]}x{key[1]}"
    print(f"{shape:<9} {key[2]:<5}  " + "  ".join(values))

key = (32, 128, "bf16")
print("\n32x128 BF16 recovery scan relative to official Triton 3.6.0:")
for label in labels:
    timing = rows[label][key]
    print(f"  {label:<16} {timing:.6f} ms  {timing / baseline[key]:.3f}x")
PY

archive="${WORKSPACE}/$(basename "${RESULTS_DIR}").tgz"
tar -C "${RESULTS_DIR}" -czf "${archive}" .

echo
echo "Recovery benchmark complete. STOP THE H100 POD NOW."
echo "Summary: ${RESULTS_DIR}/summary.txt"
echo "Archive: ${archive}"
