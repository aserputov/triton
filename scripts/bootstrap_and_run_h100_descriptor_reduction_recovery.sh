#!/usr/bin/env bash
set -euo pipefail

# Download the GitHub Actions-built candidate wheels, verify every checksum,
# create the manifest expected by the H100 runner, and start the benchmark.

WORKSPACE="${WORKSPACE:-/workspace}"
RELEASE_REPO="aserputov/triton"
RELEASE_TAG="descriptor-reduction-recovery-v1"
RELEASE_BASE="https://github.com/${RELEASE_REPO}/releases/download/${RELEASE_TAG}"
ARTIFACT_ROOT="${WORKSPACE}/${RELEASE_TAG}"
DOWNLOAD_DIR="${ARTIFACT_ROOT}/downloads"
WHEEL_ROOT="${ARTIFACT_ROOT}/wheels"
MANIFEST="${WHEEL_ROOT}/manifest.tsv"
BENCH="${ARTIFACT_ROOT}/bench_desc_load_reduce.py"
RUNNER="${ARTIFACT_ROOT}/run_h100_descriptor_reduction_recovery.sh"

LABELS=(
  official_3_6_0
  pre_9219
  post_9219
  post_9220
  post_9221
  official_3_7_0
)
COMMITS=(
  7c56a5e40f7fd928dfd5c72902d5def0097db73a
  750952252ba7cbbe6cb64af13e00d356f7026a8c
  483327f0336aac7feb02a0eed45c2a6e2207ca00
  b5e3800aec693eed012da1574c0e95edb20655eb
  bb75a870803727d80bb1900a59b26500f18fec16
  5f3f125e8f63c24613f1f73b937442864f263f94
)

die() {
  echo "ERROR: $*" >&2
  exit 1
}

single_wheel() {
  local directory="$1"
  local wheels=()
  shopt -s nullglob
  wheels=("${directory}"/triton-*.whl)
  shopt -u nullglob
  [[ "${#wheels[@]}" -eq 1 ]] || return 1
  printf '%s\n' "${wheels[0]}"
}

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
[[ "${GPU_NAME}" == *H100* ]] || die "expected H100, found ${GPU_NAME}"

mkdir -p "${DOWNLOAD_DIR}" "${WHEEL_ROOT}"

curl -fsSL "${RELEASE_BASE}/SHA256SUMS" -o "${DOWNLOAD_DIR}/SHA256SUMS"
for label in "${LABELS[@]}"; do
  curl -fsSL "${RELEASE_BASE}/${label}.tgz" -o "${DOWNLOAD_DIR}/${label}.tgz"
done
curl -fsSL "${RELEASE_BASE}/bench_desc_load_reduce.py" \
  -o "${DOWNLOAD_DIR}/bench_desc_load_reduce.py"
curl -fsSL "${RELEASE_BASE}/run_h100_descriptor_reduction_recovery.sh" \
  -o "${DOWNLOAD_DIR}/run_h100_descriptor_reduction_recovery.sh"

(
  cd "${DOWNLOAD_DIR}"
  sha256sum -c SHA256SUMS
)

cp "${DOWNLOAD_DIR}/bench_desc_load_reduce.py" "${BENCH}"
cp "${DOWNLOAD_DIR}/run_h100_descriptor_reduction_recovery.sh" "${RUNNER}"
chmod +x "${RUNNER}"

manifest_tmp="${MANIFEST}.tmp"
printf 'label\tcommit\twheel\tsha256\n' >"${manifest_tmp}"
for i in "${!LABELS[@]}"; do
  label="${LABELS[$i]}"
  commit="${COMMITS[$i]}"
  output_dir="${WHEEL_ROOT}/${label}"
  mkdir -p "${output_dir}"
  tar -C "${output_dir}" -xzf "${DOWNLOAD_DIR}/${label}.tgz"
  wheel="$(single_wheel "${output_dir}")" ||
    die "expected exactly one wheel for ${label}"
  checksum="$(sha256sum "${wheel}" | awk '{print $1}')"
  printf '%s\t%s\t%s\t%s\n' \
    "${label}" "${commit}" "${wheel}" "${checksum}" >>"${manifest_tmp}"
done
mv "${manifest_tmp}" "${MANIFEST}"

echo "Release artifacts verified. Starting the no-build H100 benchmark."
WHEEL_ROOT="${WHEEL_ROOT}" MANIFEST="${MANIFEST}" BENCH="${BENCH}" \
  bash "${RUNNER}"
