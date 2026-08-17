#!/usr/bin/env bash
set -euo pipefail

# Build all source candidates on a CPU-only pod.  The resulting wheels are
# stored on the persistent /workspace volume so the H100 is used only for the
# short benchmark run.

PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
MAX_JOBS="${MAX_JOBS:-8}"
WORKSPACE="${WORKSPACE:-/workspace}"
SOURCE_DIR="${WORKSPACE}/triton-recovery-source"
WHEEL_ROOT="${WORKSPACE}/triton-recovery-wheels"
TRITON_HOME="${WORKSPACE}/triton-recovery-home"
MANIFEST="${WHEEL_ROOT}/manifest.tsv"
BENCH_DIR="${WORKSPACE}/triton-recovery-benchmark"
BENCH="${BENCH_DIR}/bench_desc_load_reduce.py"

TRITON_REPO="https://github.com/triton-lang/triton.git"
REPRO_COMMIT="b37f9c721c7ee7a0aa4e03691f6e8b4836b2cefe"
BENCH_SHA256="be0867a4937ecda465067c75a55b01c045afc1c23a0340428a383691f0fbe4e3"
BENCH_URL="https://raw.githubusercontent.com/aserputov/triton/${REPRO_COMMIT}/python/test/microbenchmark/tma_descriptor_load_reduce/bench_desc_load_reduce.py"

OFFICIAL_LABELS=(official_3_6_0 official_3_7_0)
OFFICIAL_VERSIONS=(3.6.0 3.7.0)
OFFICIAL_COMMITS=(
  7c56a5e40f7fd928dfd5c72902d5def0097db73a
  5f3f125e8f63c24613f1f73b937442864f263f94
)

# These four points bracket the reduction-lowering stack merged after 3.6.0.
SOURCE_LABELS=(pre_9219 post_9219 post_9220 post_9221)
SOURCE_COMMITS=(
  750952252ba7cbbe6cb64af13e00d356f7026a8c
  483327f0336aac7feb02a0eed45c2a6e2207ca00
  b5e3800aec693eed012da1574c0e95edb20655eb
  bb75a870803727d80bb1900a59b26500f18fec16
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

require_single_wheel() {
  local directory="$1"
  local wheel
  wheel="$(single_wheel "${directory}")" ||
    die "expected exactly one Triton wheel in ${directory}"
  printf '%s\n' "${wheel}"
}

mkdir -p "${WHEEL_ROOT}" "${BENCH_DIR}" "${TRITON_HOME}"

curl -fsSL "${BENCH_URL}" -o "${BENCH}"
printf '%s  %s\n' "${BENCH_SHA256}" "${BENCH}" | sha256sum -c -

if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
  git clone "${TRITON_REPO}" "${SOURCE_DIR}"
fi

git -C "${SOURCE_DIR}" remote set-url origin "${TRITON_REPO}"
git -C "${SOURCE_DIR}" fetch --tags origin

export MAX_JOBS
export TRITON_BUILD_PROTON=OFF
export TRITON_HOME

# Download release wheels once.  These establish whether the recovery is
# already present in 3.7.0 without spending CPU time rebuilding releases.
for i in "${!OFFICIAL_LABELS[@]}"; do
  label="${OFFICIAL_LABELS[$i]}"
  version="${OFFICIAL_VERSIONS[$i]}"
  output_dir="${WHEEL_ROOT}/${label}"
  mkdir -p "${output_dir}"
  if ! single_wheel "${output_dir}" >/dev/null 2>&1; then
    "${PYTHON_BIN}" -m pip download \
      --no-deps \
      --only-binary=:all: \
      --dest "${output_dir}" \
      "triton==${version}"
  fi
  require_single_wheel "${output_dir}" >/dev/null
done

# Build adjacent source commits in one checkout so CMake/Ninja can reuse the
# previous build.  No GPU is required for this phase.
for i in "${!SOURCE_LABELS[@]}"; do
  label="${SOURCE_LABELS[$i]}"
  commit="${SOURCE_COMMITS[$i]}"
  output_dir="${WHEEL_ROOT}/${label}"
  mkdir -p "${output_dir}"

  if single_wheel "${output_dir}" >/dev/null 2>&1; then
    [[ -f "${output_dir}/source.commit" ]] ||
      die "${label} has a wheel but no source.commit provenance stamp"
    recorded_commit="$(<"${output_dir}/source.commit")"
    [[ "${recorded_commit}" == "${commit}" ]] ||
      die "${label} wheel was built from ${recorded_commit}, expected ${commit}"
    echo "Reusing ${label}: $(single_wheel "${output_dir}")"
    continue
  fi

  git -C "${SOURCE_DIR}" diff --quiet ||
    die "tracked files in ${SOURCE_DIR} are modified"
  git -C "${SOURCE_DIR}" checkout --detach "${commit}"
  actual_commit="$(git -C "${SOURCE_DIR}" rev-parse HEAD)"
  [[ "${actual_commit}" == "${commit}" ]] || die "checkout mismatch for ${label}"

  "${PYTHON_BIN}" -m pip install -r "${SOURCE_DIR}/python/requirements.txt"
  (
    cd "${SOURCE_DIR}"
    "${PYTHON_BIN}" -m pip wheel \
      --no-build-isolation \
      --no-cache-dir \
      --no-deps \
      --wheel-dir "${output_dir}" \
      .
  )
  require_single_wheel "${output_dir}" >/dev/null
  printf '%s\n' "${commit}" >"${output_dir}/source.commit"
done

manifest_tmp="${MANIFEST}.tmp"
printf 'label\tcommit\twheel\tsha256\n' >"${manifest_tmp}"

for i in "${!OFFICIAL_LABELS[@]}"; do
  label="${OFFICIAL_LABELS[$i]}"
  commit="${OFFICIAL_COMMITS[$i]}"
  wheel="$(require_single_wheel "${WHEEL_ROOT}/${label}")"
  checksum="$(sha256sum "${wheel}" | awk '{print $1}')"
  printf '%s\t%s\t%s\t%s\n' \
    "${label}" "${commit}" "${wheel}" "${checksum}" >>"${manifest_tmp}"
done

for i in "${!SOURCE_LABELS[@]}"; do
  label="${SOURCE_LABELS[$i]}"
  commit="${SOURCE_COMMITS[$i]}"
  recorded_commit="$(<"${WHEEL_ROOT}/${label}/source.commit")"
  [[ "${recorded_commit}" == "${commit}" ]] ||
    die "provenance mismatch while writing manifest for ${label}"
  wheel="$(require_single_wheel "${WHEEL_ROOT}/${label}")"
  checksum="$(sha256sum "${wheel}" | awk '{print $1}')"
  printf '%s\t%s\t%s\t%s\n' \
    "${label}" "${commit}" "${wheel}" "${checksum}" >>"${manifest_tmp}"
done

mv "${manifest_tmp}" "${MANIFEST}"

echo
echo "CPU build complete. No H100 was used."
echo "Manifest: ${MANIFEST}"
echo "Benchmark: ${BENCH}"
echo "The H100 runner will refuse to start unless every wheel and checksum is present."
