# Descriptor-load reduction recovery search

Triton 3.6.0 regressed the H100 `descriptor_load -> max-reduce`
microbenchmark, while the 2026-08-16 `main` commit `800558f674` no longer
shows the large slowdown.  The descriptor-load coalescing rule was not
reverted between those revisions.  The strongest recovery candidates are the
reduction-lowering stack merged immediately after 3.6.0:

| Label | Commit | Change |
| --- | --- | --- |
| `pre_9219` | `750952252b` | Parent of the new reduction lowering |
| `post_9219` | `483327f033` | LinearLayout-based reduction lowering; can avoid a shared-memory round trip |
| `post_9220` | `b5e3800aec` | Tree reductions for in-thread values |
| `post_9221` | `bb75a87080` | Cross-CTA reduction support |

The preferred workflow uses free GitHub-hosted Actions:

1. Pushing this branch triggers
   `.github/workflows/descriptor-reduction-recovery-wheels.yml`.  Four source
   candidates build in parallel, while official 3.6.0 and 3.7.0 wheels are
   downloaded from PyPI.
2. The workflow publishes the six wheels as checksummed assets in the
   `descriptor-reduction-recovery-v1` pre-release.
3. On H100, run
   `bootstrap_and_run_h100_descriptor_reduction_recovery.sh`.  It verifies all
   release checksums before invoking the no-build benchmark runner.
4. Stop the H100 as soon as the script prints `STOP THE H100 POD NOW`.

The CPU Pod workflow remains available as a fallback and has the same hard
cost boundary:

1. Run `build_descriptor_reduction_recovery_wheels.sh` on a CPU-only pod with
   the persistent network volume mounted at `/workspace`.  It downloads the
   official 3.6.0 and 3.7.0 wheels, builds the four source candidates, pins the
   benchmark by commit and SHA-256, and creates a checksummed manifest.
2. Do not deploy an H100 until the CPU script prints `CPU build complete`.
3. Mount the same network volume on an H100 and run
   `run_h100_descriptor_reduction_recovery.sh`.  Its preflight fails before
   benchmarking if any artifact is missing or corrupt.  It only installs
   prebuilt wheels and benchmarks them; it never builds Triton source.
4. Stop the H100 as soon as the script prints `STOP THE H100 POD NOW`.

The H100 matrix contains three shapes and two dtypes for six compiler points.
It is sufficient to locate the recovery interval while keeping the paid run
short.  The generated archive contains JSONL timings, compiler dumps, exact
commits, checksums, environment information, and the comparison table.
