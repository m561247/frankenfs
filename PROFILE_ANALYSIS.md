# Profile Analysis (bd-3ib.1)

## Metadata

- Profile date (UTC): `2026-02-13T08:38:17Z`
- Commit: `fe476693ab708709fbb7d83d3d430953785bb6b1`
- Kernel: `Linux 6.17.0-14-generic x86_64 GNU/Linux`
- CPU: `AMD Ryzen Threadripper PRO 5995WX 64-Cores` (128 logical CPUs)
- Toolchain:
  - `cargo 1.95.0-nightly (ce69df6f7 2026-02-12)`
  - `rustc 1.95.0-nightly (47611e160 2026-02-12)`
  - `flamegraph-flamegraph 0.6.10`
  - `perf version 6.17.9`

## Scope

Target bead: `bd-3ib.1` ("Profile read path and generate flamegraph").

Canonical target command:

```bash
ffs inspect conformance/golden/ext4_8mb_reference.ext4 --json
```

**Status (2026-02-27):** Non-contiguous ext4 journal extents are now fully supported.
The golden fixture (`ext4_8mb_reference.ext4`) has 3 journal segments and inspects
successfully in ~3.7 ms, producing:

```json
{"filesystem":"ext4","block_size":4096,"inodes_count":2048,"blocks_count":2048,"volume_name":"ffs-ref","free_blocks_total":851,"free_inodes_total":2033}
```

The original profiling run (2026-02-13) predated the multi-segment journal support
and used a workaround no-journal image instead:

```bash
cargo flamegraph --root -p ffs-cli --output profiles/flamegraph_cli_inspect.svg -- \
  inspect /tmp/ffs-profile-nojournal.fjnSmr.ext4 --json
```

Artifact:

- `profiles/flamegraph_cli_inspect.svg`

## Hotspots

Sample summary from `perf report --stdio -i perf.data`:

- Total samples: `18`
- Event: `cycles:P`
- Lost samples: `0`

Top symbols:

| Self % | Symbol | Shared Object | Interpretation |
|---:|---|---|---|
| 32.08% | `vma_interval_tree_remove` | kernel | ELF/object mapping lifecycle during process startup |
| 31.96% | `perf_iterate_ctx` | kernel | perf event/context overhead around exec/mmap |
| 30.98% | `srso_alias_return_thunk` | kernel | kernel return-thunk overhead in sampled startup path |
| 2.29% | `mas_preallocate` | kernel | mmap/VMA tree setup work |
| 2.11% | `_dl_map_object_deps` | `ld-linux-x86-64.so.2` | dynamic loader dependency mapping |

## Baseline Tie-In

From `baselines/baseline-20260213.md`:

- `ffs-cli inspect ext4_8mb_reference.ext4 --json` was skipped at baseline time due to the (now-resolved) non-contiguous journal extent limitation.
- Available baseline numbers were for parity/check-fixtures commands only (`~0.9–1.2 ms` range).

**Update (2026-02-27):** The inspect path now works on the canonical fixture. Inspect-path regression tracking can be added to future baselines.

## Opportunity Matrix

| Candidate | Impact | Confidence | Effort | Score (I*C/E) | Status |
|---|---|---|---|---:|---|
| ~~Implement ext4 non-contiguous journal extent support~~ | ~~High~~ | ~~High~~ | ~~Medium~~ | ~~3.0~~ | **DONE** — multi-segment journal replay implemented |
| Add a repeatable inspect profiling harness (looped workload) to raise sample count and expose Rust hot code | Medium | High | Low | 4.0 | Open — current profile has only 18 samples, dominated by loader/startup |
| Add FUSE read-path flamegraph (inspect fixture path now unblocked) | Medium | Medium | Medium | 1.0 | Open — no longer blocked by journal extent support |
| Optimize CLI cold-start overhead (link/load + startup path) after longer-run profile confirms bottleneck | Low-Medium | Low | Medium | 0.5 | Open — low-confidence, needs profiling harness first |

## Recommended Next Targets

1. ~~Unblock canonical inspect path by implementing support for non-contiguous ext4 journal extents.~~ **DONE.**
2. Add a deterministic profiling harness that runs inspect repeatedly in one invocation window (to collect meaningful Rust-symbol samples).
3. Re-run flamegraph against canonical fixture and update this document with post-fix hotspot data.
4. Add FUSE read-path flamegraph (now unblocked).

## Limitations

- The original (2026-02-13) run used a temporary no-journal ext4 image rather than the canonical golden ext4 fixture. The journal extent blocker is now resolved.
- The sample count (18) is too low for high-confidence micro-optimization decisions.
- A useful application-level hotspot map requires a longer-running inspect workload (or in-process repeated inspection).
