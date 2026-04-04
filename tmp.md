Mode: decode-only (`T=1`), `gdn_v3` now uses CTA shape `(32, 1, 1)` with a batch-aware V tile: `BV=2` for `B==1`, `BV=8` otherwise.

# Current v3 optimization plan

- [x] Step 1: Hoist `q` load earlier so it overlaps with `old_v` reduction/update work.
- [x] Step 2: Hoist scalar `v` load earlier to shorten the lane-0 broadcast path.
- [x] Step 3: Rewrite the update as `decay once + dv * k` to reduce dependent ops.
- [x] Step 4: Replace gate math with the archive fast-math form, keeping current indexing semantics.
- [x] Step 5: Stage block-uniform scalars once per block in shared memory.
- [x] Step 6: Remove shared `q/k` staging and use direct per-warp `.cs` register loads instead.
- [x] Step 7: Remove shared `gate/beta` staging and use warp-local compute plus shuffle broadcast.
- [x] Step 8: Switch `gdn_v3` to a 1D block shape (`32` threads, one warp per block).
- [x] Step 9: Add a 2-row V tile per warp in `gdn_v3`.
- [x] Step 10: Expand the V tile in `gdn_v3` from 2 rows per warp to 8 rows per warp.
- [x] Step 11: Converge `gdn_v3` toward archive structure: fixed contest-shape constants, `(tile, B*HV)` grid, `uint2/uint4` vector loads, inline-PTX paired reduction, and no V-tail checks.
- [ ] Step 12: Trim `gdn_v3` live state: pack V tile as bf16 pairs and write outputs directly from owning lanes instead of keeping `out_vals[8]`. Reverted: `70 -> 69` regs but no local timing improvement.
- [ ] Step 13: Change only single-use tile loads (`state`, `v`) to streaming `.cs` PTX while leaving reusable `q/k` on normal cached vector loads. Reverted: no local timing improvement.
- [ ] Step 14: Compute warp-uniform `gate/beta` once in lane 0 and broadcast, instead of repeating the SFU path on all 32 lanes. Reverted: no local timing improvement.
- [x] Step 15: Test a smaller warp row tile (`BV=4`) to increase CTA count for B200 workload-0 underfill (`B=1`).
- [x] Step 16: Push the same underfill hypothesis further with `BV=2` after `BV=4` improved Modal workload-0 from `3.39 us` to `2.66 us`.
- [x] Step 17: Turn the row-tile search into a dispatcher: `BV=2` for `B==1` where B200 is underfilled, `BV=8` otherwise for better row reuse on larger batches.
- [x] Step 22: Refine the dispatcher for moderate batches: `B<=4 -> BV=4` while keeping `B==1 -> BV=2` and larger batches on `BV=8`. Kept: Modal workload `10` (`B=4`) improved from the old `~3.10 us` band to `2.98 us`, while workload `0` stayed flat locally.
- [ ] Step 23: Extend the medium bucket to `B<=8 -> BV=4`. Reverted: local workload `18` (`B=8`) regressed from `12.29 us` to `13.31 us`, so it did not earn a Modal run.
- [ ] Step 24: In the `BV=2` hot path, load `v` once in lane 0 and broadcast with shuffles instead of redundantly loading the same 2 bf16 scalars in all 32 lanes. Reverted: Modal workload `0` regressed from `2.53 us` to `2.56 us`.
- [x] Step 25: Route the `q/k` vector loads through `__ldg` so the hot path can use the read-only cache for data reused across all `i_v` tiles of a head. Kept: Modal workload `0` improved from `2.53 us` to `2.51 us`.
- [ ] Step 26: Apply the same `__ldg` read-only hint to scalar gate inputs (`A_log`, `dt_bias`, `a`, `b`). Reverted: Modal workload `0` returned to `2.53 us`.
- [ ] Step 27: Apply the same `__ldg` read-only hint to `v` tile loads. Reverted: local workload `0` stayed at `5.12 us`, but Modal workload `0` regressed badly from `2.51 us` to `2.67 us`.
- [ ] Step 28: Stream the single-use `state` tile loads with `ld.global.cs.v4.f32` while leaving reusable `q/k` on `__ldg`. Reverted: correctness spot-check passed and local workload `0` stayed at `5.12 us`, but Modal workload `0` regressed from `2.51 us` to `2.56 us`.
- [ ] Step 29: Remove dead scalar launch parameters from `gdn_v3` to shrink per-block parameter traffic on the fixed-shape hot path. Reverted: correctness spot-check passed and local workload `0` stayed at `5.12 us`, but Modal workload `0` regressed from `2.51 us` to `2.59 us`.
- [ ] Step 30: Replace `__stcs` with plain write-back `float4` stores for `new_state`. Reverted: correctness spot-check passed and local workload `0` stayed at `5.12 us`, but Modal workload `0` regressed from `2.51 us` to `2.56 us`.
- [ ] Step 31: Move the single-use `state` tile off the read-only path and use plain `float4` global loads while keeping `q/k` on `__ldg`. Reverted: correctness spot-check passed and local workload `0` stayed at `5.12 us`, but Modal workload `0` regressed from `2.51 us` to `2.58 us`.
- [ ] Step 32: Try a dedicated `B==1`, `BV=1` one-warp hot kernel to turn the B200 underfill diagnosis into more CTAs with lower per-block work. Reverted: local workload `0` regressed badly from `5.12 us` to `8.22 us`.
- [ ] Step 33: Replace the custom butterfly pair-reduction with a plain `__shfl_down_sync` reduction sequence. Reverted on correctness: local workload `0` / `10` spot-checks jumped to `out_max_abs≈1e-2`, so the all-reduce semantics were required by the current `gdn_v3` writeout pattern.
- [ ] Step 18: Test `BV=1` for the `B==1` path to increase CTA count further on B200. Reverted: Modal workload-0 regressed from `2.53 us` to `2.69 us`.
- [ ] Step 19: Revisit warp-uniform `gate/beta` broadcast for the active dispatcher as a B200-specific instruction-count reduction. Reverted: Modal workload-0 regressed from `2.53 us` to `2.72 us`.
- [ ] Step 20: Hand-specialize the `B==1` hot path into a dedicated `BV=2` kernel to remove generic template/live-state overhead. Reverted: Modal workload-0 regressed from `2.53 us` to `2.66 us`.
- [ ] Step 21: Share `q/k` across the two `v`-heads that map to the same `qk` head in the `B==1` path using a 2-warp block. Reverted before Modal: local workload-0 regressed from `5.12 us` to `6.10 us`.

# Notes

- Remaining math changes were batched into `gdn_v3` per user request.
- Do not change the CTA shape while working through this list.
- Current active hypothesis: the archive-style structure should reduce instruction count and register pressure relative to the previous generic 8-row tile.
- Current active hypothesis: remaining gap is not simple register count; next changes should target load/cache behavior or B200-specific scheduling.
- Current active hypothesis: the remaining gap is structural or architecture-specific, not a simple cache-hint issue on Ampere.
- Current active hypothesis: the best global policy is batch-bucketed rather than a single fixed tile size, because underfill dominates only when `B` is very small and moderate batches can still benefit from smaller row tiles.
- Current active hypothesis: the remaining gap on workload-0 is not solved by "more CTAs at any cost" or by cutting scalar SFU work alone; the next step should come from a fresh B200 profile of the active `BV=2` path.
- Current active hypothesis: the remaining gap is dominated by tiny-grid underfill plus irreducible per-block state traffic; the remaining viable wins are likely to require a more invasive decomposition than the current warp-tile family.
- Tooling note: `scripts/profile_modal.py` had an outdated kernel regex (`gdn_decode_kernel`) and was skipping live `gdn_v3` launches; updated to `regex:^gdn_.*` so B200 Nsight captures the active kernel.
- Parked idea: hoist `gate_x`, `gate`, and `beta` earlier in `gdn_v3` so their SFU latency can overlap with the load/unpack path; risk is longer live ranges and higher register pressure.
- Parked idea: fuse `new_state` packing/stores into the update loop instead of doing them in the later output loop; likely experiment is a single fused update+output+state-store row-pair loop.
- [ ] Step 34: Hoist `gate_x`, `gate`, and `beta` to the top of `gdn_v3` before the state/qk/v load path. Reverted by user feedback: did not yield good results.
- Active experiment: fuse `new_state` packing/stores into the row-pair update loop so each pair is fully updated, reduced for output, and written back in one pass.
- Active experiment: remove the output scratch array and have the owning lanes write each row result directly after the pair reduction, with focus on trimming live state in the `BV=8` path.
