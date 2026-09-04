# HANDOFF — dcryolike

> ## ⚠ REVIEW FINDING (2026-09-02, second model pass) — READ FIRST
>
> **The synthetic noise model was unphysical, and every headline number was
> measured on an easier problem than real noise poses.**
>
> `generate()` added iid noise per polar grid point. No real-space noise field
> produces that: real noise is white in real space, and its polar transform is
> strongly *correlated* (measured 1.000 between adjacent inner-shell points,
> 0.50 mid-shell). Three independent checks agree:
> 1. On iid noise, *unweighted* CC beats CryoLike's weighted CC by +12 top-1 —
>    i.e. the "exact likelihood" was not the matched filter for that data.
> 2. On real-space white noise transformed exactly to the polar grid, the
>    weighted CC wins. **CryoLike's quadrature weighting is correct; the
>    generator was wrong.**
> 3. Corrected ceiling (physical noise, real-space per-pixel SNR). **Box size dominates:**
>
>    | SNR | D=48 top-1 | D=48 @500 | **D=96 top-1** | **D=96 @100** | **D=96 @500** | D=96 top-1 err |
>    |----|-----------|-----------|----------------|---------------|---------------|----------------|
>    | 0.30 | 12.5% | 88.5% | **82.5%** | 100% | 100% | 0.8° |
>    | 0.10 | 1.5% | 47.0% | **57.5%** | 100% | 100% | 1.3° |
>    | 0.05 | 0.0% | 19.5% | **33.0%** | **96.5%** | **98.5%** | 2.0° |
>    | 0.02 | 0.0% | 11.0% | 12.0% | 54.0% | 68.0% | 29.6° |
>
>    (D=96: 30,522 views @ 1.19°, tol 1.49° — a *stricter* tolerance than D=48's 2.98°.
>    In-mask SNR ≈ 1.1× whole-box for this phantom, so the conventions nearly coincide.)
>
>    **The estimator-vs-proposal gap survives physical noise at realistic box size:** at
>    SNR 0.05, D=96, top-1 33% vs recall@100 96.5%. Doubling k_max took top-1 at 0.05
>    from 0% to 33% — consistent with the ~k_max^2.7 information scaling measured earlier.
>    Note the coincidence: the ORIGINAL unphysical D=48 numbers (17.5% / 95.5%) landed near
>    the physical D=96 numbers because two errors partly cancelled (noise too easy, box too
>    small). That is not a vindication of the original; it is why it looked plausible.
>
> **What survives:** the estimator-vs-proposal gap (now confirmed at D=96 under physical
> noise, see table), the SNR^-0.5 precision law (Cramér-Rao form is noise-model
> independent), grid-independence of refinement, the antipodal degeneracy
> (pure algebra), the deconvolution negative result (information-theoretic).
> **What does not:** every specific number; the network-vs-ceiling gap (trained
> on unphysical noise); the mixture margins (posterior `beta` was calibrated
> under the wrong noise model). The D=48 corrected numbers are ALSO not
> representative — real boxes are 5× larger and information grows ~k_max^2.7.
>
> **Mixture vs argmax RE-RUN under physical noise (done 2 Sep)** — `tests/iterative_halfset_physical.py`,
> D=48, cross-referenced half-sets, posterior temperature re-derived (C measured from pure
> noise = 0.786; the old iid formula was 27% too cold). **The marginalisation result survives:**
>
>    | SNR (physical) | mixture − argmax, mean FSC | relative | peak iter |
>    |----------------|----------------------------|----------|-----------|
>    | 0.3 | +0.0075 to +0.0109 | +1.7% to +2.4% | 2 |
>    | 0.1 | +0.0091 to +0.0146 | +2.5% to +4.1% | 2–3 |
>
>    Margin grows as SNR falls (more ambiguity → more to gain from marginalising). Argmax
>    decays faster after its peak (−3.6% vs −2.0% at SNR 0.1). Half-map FSC tracks truth
>    within a shell, so the gold-standard estimate stays honest. **Both branches still
>    overfit after iteration 2–3** — the low-pass-between-iterations control remains the
>    missing ingredient. Same direction and magnitude as the pre-review +2–6%, so that
>    finding was right for the wrong reason and is now right for the right one.
>
> **Low-pass control RUN (2 Sep) — NEGATIVE, and it exposed a hazard.** `LOWPASS=1` in
> `iterative_halfset_physical.py`: each half-map low-passed at its HALF-MAP FSC 0.5 crossing
> before becoming the next bank (no truth leaks in).
>
>    - **Did not remove the decay.** Both branches still peak at iter 2–3 and fall. Peak
>      values slightly LOWER than without the filter (SNR 0.3: argmax 0.4516 vs 0.4548).
>      At SNR 0.1 both decay from iter 1 and end worse (argmax −6.3% vs −2.7% unfiltered).
>    - **Mixture margin survives regardless:** +1.3–2.7% (SNR 0.3), +3.7% at iter 6 (SNR 0.1).
>    - **HAZARD — half-map FSC inflates under the mixture.** At SNR 0.1 the mixture's half-map
>      FSC 0.5 climbs 10→12→13→14 px while its truth FSC sits at 9 px: a 5-shell overestimate,
>      i.e. a gold-standard resolution claim of 14 px for a 9 px map. Argmax stays honest
>      (10 vs 9). Mechanism: cross-referencing couples the halves (A posed against B, B
>      against A → they converge on shared error); low-passing both to the same cutoff makes
>      the references near-identical; the mixture's 10 poses per particle all chase that
>      shared reference. The inflated FSC raises the next cutoff, which filters less, which
>      inflates further — a feedback loop that switches the control off by iter 4.
>    - **So the decay is NOT an over-sharp reference.** Cause still open. Leading hypothesis:
>      cross-reference coupling drift. Test = TRUE gold standard (each half against its OWN
>      reference, fully independent) and compare decay + half-map honesty.
>
> **TRUE GOLD STANDARD RUN (2 Sep) — hypothesis CONFIRMED.** `GOLD=1` in
> `iterative_halfset_physical.py`: each half posed against its OWN previous reference, fully
> independent. Iteration 1 is bit-identical to cross-referenced (same initial reference), so
> everything after is the effect of the protocol alone.
>
>    | SNR 0.3 mean FSC | iter 1 | iter 2 | iter 6 | half-map vs truth |
>    |---|---|---|---|---|
>    | cross-ref argmax | 0.4435 | **0.4548** (peak) | 0.4508 | 12p vs 12p |
>    | GOLD argmax | 0.4435 | 0.4421 | 0.4394 | **10p vs 11p (honest)** |
>    | cross-ref mixture | 0.4518 | **0.4657** (peak) | 0.4583 | 13p vs 12p |
>    | GOLD mixture | 0.4518 | 0.4510 | 0.4440 | **10p vs 11p (honest)** |
>
>    - **The iteration-2 peak is a cross-referencing artefact.** It is a *boost*: each half's
>      poses improve using the OTHER half's data via its reference, so the map briefly reflects
>      ~8000 particles' worth of information. Then the halves converge on shared error and it
>      decays. Under GOLD there is no rise at all — a gentle monotonic decline from iter 1.
>    - **Half-map FSC is honest under GOLD** (10p throughout, ≤ truth + 1 shell, both SNRs).
>      The inflation seen under low-pass (14p vs 9p) was cross-referencing + filtering; it is
>      gone. **This is the protocol to use for any resolution claim.**
>    - **Cross-referencing produces genuinely better maps at its peak** (+2.5% argmax, +3.1%
>      mixture over GOLD's best) — the classic trade: shared information vs honest estimate.
>    - **A residual slow decline survives GOLD** (−0.9% argmax, −1.7% mixture over 6 iters at
>      0.3; −1.1% / −1.6% at 0.1). Cross-referencing explained the peak and most of the decay,
>      not all of it. Remaining mechanism: poses chasing noise in the raw, unfiltered reference.
>      The low-pass control failed under cross-ref because of the FSC feedback loop; under GOLD
>      the FSC is honest, so the loop is gone and **GOLD + LOWPASS is now the right test.**
>    - **Mixture margin survives a fifth protocol:** GOLD +1.0–2.0% (SNR 0.3), +1.9–2.6% (0.1),
>      every iteration. Protocols tested: contaminated (artefact), decontaminated, shared-ref
>      (self-reinforcement artefact), cross-ref, GOLD. Direction consistent in all four valid ones.
>
> **GOLD + LOWPASS RUN (2 Sep) — NEGATIVE, decisively.** Low-passing each half-map at its
> half-map FSC 0.5 crossing before it becomes the next bank, under gold standard:
>
>    | | GOLD raw ref, iter1→6 | GOLD + LP, iter1→6 | half-map |
>    |---|---|---|---|
>    | SNR 0.3 argmax | 0.4435→0.4394 (−0.9%) | 0.4435→**0.4213 (−5.0%)** | 10p honest |
>    | SNR 0.3 mixture | 0.4518→0.4440 (−1.7%) | 0.4518→**0.4275 (−5.4%)** | 10p honest |
>    | SNR 0.1 argmax | 0.3616→0.3578 (−1.1%) | 0.3616→**0.3340 (−7.6%)** | 9p honest |
>    | SNR 0.1 mixture | 0.3711→0.3651 (−1.6%) | 0.3711→**0.3339 (−10.0%)** | 9p honest |
>
>    - The FSC inflation is gone (GOLD works), but the map quality falls 5–10% instead of 1–2%.
>    - **Why:** the half-map FSC 0.5 crossing is a CONSERVATIVE estimate under gold standard
>      (that is what makes it honest). Cutting the reference there removes REAL signal — at
>      SNR 0.3 the map is good to 11p and the cut is at 10p. Worse reference → worse poses →
>      lower FSC → tighter cut (10→9p at SNR 0.1) → a downward spiral. The opposite failure
>      from the cross-ref run, same root cause: a hard cutoff driven by half-map FSC feeds back.
>    - **The mixture margin collapses to zero** at SNR 0.1 by iter 4 (0.3429 vs 0.3429). A
>      low-resolution reference makes every pose ambiguous, so spreading over basins becomes
>      uniform smearing. The margin needs a reference good enough to discriminate.
>    - A hard cut at FSC 0.5 is NOT what real pipelines do — RELION weights the reference
>      smoothly by per-shell SSNR and only uses 0.143 for the final claim. This control was
>      the crude version and the result shows why the crude version is wrong.
>
> **REINTERPRETATION of the residual decline.** Iteration 1 poses are estimated against a
> low-passed TRUTH volume — clean, noise-free. Every later iteration uses a real noisy
> reconstruction. So the drop from iter 1 is mostly the loss of a privileged clean reference,
> not pipeline instability. Under GOLD raw-ref, iter 2→6 is only −0.6% (argmax) / −1.6%
> (mixture) at SNR 0.3. **The pipeline is effectively stable from iteration 2.** The "decay"
> question is closed: the remaining 0.5–1.5% is noise accumulation at the level one expects.
>
> **PROTOCOL VERDICT: gold standard with the raw, unfiltered reference is the best tested.**
> Honest FSC, stable from iter 2, mixture margin intact (+1–2.6%).
>
> **Frequency marching RE-RUN under physical noise (2 Sep)** — `adaptive_sampling_physical.py`.
> Coarse stage k_cut=6 (8 shells, 9.5° grid, 662 views, **0.8% of full cost**), truth within
> one coarse cell of a top-100 candidate: SNR 1.0 **100%**, 0.3 **92%**, 0.1 **66%**. Viable at
> SNR ≥ 0.3; at 0.1 the first level loses ~1/3 of particles. Qualitative claim from the old
> generator holds. (Rows use their own grid-cell tolerance, so levels are not directly
> comparable as accuracy — only as survival for hierarchical search.)
>
> **Stage timing (2 Sep, D=48, uncontended)** — `stage_timing.py`. Per particle: coarse search
> (208 views) 38 µs, **refinement (100 steps) 17,452 µs = 99%**, mixture insert (10 poses)
> 163 µs, Wiener divide negligible. ~29 min/iteration at 10⁵ particles, all refinement.
> **Search is not the bottleneck; pose refinement is, by two orders of magnitude.** Levers:
> fewer steps (100 was generous), refine only the top mode, refine every Nth iteration.
>
> **Ceiling re-established at D=96 (done 2 Sep):** `tests/recall_ceiling_physical_boxsize.py`.
> Memory note: P=512 at D=96 puts the bank at 6.1 GB; the run peaked near the 16 GB limit.
> `recall_ceiling_physical_boxsize_lowmem.py` (P=256, smaller chunks, run with `python -u`)
> is the fallback if a larger D or N is attempted.
>
> **Fix landed:** `generator.generate_physical()` — real-space white noise,
> exact DFT to polar, real-space SNR (whole box, CTF-free projection).
> Validated: bridge cosine 1.00000000, realised SNR exact, correlation
> structure asserted. `recall_sweep_physical.py` is the corrected sweep.
> `generate()` kept, with a warning, so earlier scripts stay reproducible.
>
> Also cleared by review: `refine.py` has no pole singularity (polar 1–3°:
> 0.54–0.71° median, 96–98% recovered). Minor: `adaptive.py` stuffs
> `n_polar/n_azim` with point counts (unread, but misleading); runner PASS
> count is inflated by 1–2 non-verdict tokens in `test_ctf`/`test_generator`.

Repo `git@github.com:wtang8/dcryolike.git` (private) · branch `main` @ `ea9f4d4` · tree clean.
Layout: `dcryolike/` (ours) + `cryolike/` (Flatiron reference, gitignored, push URL deliberately
disabled) + `venv/` (gitignored, rebuild from `dcryolike/requirements.txt`).

## 1. Current State

**2026-09-04 (afternoon), heterogeneous reconstruction prepared, not yet run on the GPU.**
Plan (linear latent model, exact marginalisation of gain+latent per pose, heterogeneity enters
AFTER the V0 shortlist) is implemented and unit-verified on the CPU. New files:
`dcryolike/hetero.py` (Gram-over-all-gamma, closed-form marginal, per-voxel (K+1)x(K+1) M-step,
PCA gauge, rotation-leakage diagnostic), `dcryolike/pipeline.py` (Bank/select_modes/reconstruct
lifted from the halfset script), `generator.hinge_phantom` + `generate_physical_hetero`,
`tests/test_hetero.py` (14 checks, runs with `CUDA_VISIBLE_DEVICES=`), and three scripts:
`tests/hetero_gate0.py` (D=96 recall vs arm swing, consensus bank), `tests/hetero_classes_pilot.py`
(stage 2), `tests/hetero_linear_pilot.py` (stage 3). Both pilots execute end to end on the CPU
through the gridding backend (`DEV=cpu BACKEND=gridding SMOKE=1`); their NUFFT/GPU path is
untested. The phantom refactor is bit-identical to the previous `gaussian_atom_volume`.

**WSL crashes today (14:24, 15:05, 15:25, 15:31) were host-RAM OOM kills** of the four-worker
`measure_pose_search` job (5 x ~2 GB in a 15.9 GB VM), not dcryolike. Run heavy scripts under
`systemd-run --user --scope -p MemoryMax=5G -p MemorySwapMax=0 ...` and not concurrently with
that job.

**Implemented and verified — 11 suites / 50 assertions, `./dcryolike/run_tests.sh` → ALL GREEN.**
Relocatable: suite and harness both verified running from a foreign checkout invoked from `/`.

| module | status |
|---|---|
| `dcryolike/slicing.py` | NUFFT + gridding slices. NUFFT backward = analytic adjoint; point gradients finite-difference verified. Central slice theorem 4e-15, forward vs exact DFT 9e-7, adjoint cosine 1.0000000000. |
| `dcryolike/adaptive.py` | Shared-radius 2D/3D shells. Rotation preserves radius (1.4e-14) → **no cross-frequency interpolation**, all error angular. |
| `dcryolike/spheregrid.py` | SphereShell-backed ring grid. Quadrature 3.8e-13, adjoint 1.5e-16. sin_taper/HEALPix were built, measured, and **dropped** (SphereShell tied on interpolation, ~7000× better on quadrature). |
| `dcryolike/ill.py` | Gain/offset/noise-marginalised likelihood + **per-γ Isx correction**. Restores exact invariance (1.000000 vs 0.921 at gain 0.4/offset −30). Cost 0.66→1.57 ms. |
| `dcryolike/generator.py` | Synthetic particles; CTF matches CryoLike `_ctf_relion` to 1e-13. SNR calibrated to 0.21%. |
| `dcryolike/refine.py` | Gradient pose refinement vs exact likelihood. |
| `dcryolike/proposal.py` | Two-tower net on cross-shell phase features. **Works but under-performs — see §2.** |

**Not done / known broken:**
- **The per-γ ILL fix is NOT ported upstream.** `cryolike/.../integrated_log_likelihood_kernel.py`
  still has the broadcast `Isx_ms = torch.sum(..., dim=(2,3))`. Patch = replace that line + its two
  `unsqueeze` calls with `isx_per_gamma(...).unsqueeze(2)` plus a dtype cast (irfft returns real).
  Everything downstream (A/B/C/D, constant) unchanged; reduces to current behaviour for a radial mask.
- `proposal_net.pt` (46 MB) is **local only**, gitignored. Regenerate via `tests/train_proposal.py` (~7 min).
- Analysis scripts (`recall_sweep`, `basin_*`, `deconv_*`, `iterative_*`, `mixture_vs_argmax`,
  `capture_vs_box`, `adaptive_sampling`) assert nothing by design — runner labels them `NO ASSERTIONS`.
- No CI. One machine, one GPU (RTX 5070 Ti / sm_120 — torch **must** come from the cu128 index).

## 2. Working Hypotheses & Roadblocks

**The live question: is the proposal network worth building at all?** Evidence has moved against its
original justification and toward marginalisation instead.

Measured chain (D=48 phantom, SNR by our convention — internally consistent only, do not compare to
literature figures):
1. Exact-likelihood ceiling at SNR 0.05: top-1 **17.5%**, recall@500 **95.5%** → proposal framing beats
   point estimation. *(This is the result in the published artifact.)*
2. But refinement reaches the information limit **from any starting grid** — 208 refs ≡ 7,956 refs,
   tails included. So the coarse search was already cheap; the network's speed argument mostly evaporates.
3. Angular precision is noise-limited, not resolution-limited: σ ∝ SNR^-0.5 over five decades, no floor.
   Pose error of 2–5° costs 0–1 resolution shells → **precision is essentially solved**.
4. The real damage is **bias**: 16% of particles at SNR 0.05 land in a wrong basin, and their runner-up
   scores 0.997 of the top — statistically tied. Argmax converts a coin flip into coherent wrong signal.
5. Deconvolution of pose blur **does not work** (0 of 2–5 shells recovered even with oracle knowledge;
   a per-shell scalar cannot change FSC at all). Blur must be prevented, not undone.
6. Mixture (top-10 posterior-weighted) vs argmax: **+12.9% (artifact) → +3.9% → −2..−5% → +2..+6%**
   across contaminated / decontaminated / shared-reference-iterative / **cross-referenced half-set**.
   Only the last protocol is trustworthy. The effect is real, small, and protocol-dependent.

**Architectural commitments:**
- The network **proposes**, never decides. Exact likelihood adjudicates → floor = current method.
  (cryoDRGN-AI rejects amortised encoders for the same reason: they memorise on noisy data.)
- Shared radial list for 2D and 3D grids; SphereShell only.
- Score against **ground truth**, not half-map FSC, whenever the reference could be contaminated.

**Roadblocks:**
- **Reference decoupling is load-bearing.** A particle influencing the map it is scored against reversed
  the mixture result's sign. Any new loop must cross-reference half-sets.
- **Both branches overfit after iteration 2–3** even cross-referenced (argmax 0.4331→0.4547 peak then
  decay). Missing ingredient is a **resolution-limited reference** (low-pass between iterations), not
  more iterations. Until that is added, iterative numbers are confounded.
- Proposal net is at 73.5% recall@500 vs the 95.5% ceiling at SNR 0.05. **Untested hypothesis:** hard
  phase normalisation `C/|C|` discards reliability, giving a pure-noise coefficient the same weight as a
  confident one. Proposed fix: Wiener shrinkage `C/(|C|+λ)` with λ tied to the noise level.
- The half-set reference is still built from **true poses** — more favourable than a real refinement.

**Traps that already burned us** (see `dcryolike/README.md`): gradcheck+adjoint pass on a wrong operator;
coverage metric inspected padding columns; `|FFT_θ|` is provably identical for antipodal views; pose-search
bank drawn from ground truth. Each was caught only by a check with a *predictable* answer.

## 3. Exact Next Action

**Heterogeneous reconstruction, gate zero** (as soon as the GPU is free):

```
systemd-run --user --scope -p MemoryMax=5G -p MemorySwapMax=0 \
    venv/bin/python -u dcryolike/tests/hetero_gate0.py
```

It prints recall@K of the consensus-volume bank at D=96 for arm swings of 0, +-10, +-30 degrees
at physical SNR 0.3 / 0.1 / 0.05 (the swing-0 rows must reproduce the D=96 table above). The
decision: flat recall@100 across swing means the shortlist survives motion and the plan stands
(run `hetero_classes_pilot.py`, then `hetero_linear_pilot.py` with `K=0` and `K=1` at SNR 0.1
and compare the `V(z)` and `half` columns to `V0`); a collapse means the search itself must be
class-aware. Before either pilot, run `GPU: ./dcryolike/run_tests.sh` (test_hetero TEST 7 is the
generator bridge, GPU-only) and a `SMOKE=1` pilot on the GPU to shake out the NUFFT path.

Previously queued (still valid, lower priority now):

**The diagnostic phase is complete.** GOLD + raw reference is the protocol: honest FSC, stable
from iteration 2, mixture margin intact. The low-pass control was tested twice and failed both
ways (inflation under cross-ref; a 5–10% downward spiral under gold). The residual decline is
mostly the privileged clean initial reference. What remains is confirmation and engineering,
not diagnosis:

1. **Mixture at D=96, GOLD raw-ref protocol.** The margin is a D=48 result; D=96 is what
   transfers. Heavy (~30 min uncontended); use the lowmem pattern. This is the last
   measurement the marginalisation claim needs.
2. **Soft SSNR weighting of the reference** (RELION-style, per-shell, smooth). The only
   principled version of the reference-filtering idea left untested. Expected gain is small
   (recovering ≤1–2%), and it risks the same spiral if the weighting is too aggressive. Low
   priority unless (1) shows the margin is reference-quality-limited at D=96.
3. **Cut refinement cost** (99% of per-iteration time): fewer steps, top mode only, every
   Nth iteration.
4. **Write-up.** The body of validated results is now substantial: differentiable CryoLike with
   analytic adjoints; the physical-noise ceiling at D=96; precision without a resolution floor;
   grid-independence of refinement; the antipodal degeneracy; deconvolution impossibility;
   marginalisation margin across five protocols; and the protocol lessons (contamination,
   noise model, cross-referencing, hard low-pass). That is a paper's worth of negative and
   positive results.

```
./dcryolike/run_tests.sh
```

Then paste into a new session:

> Read `HANDOFF.md` (the REVIEW FINDING block first, including the D=96 table) and
> `dcryolike/README.md` in /home/wtang/cryoem. The noise model was replaced by
> `generator.generate_physical()` and the recall ceiling has been re-established at D=96.
> The diagnostic phase is complete (HANDOFF review block, "PROTOCOL VERDICT"). GOLD + raw
> reference is the protocol. Now confirm the marginalisation margin at realistic box size:
> port `tests/iterative_halfset_physical.py` to D=96 using the memory pattern from
> `tests/recall_ceiling_physical_boxsize_lowmem.py` (P=256, chunked bank, complex64,
> `python -u`), run `GOLD=1 LOWPASS=0` at physical SNR 0.1 and 0.05 with 8000 particles,
> 6 iterations. Report per iteration truth FSC, half-map FSC, mean FSC for argmax and
> mixture, and the margin. Keep the measured-C posterior temperature. If the margin holds
> at D=96, the marginalisation claim is complete; if it vanishes, report that plainly.
> Stop and diagnose if any control looks impossible. `dcryolike/tests/iterative_halfset.py` runs cross-referenced
> half-set refinement (argmax vs top-10 posterior mixture), scored against ground truth. Both branches
> peak at iteration 2–3 then decay, which is overfitting to the reference and confounds the
> mixture-vs-argmax comparison.
> Add a resolution-limited reference: low-pass each new half-map before it is used as the pose-search
> bank, with the cutoff set from that half-map's own FSC-vs-truth (or a fixed schedule, whichever you
> can justify). Re-run `SNR=0.05` and `SNR=0.1` and report whether the decay disappears and whether the
> mixture's +2..+6% mean-FSC advantage survives.
> Score against ground truth, never half-map FSC. Do not let a particle influence the reference it is
> posed against. If a control result looks impossible (e.g. true poses losing), stop and diagnose before
> reporting numbers.

Files to reference: `dcryolike/tests/iterative_halfset.py`, `dcryolike/tests/iterative_refine.py`
(shared-reference variant, for contrast), `dcryolike/dcryolike/generator.py`.

**Alternates:** (a) Wiener shrinkage in `dcryolike/proposal.py::crossshell_phase`, retrain via
`tests/train_proposal.py`, compare to the ceiling table in `tests/recall_sweep.py`.
(b) Port the per-γ ILL patch upstream into `cryolike/` and open a PR to flatironinstitute/CryoLike.

**Published artifact** (private, yours): https://claude.ai/code/artifact/5eb5c62a-4a61-4e50-96b8-0526f66e6632
— the recall-ceiling writeup. Note it predates findings 2–6 above; §2's later results are not in it.
