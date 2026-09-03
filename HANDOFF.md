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

**Both post-review re-runs are done** (D=96 ceiling; mixture vs argmax under physical noise —
tables in the review block). The marginalisation result stands. What remains, in order of value:

1. **The low-pass control.** Every iterative run — old and new, shared and cross-referenced —
   peaks at iteration 2–3 and decays. That is the reference sharpening past what the data
   supports. Low-pass each half-map at its own FSC-vs-truth cutoff (or a fixed schedule) before
   it becomes the next bank. If the decay disappears and the mixture margin holds, the
   marginalisation result is clean; if the margin vanishes, it was partly an overfitting artefact.
2. **Mixture at D=96.** Confirms the margin at realistic box size. Heavy: 8000 particles × 10
   modes × 25k points scattered into 96³. Use the lowmem pattern; expect ~30 min.
3. **Proposal network gap under physical noise at D=96.** Lowest value — the review showed the
   coarse search is already cheap.

```
./dcryolike/run_tests.sh
```

Then paste into a new session:

> Read `HANDOFF.md` (the REVIEW FINDING block first, including the D=96 table) and
> `dcryolike/README.md` in /home/wtang/cryoem. The noise model was replaced by
> `generator.generate_physical()` and the recall ceiling has been re-established at D=96.
> The mixture-vs-argmax comparison has been re-run under physical noise
> (`tests/iterative_halfset_physical.py`) and the margin survives (+2–4% mean FSC). Both
> branches still peak at iteration 2–3 and decay. Add the low-pass control: before each
> half-map becomes the next pose-search bank, low-pass it at a cutoff derived from that
> half-map's own FSC (against truth for this synthetic study; note that a real pipeline would
> use half-map FSC). Re-run at SNR 0.3 and 0.1 and report whether the decay disappears and
> whether the mixture margin survives. Keep cross-referenced half-sets, score against truth,
> keep the measured-C posterior temperature. `dcryolike/tests/iterative_halfset.py` runs cross-referenced
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
