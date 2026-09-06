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

**2026-09-06 (early morning). Heterogeneous reconstruction: gate passed, pilots run, one component
of a real hinge motion recovered from a random start at D=96.** Everything is committed and pushed
(`main`); logs are under `dcryolike/results/hetero_*`. The machine was shut down at the end of the
session on the user's instruction.

Chain of results, in order (all physical noise, gold-standard halves where iterative):

| step | setting | result |
|---|---|---|
| gate zero | D=96, consensus bank, arm swing 0/±10/±30° | recall@100 flat: 100/100/96.5% at SNR 0.3/0.1/0.05 for every swing. Heterogeneity enters AFTER the shortlist. |
| E-step isolation | true pose + true component | rho(z,t) 0.85 (SNR 0.1), 0.92 (0.3) at D=96, matching the LS bound; at D=32 only 0.13 (box too small). E-step verified. |
| classes, D=48, SNR 0.1, ±15° | consensus / random / truth start | 51% / 51% / 58%: per-particle class evidence ~0.4 nats; maps drift back to consensus. Machinery classifies at SNR 0.3 ±30° from truth (84%). |
| linear, D=48 | K=1 any variant | rho ≤ 0.02: D=48 SNR 0.1 has 1.5% top-1 pose recall, nothing per-particle can work. |
| linear, D=96, SNR 0.3, unfiltered | 10 iters | rho 0.11→0.82, but amplitude 0.75 vs true 0.16 (PPCA noise inflation); V(z) worse than V0. |
| linear, D=96, band-limited k≤24 | 4 iters | rho 0.11→0.53, faster (noise dimensionality cut 8×). |
| linear, D=96, SNR 0.1, band-limited + half-set FSC shrinkage | 10 iters | rho 0.78, R² 0.59, shrunk amplitude 0.164 (true ≈0.16). V(z)=V0 at the 0.5 crossing (component resolved only to ~10 px). |
| linear, D=96, SNR 0.3, band-limited + shrinkage | 10 iters | rho 0.89, R² 0.80 by iter 4; **V(z) beats V0 in mean FSC at iters 2–5 (peak 0.4872 vs 0.4835)**, then the amplitude bias (shrunk 0.06→0.12, rising) overtakes and V(z) falls below V0 from iter 7. |

New/changed code this session: `hetero.py` (Gram-over-γ, closed-form gain+z marginal, per-voxel
M-step, PCA gauge, leakage), `pipeline.py` (Bank with complex64 option, memory-safe neighbour index
for 30k views, `duty_sleep` **GPU duty cycle default 0.8**), generator (`hinge_phantom`,
`generate_physical_hetero` chunked; DFT matrix chunked), `tests/test_hetero.py` (17 checks, in the
harness), `tests/hetero_estep_check.py`, `tests/hetero_gate0.py`, `tests/hetero_classes_pilot.py`
(INIT=consensus|random|truth), `tests/hetero_linear_pilot.py` (K, U_KMAX, U_SHRINK, U_INIT, D,
VIEW_DEG, mean-FSC columns).

Operational: the WSL crashes on 4 Sep were host-RAM OOM from the user's own multi-worker job, not
dcryolike. Run heavy scripts in `systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0`
(5 GB once gave a CUDA "unknown error" at D=96). D=96 generation peaked >13 GB GPU before the
chunking fix; ~9 GB after. Pausing a torch process from outside (SIGSTOP) does NOT cap GPU use —
kernels are queued seconds ahead — hence the in-process duty cycle. Never `pgrep -f`/`pkill -f` a
pattern from a shell whose own command line contains it (it kills the shell; use
`scratchpad/stop_run.sh`-style helper scripts).

## 2. Working Hypotheses & Roadblocks

- **The plan's premise holds.** Motion does not touch the shortlist at D=96 (gate), so pose
  proposal + marginalisation extends to heterogeneity unchanged. Verified, not assumed.
- **Per-particle decisions are information-limited; population models are not.** Classification
  fails at SNR 0.1 for a ±15° swing (0.4 nats/particle) while the linear model pools 4,000 weak
  signals and recovers the component (rho 0.78–0.89). This is the 3DVA-vs-classification story,
  reproduced with exact likelihoods.
- **Detection threshold.** With N particles per half and p Fourier coordinates, the motion's
  covariance eigenvalue competes with the noise bulk edge σ²(1+√(p/n_eff))²; at 8,000 particles the
  ratio is ~1 unfiltered and ~4 band-limited to k≤24. Band-limiting the component is the cheap
  lever; more particles is the honest one (ratio ∝ N).
- **Amplitude bias is the open problem.** EM with the diagonal-in-Fourier M-step inflates the
  component: the noise in E[z] is correlated with U itself in BOTH halves, so it is a systematic
  bias that the half-set FSC shrinkage slows but does not remove (SNR 0.3: V(z) beats V0 for
  iterations 2–5, then falls below). Candidate fixes, in order of principle: (a) PPCA-consistent
  scale, i.e. subtract the noise contribution from the component's fitted variance per shell
  (λ − σ²), (b) the exact coupled normal equations by conjugate gradient instead of the per-voxel
  diagonal approximation, (c) stop early at the mean-FSC peak (a cheat for diagnostics only).
- **The FSC 0.5 crossing is blind to the component** (resolved only to 10–14 px while the crossing
  sits at 21–25 px). Judge V(z) by mean FSC or by a masked real-space CC on the arm.
- Symmetry-breaking for discrete classes (random split) does not work at 4,000 particles per
  half even at SNR 0.3; not central to the plan, parked.

## 3. Exact Next Action

Fix the amplitude bias, then re-run the D=96 SNR 0.3 linear pilot and confirm V(z) stays above V0:

1. In `hetero_linear_pilot.py`, replace the FSC shrinkage with a PPCA-consistent per-shell scale:
   per shell, estimate the noise power in U from the half-difference (U_a − U_b)/2 and subtract it
   from the component power before rescaling (λ_shell − σ²_shell, floored at 0). Keep `U_KMAX=24`.
2. Run (about 45 min at the 80% duty):
```
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 \
    env D=96 K=1 SNR=0.3 ITERS=10 U_KMAX=24 venv/bin/python -u dcryolike/tests/hetero_linear_pilot.py
```
   Success = mean-FSC V(z) > V0 at every iteration from 2 on, shrunk amplitude ≈ 0.16 and flat.
3. Then NP=16000 (threshold ratio doubles) and K=2 (a hinge is nonlinear; the second component
   should carry the quadratic term, leakage should stay ~0).
4. Only after that: cut refinement cost, write-up.

Previously queued and still valid: mixture at D=96 on the GOLD raw-reference protocol (homogeneous
claim); soft SSNR reference weighting (low priority).
