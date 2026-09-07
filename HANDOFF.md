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

**2026-09-07, evening. The project's central negative conclusion was wrong, and the
error was in the instrument.** Everything committed; harness green; nothing running.

### What was concluded, and why it was wrong

Earlier today this file recorded that pose marginalisation is "an artefact of small box
size": the mixture-vs-argmax margin measured +1-4% at D=48 but only +0.06-0.10% at D=96,
and a posterior-anatomy measurement showed the pose posterior is single-basin at D=96.
All of that is accurate as far as it goes. It does not support the conclusion.

The mixture in every one of those runs used a shortlist of **10 local maxima at least
NEIGH degrees apart, weighted by their PEAK values, with the in-plane angle MAXIMISED
rather than integrated**. Measuring what fraction of the full-grid posterior that
shortlist actually holds:

| selection | captured posterior mass | effective directions |
|-----------|------------------------|---------------------|
| 10 peaks (every earlier result) | **2.1%** at D=96, 3.3% at D=48 | 1.94 |
| 10 dense + in-plane marginal | 13.1% | 9.97 |
| 50 dense | 45.0% | 47.8 |
| 200 dense | 80.6% | 148.6 |
| 1000 dense | 98.2% | 286.5 |

So "the mixture" was a softmax normalised over about two percent of the probability, with
an effective spread of one direction. It was never a marginal. The negative result
measured competition between distinct basins, which genuinely vanishes with box size,
while the benefit lives in the WIDTH OF THE WINNING BASIN, which does not.

### The corrected result

Proper marginalisation (dense views, in-plane integrated, 200 modes, ~100% of posterior
mass), D=96, 8,000 particles, gold standard, raw reference
(`results/marginalisation_proper_D96_8k.txt`):

| | SNR 0.3 | SNR 0.1 |
|--------------------------|----------|----------|
| margin, iterations 2-6 | **+3.4 to 4.0%** | **+7.3 to 8.8%** |
| resolution vs truth | 26 px vs 25 | 22 px vs 21 |
| half-map | 24 px vs 23 | 20 px vs 19 |
| effective directions | 25.7 | 80.5 |
| peak-shortlist margin, same box | +0.06-0.10% | — |

The margin persists at full particle count, grows with noise in proportion to the
effective-direction count, is flat from iteration 3 (converged, not transient), and gains
a full resolution shell on an independent metric. It exceeds the +1-4% originally claimed
at D=48.

**Pose marginalisation works at realistic box size. The shortlist was the problem.**

### Mechanism

The benefit is not choosing between rival orientations; it is declining to collapse a
peak that has genuine width. Argmax snaps each particle to one of 30,522 grid points and
inserts coherent signal at a slightly wrong orientation, which accumulates as coherent
bias. The marginal inserts with the uncertainty the data support. That is why it shows up
as a resolution shell and not only as a mean-FSC number, and why it scales with noise.

### What this invalidates

1. Section 1 as written this morning, and the "regime-limited" framing throughout.
2. The evidence-gap distribution work (`results/evidence_gap_distribution_D96.txt`) and
   the adaptive-rule dead end: both measured BETWEEN-basin competition.
3. `results/pose_posterior_anatomy.txt` remains valid but its within-basin spread
   (0.38-0.83 deg) is inconsistent with 25-80 effective directions on a 1.19 deg grid,
   which implies ~6 deg of support. The scan used a +-4 deg cone and probably truncated
   the posterior. Redo with a wider cone before quoting either number.
4. Both published artifacts, which currently present the box-size story as the answer.
5. **All heterogeneous conclusions.** The conformational latent was estimated at the same
   10 peak modes with effective spread 1.00. With 25-80 directions of real pose
   uncertainty the latent posterior is far broader than anything measured, and the
   components were fitted at over-confident poses.

### Infrastructure (unchanged, all still valid)

Single precision by default (`DTYPE=double` restores the old path, bit-identical);
generators chunked and built in place; `iterative_halfset_physical.py` runs at D=96 and
now takes `MODES`, `PEAKS` and `GAMMA_MARG` from the environment, defaults reproducing
the D=48 result bit-identically; `test_hetero_scale.py` pins the reconstruction-scale
fix. 15 suites green.

## 2. Working Hypotheses & Roadblocks

- **The right object is a local posterior, not a shortlist of discrete candidates.** 200
  modes captured essentially all the mass, so a shortlist is still the right computational
  object, but it must be a REGION around the peak with proper weights. A proposal network
  should predict a direction and a width, which is an easier learning target than picking
  candidates.
- **Refinement and marginalisation answer different questions.** The earlier result that
  refinement reaches the accuracy limit from any starting grid is about finding the PEAK.
  This result says the peak is the wrong object. The project treated the first as settling
  the second.
- **A conclusion is only as good as the statistic behind it.** Two errors this session had
  the same shape: a mode count that could not see within-basin structure, and a mixture
  that discarded 98% of its posterior. Both were used to declare an absence. Any future
  claim of "no effect" should be accompanied by a measurement of what the instrument can
  detect.

## 3. Exact Next Action

**Re-run the D=48 comparison with proper marginalisation.** The peak shortlist captured
only 3.3% of the posterior there, so the original +1-4% is probably an underestimate too,
and the box-size comparison needs both ends measured the same way.

```
env D=48 NP=8000 ITERS=6 SNR=0.1 GOLD=1 MODES=200 PEAKS=0 GAMMA_MARG=1 \
    venv/bin/python -u dcryolike/tests/iterative_halfset_physical.py
```

Prediction to write down first: the D=48 margin with proper marginalisation exceeds the
+1.9-2.6% recorded with the peak shortlist at this SNR, and lands near or above the
+7.3-8.8% just measured at D=96, since the posterior is broader at the smaller box.

Then, in order: redo the anatomy scan with a wider cone; re-run the heterogeneous linear
pilot with `MODES=200 PEAKS=0 GAMMA_MARG=1`; rewrite both artifacts.


---

### Superseded, kept for the record



**2026-09-07, 15:30. The project's central positive claim has been characterised and
is regime-limited.** Everything committed and pushed; harness green; nothing running.

### The result

The mixture-vs-argmax margin (posterior-weighted insertion over a pose shortlist,
reported at +1-4% mean FSC across five protocols) is **an artefact of small box size**.
Measured at D=96, gold standard, raw reference, SNR 0.3, 8000 particles
(`results/mixture_D96_snr03_gold.txt`):

| iteration | D=48 (published) | D=96 (measured) |
|-----------|------------------|-----------------|
| 1         | +1.0 to +2.0%    | **+0.06%**      |
| 2         | "                | **+0.10%**      |
| 3         | "                | **+0.08%**      |

Fifteen to twenty times smaller, and the resolutions are identical to the shell
(25 px vs truth, 23 px half-map, both branches).

### Why: the anatomy of the pose posterior

`tests/pose_posterior_anatomy.py` (`results/pose_posterior_anatomy.txt`) replaces the
old habit of grid-searching (D, SNR), which was wrong because those are not independent
knobs. The posterior temperature beta = ||y||^2_w/(2 sigma^2 C) converges to D^2/2 -- the
PIXEL COUNT -- almost independently of SNR. What SNR moves is the EVIDENCE GAP between
competing poses. One scalar, spanning three orders of magnitude:

| D  | SNR  | beta | gap (nats) | basins | within-basin views | spread | tolerance |
|----|------|------|-----------|--------|--------------------|--------|-----------|
| 48 | 0.30 | 1432 |   3.1     | 1.40   | 606                | 0.95   | 2.98      |
| 48 | 0.10 | 1242 |   0.9     | 5.19   | 764                | 0.97   | 2.98      |
| 48 | 0.05 | 1198 |   0.5     | 7.27   | 809                | 0.97   | 2.98      |
| 64 | 0.30 | 2495 |  26.2     | 1.00   | 381                | 0.91   | 2.24      |
| 64 | 0.10 | 2196 |   2.4     | 1.69   | 569                | 0.96   | 2.24      |
| 96 | 0.30 | 5236 | 189.5     | 1.00   |  50                | 0.38   | 1.49      |
| 96 | 0.10 | 4830 |  47.5     | 1.00   | 152                | 0.65   | 1.49      |
| 96 | 0.05 | 4721 |  15.1     | 1.00   | 256                | 0.83   | 1.49      |

**BETWEEN-basin structure vanishes with box size**: 7.27 competing basins at D=48/0.05,
exactly 1.00 at D=96 at every SNR tested. A shortlist mixture captures exactly this, so
its advantage disappears.

**WITHIN-basin width does NOT vanish** -- 50 to 800 effective directions inside the
winning basin everywhere -- but it stays a roughly CONSTANT FRACTION of the success
tolerance (0.26-0.56 at D=96, 0.32 at D=48), because both scale as 1/k_max. So the pose
is determined to the same fraction of a resolution element at every box size, and finer
local marginalisation would buy the same small amount everywhere. It is not a recoverable
advantage.

### Two of my own errors, corrected

1. I first reported "n_eff = 1.00 at D=96, nothing to average". That statistic counted
   only local maxima >= NEIGH degrees apart, collapsing each basin to one mode. It meant
   ONE BASIN, not a determined pose. The dense-scan columns above are what it missed.
2. I claimed a "shared-latent E-step" was the missing piece for the heterogeneous work.
   It was not: `pipeline.reconstruct` already inserts per-mode CONDITIONAL moments
   weighted by mode probability, which IS exact EM for the joint (pose, latent) posterior.

### Heterogeneous results (all stand)

Gate: motion does not disturb the shortlist (recall@100 flat at 100/100/96.5% for swings
of 0, +-10, +-30 deg). E-step verified in isolation (rho 0.85 at the true pose, SNR 0.1,
D=96). Linear model recovers a hinge from a random start; at D=96, SNR 0.3, single
precision, independent halves: 8,000 particles -> rho 0.889, component 14 px, margin
+0.0042; 16,000 -> rho 0.893, component 16 px, margin +0.0047. Particle count limits the
component's resolution; it buys no extra margin.

### Infrastructure

Single precision by default (`DTYPE=double` restores the old path, bit-identical);
`generate_physical` and `generate_physical_hetero` both chunked and built in place
(`keep_clean=False` carries the weighted noise norm instead of the noiseless stack);
`iterative_halfset_physical.py` runs at D=96 (chunked bank, neighbour index, freed
before the neighbourhood stage) and still reproduces the D=48 margin bit-identically;
`test_hetero_scale.py` in the harness pins the reconstruction-scale fix and shows the
GAUGE PIN, not the gridding correction, is load-bearing. 15 suites green.

## 2. Working Hypotheses & Roadblocks

- **The controlling parameter is the evidence gap in nats, not (D, SNR).** Any future
  claim about marginalisation should be plotted against it. beta itself is nearly
  D^2/2 and carries no independent information.
- **Marginalising the pose helps only where competing basins are tied**, i.e. gap of
  order a few nats or less. That is D<=64 at SNR<=0.1 for this phantom. Real particle
  boxes are larger, so the honest headline is: this is a real effect with a
  characterised and narrow domain of validity, not a general improvement.
- **What has NOT been tested and could still change the picture:** real particles have
  a DISTRIBUTION of evidence gaps within one dataset (defocus, ice thickness, particle
  mass all vary). If a usable fraction of particles sits in the low-gap regime, a
  per-particle adaptive rule -- marginalise when the gap is small, argmax otherwise --
  would capture the benefit where it exists at no cost elsewhere. That is the one
  remaining route to a general claim and it is cheap to test.
- The heterogeneous model is a verified reimplementation of a 3DVA-class method with an
  unusually strong validation harness. Its distinctive element (averaging the latent over
  the pose posterior) inherits the same domain limit as the homogeneous claim.

