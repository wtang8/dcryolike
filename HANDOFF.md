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

## 3. Exact Next Action

**Measure the distribution of the evidence gap, not its median.** Every number above is a
median over particles at a fixed nominal SNR. Real datasets vary per particle. Generate a
stack with realistic per-particle variation (defocus 0.5-2.5 um already varies; add
per-particle SNR spread of the kind ice-thickness produces, say a factor of 3 either way)
at D=96, and report the histogram of the evidence gap and the fraction of particles below
~3 nats, the threshold where the anatomy table shows basins genuinely compete.

Prediction to write before running: at D=96 with a 3x SNR spread, fewer than 10% of
particles fall below 3 nats, and the aggregate margin from an adaptive rule is under 0.5%.

Decision: if the low-gap fraction is small, the project's honest conclusion is written and
the work is a characterisation of when pose marginalisation helps, which is publishable as
such. If it is large (>25%), the adaptive rule is worth building and is the general claim.

\n---\n\n### Superseded, kept for the record\n\n

**2026-09-07, 04:30. A measurement made tonight changes what this project can claim.**
Everything is committed and pushed; harness green (14 suites); nothing running.

**THE FINDING: the pose posterior is one-hot at realistic box size.**
`tests/hetero_mode_spread.py` measures the effective number of shortlist modes,
exp(entropy of the posterior weights), with no fitting. Results
(`results/hetero_mode_spread_D{48,96}.txt`, 1000 particles per row, consensus reference):

| SNR  | D=48 n_eff | D=96 n_eff | D=48 w_true | D=96 w_true |
|------|-----------|-----------|------------|------------|
| 0.30 | 1.55      | **1.00**  | 0.12       | 0.83       |
| 0.10 | 5.32      | **1.00**  | 0.02       | 0.52       |
| 0.05 | 7.25      | **1.00**  | 0.01       | 0.33       |
| 0.02 | 8.74      | 2.68      | 0.00       | 0.07       |

(w_true = posterior weight on modes actually within tolerance of the true pose.)

**Why:** the posterior temperature is ~1,200 at D=48 and ~4,800 at D=96. A larger
image carries proportionally more evidence per particle, so the likelihood sharpens
and one mode takes all the weight. At D=96, SNR 0.05 the top mode is WRONG 67% of the
time and the posterior is confidently wrong: marginalisation cannot repair an error the
posterior does not know it is making.

**Consequences, in order of importance:**

1. **The heterogeneous "average the latent over the shortlist" claim is dead at D=96.**
   Averaging over a one-hot posterior is arithmetically identical to taking the argmax.
   Confirmed independently: MODES=1 and MODES=10 gave trajectories agreeing to three
   decimals (drivers 10 vs 11).
2. **The project's headline homogeneous result is predicted to collapse at D=96.**
   The mixture-vs-argmax margin (+1-4% mean FSC, reported across five protocols) was
   measured ENTIRELY at D=48, exactly where the posterior spreads. "Mixture at D=96" has
   been the outstanding item in this file since the diagnostic phase closed and was never
   run. It is now queued with a predicted NEGATIVE outcome (jobq 21, 22).
3. **The mechanism was never posterior correctness.** At D=48, w_true is 0.01-0.12: the
   spread is over mostly WRONG poses. The margin, if real, comes from diluting coherent
   bias into noise that averages away over many particles, not from the shortlist
   containing the answer. Any writeup that implies the latter is wrong.

**Also corrected tonight:** I claimed a "shared-latent E-step" was the missing piece.
It was not. `pipeline.reconstruct` line 225 already inserts per-mode CONDITIONAL moments
weighted by mode probability, which IS exact EM for the joint (pose, latent) posterior.
A single shared latent would be a cruder factorised approximation. Nothing needed writing.

**Heterogeneous results that stand** (all D=96, physical noise, independent halves,
single precision, `results/hetero_pilots_D96_driver18_8k_vs_16k_single.txt`):

| particles | rho(z, true angle) | component crossing | mean-FSC margin over V0 |
|-----------|--------------------|--------------------|------------------------|
| 8,000     | 0.889              | 14 px              | +0.0042                |
| 16,000    | 0.893              | **16 px**          | +0.0047                |

Doubling the data buys two shells of component resolution and no extra margin. The
gate (motion does not disturb the shortlist), the E-step isolation check (rho 0.85 at
the true pose), and the scale fix all stand.

**Infrastructure changed tonight:** single precision is now the default for experiment
code (`DTYPE=double` restores the old path, bit-identical); the generator fits 16,000
particles at D=96 (9.0 -> 4.5 GB) and reports its budget upfront;
`tests/test_hetero_scale.py` is in the harness and shows the GAUGE PIN, not the gridding
correction, is what holds the reconstruction scale (drift: gauge only +1.0%, correction
only -13.5%, neither -13.7%); `iterative_halfset_physical.py` now runs at D=96
(D/NP/ITERS/MODES/DTYPE/VIEW_DEG from the environment, chunked bank, neighbour index)
and still reproduces the D=48 margin (+1.0% at iteration 2, SNR 0.05).

## 2. Working Hypotheses & Roadblocks

- **Box size, not noise, decides whether marginalising the pose can help.** The
  temperature scales with the evidence per image, so the posterior sharpens as D grows.
  Every marginalisation result in this project was obtained at D=48 and must be re-read
  in that light. This is the single most important thing to carry forward.
- **If the D=96 mixture run comes out negative** (predicted), the honest position is:
  the pipeline is a well-verified reimplementation of a known linear model (nearest
  relative: cryoSPARC 3DVA) with an unusually strong validation harness, and the
  distinctive claim did not survive contact with realistic box size. That is publishable
  as a negative result about when pose marginalisation helps, which is not currently
  known in the field with this precision.
- **If it comes out positive**, the mechanism is NOT what was assumed and needs
  re-derivation before anything rests on it, because n_eff = 1.00 says there is nothing
  to average.
- Untested and now lower priority: K=2 (hinge nonlinearity), SNR 0.1 at 16,000 particles,
  CryoLike's fully integrated likelihood (offset + noise scale) for real data.

