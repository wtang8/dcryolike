# HANDOFF — dcryolike

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

Recommended: add the low-pass control to the iterative loop — it unconfounds the only live result.

```
./dcryolike/run_tests.sh
```

Then paste into a new session:

> Read `HANDOFF.md` and `dcryolike/README.md` in /home/wtang/cryoem.
> Continue the marginalisation thread. `dcryolike/tests/iterative_halfset.py` runs cross-referenced
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
