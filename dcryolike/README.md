# dcryolike verification harness

    ./dcryolike/run_tests.sh          # the suite; exit code 0 iff everything passed
    ./dcryolike/verify_harness.sh     # proves the suite can DETECT failure
    ./venv/bin/python dcryolike/check_env.py

## Why there are two scripts

A green suite proves nothing on its own — every check in it might be vacuous.
Four times in this project a check looked fine and measured nothing:

| what looked right | what was actually happening |
|---|---|
| gradcheck + adjoint test passing | the slicing operator had the wrong geometry; both only compare autograd to the forward, whatever the forward is |
| "100% coverage" from `children_of` | the metric inspected padding columns, and reported 100% at radii that missed a third of the grid |
| `\|FFT_theta\|` as an in-plane invariant | provably identical for antipodal views, so nearest neighbour was 180 degrees for every view |
| mixture beating argmax by +12.9% | the pose-search bank came from the ground truth, so it selected noise that agreed with the answer |

Each was caught by a check with a *predictable* answer — the central slice
theorem, a deliberately-too-small radius, a known-bad baseline, true poses that
must not lose. That is the pattern the suite is built around.

## Guards

`run_tests.sh` applies three independent guards per suite, because any one can
silently stop working:

1. **exit code** — catches crashes and assertion failures
2. **FAIL string** — catches a failure the exit code missed
3. **minimum check count** — catches a suite that stopped asserting

`verify_harness.sh` injects real faults and confirms each guard fires. It
reverts via `trap`, so an interrupted run still restores the tree.

**Known limit:** no counting metric can detect an assertion that always passes —
it still prints PASS, still exits 0, still meets any floor. Only fault injection
catches that, which is why `verify_harness.sh` exists.

## Environment

`requirements.txt` is pinned. The non-obvious constraint: **torch must come from
the cu128 index**. The GPU is sm_120 (Blackwell), which CUDA 12.4 does not know
about, and the default PyPI wheel fails at kernel launch — not at import, so
`torch.cuda.is_available()` returns True right up until it breaks. Every check
in `check_env.py` therefore executes a kernel rather than querying a flag.

## What is NOT verified

- The two analysis scripts (`recall_sweep`, `basin_structure`) assert nothing by
  design — they are measurements. The runner labels them `NO ASSERTIONS`.
- No CI. The suite runs on one machine, one GPU.
- `proposal.py` training dynamics are unverified; only its feature contract is.
