#!/bin/bash
# Validation tests (test_*.py) plus a smoke run of the fast analysis scripts.
PY=/home/wtang/cryoem/venv/bin/python
cd /home/wtang/cryoem
pass=0; fail=0; declare -a failed
# Three independent guards, because any one of them can silently stop working:
#   1. exit code   - catches crashes AND assertion failures
#   2. FAIL string - catches a failure the exit code missed
#   3. min checks  - catches a suite that stopped asserting (the "0 checks" trap)
run () {
  local f=$1 label=$2 to=$3 minchecks=${4:-1}
  [ -f "$f" ] || { printf "%-26s MISSING\n" "$label"; fail=$((fail+1)); failed+=("$label: file missing"); return; }
  printf '%-26s ' "$label"
  local t0=$(date +%s)
  out=$(timeout "$to" $PY "$f" 2>&1); rc=$?
  local dt=$(( $(date +%s) - t0 ))
  if [ $rc -ne 0 ]; then
    echo "ERROR (rc=$rc, ${dt}s)"; fail=$((fail+1)); failed+=("$label: exit $rc")
    echo "$out" | tail -3 | sed 's/^/      /'
    return
  fi
  nf=$(echo "$out" | grep -cE "FAIL|ENVIRONMENT PROBLEM")
  np=$(echo "$out" | grep -c "PASS")
  if [ "$np" -lt "$minchecks" ] && [ "$minchecks" -gt 0 ]; then
    echo "COVERAGE LOSS ($np checks, expected >= $minchecks, ${dt}s)"
    fail=$((fail+1)); failed+=("$label: only $np checks, expected >= $minchecks"); return
  fi
  if [ "$nf" -gt 0 ]; then
    echo "FAIL ($np pass, $nf fail, ${dt}s)"; fail=$((fail+1)); failed+=("$label: $nf failing")
    echo "$out" | grep "FAIL" | head -3 | sed 's/^/      /'
  elif [ "$np" -eq 0 ]; then
    echo "ran  (NO ASSERTIONS - smoke only, ${dt}s)"; pass=$((pass+1))
  else
    echo "ok   ($np checks, ${dt}s)"; pass=$((pass+1))
  fi
}
echo "=========== ENVIRONMENT ==========="
run dcryolike/check_env.py             "environment"          120 10
echo
echo "=========== VALIDATION TESTS ==========="
run dcryolike/tests/test_determinism.py "determinism"          300 3
run dcryolike/tests/test_slicing.py    "slicing operators"    300 6
run dcryolike/tests/test_generator.py  "particle generator"   300 5
run dcryolike/tests/test_ctf.py        "CTF vs cryolike ref"  300 5
run dcryolike/tests/test_grids.py      "adaptive shell grids" 600 5
run dcryolike/tests/test_refine.py     "pose refinement"      900 5
run dcryolike/tests/test_proposal.py   "proposal features/net" 600 5
run dcryolike/tests/test_ill.py        "marginalised likelihood" 600 2
echo
echo "=========== ANALYSIS SCRIPTS (exit-code only, not verification) ==========="
run dcryolike/tests/recall_sweep.py    "recall ceiling sweep"  600 0
run dcryolike/tests/basin_structure.py "basin structure"       900 0
echo
echo "======================================="
echo "suites passed: $pass   failed: $fail"
for f in "${failed[@]}"; do echo "  - $f"; done
[ $fail -eq 0 ] && echo "ALL GREEN"
exit $fail
