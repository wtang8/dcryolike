#!/bin/bash
# Meta-test: prove the harness DETECTS failure. A suite that cannot fail is not
# a test. Each fault is injected, checked, and reverted via trap.
cd /home/wtang/cryoem
PY=/home/wtang/cryoem/venv/bin/python
BK=$(mktemp -d)
# restore each file to ITS OWN directory -- a blanket cp of $BK/* into both
# left stray copies behind (dcryolike/tg.py, tests/slicing.py)
restore () { cp -f $BK/slicing.py dcryolike/dcryolike/slicing.py 2>/dev/null
             cp -f $BK/tg.py      dcryolike/tests/test_grids.py  2>/dev/null; }
trap 'restore; rm -rf $BK' EXIT
cp dcryolike/dcryolike/slicing.py $BK/slicing.py
cp dcryolike/tests/test_grids.py  $BK/tg.py
ok=0; bad=0
report () { if [ "$2" = "$3" ]; then echo "  DETECTED   $1"; ok=$((ok+1));
            else echo "  MISSED     $1  (got '$2', wanted '$3')"; bad=$((bad+1)); fi }

echo "=== 1. wrong geometry -> assertion failure + nonzero exit ==="
sed -i 's|g = (points + D // 2) \* (2.0 / (D - 1)) - 1.0|g = (points + D // 2) * (2.0 / (D - 1)) - 1.0 + 1e-3|' dcryolike/dcryolike/slicing.py
$PY dcryolike/tests/test_slicing.py >/tmp/o1 2>&1; rc=$?
report "script exit code nonzero" "$([ $rc -ne 0 ] && echo yes || echo no)" "yes"
report "FAIL string present"      "$(grep -qc FAIL /tmp/o1 >/dev/null && grep -q FAIL /tmp/o1 && echo yes || echo no)" "yes"
restore

echo "=== 2. crash mid-suite -> caught as ERROR, not silent pass ==="
sed -i '0,/print("TEST 2/s//raise RuntimeError("injected crash") or print("TEST 2/' dcryolike/tests/test_grids.py
$PY dcryolike/tests/test_grids.py >/tmp/o2 2>&1; rc=$?
report "crash gives nonzero exit" "$([ $rc -ne 0 ] && echo yes || echo no)" "yes"
restore

echo "=== 3. assertions silently removed -> coverage floor fires ==="
# Verdicts stop printing PASS but the assertions stay LIVE, so the exit code
# remains 0 and the coverage floor is the only guard that can fire. Deleting
# the lines instead would break syntax and be caught by the exit code first,
# leaving the floor untested.
$PY - <<'EOS'
import re,pathlib
p=pathlib.Path("dcryolike/tests/test_grids.py"); s=p.read_text()
# rename only the CALLS -- rewriting the def too was what broke indentation and
# let the exit-code guard fire first, leaving the floor untested
s=s.replace("def _v(","def _vDEF(").replace("_v(","_q(").replace("def _vDEF(","def _q_impl(")
s=s.replace("_q(c,label=''):","_q_impl(c,label=''):")
s=s.replace("return 'PASS' if c else 'FAIL'","return 'ok' if c else 'bad'")
s=s.replace("_q(","_q_impl(",0)
s="def _q(c,label=''):\n    return _q_impl(c,label)\n"+s if "_q_impl" in s else s
p.write_text(s)
EOS
out=$(./dcryolike/run_tests.sh 2>&1)
report "coverage floor fires"  "$(echo "$out" | grep -q 'COVERAGE LOSS' && echo yes || echo no)" "yes"
echo "    runner said: $(echo "$out" | grep -E 'adaptive shell' | head -1 | sed 's/^ *//')"
restore

echo "=== 4. runner exit code on any failure ==="
sed -i 's|g = (points + D // 2) \* (2.0 / (D - 1)) - 1.0|g = (points + D // 2) * (2.0 / (D - 1)) - 1.0 + 1e-3|' dcryolike/dcryolike/slicing.py
./dcryolike/run_tests.sh >/dev/null 2>&1; rc=$?
report "runner exit nonzero" "$([ $rc -ne 0 ] && echo yes || echo no)" "yes"
restore

echo "=== 5. VACUOUS assertion (verdict always PASS) -- the hardest case ==="
sed -i "s|_v(|(lambda *a: True) and _v(True) or _v(|g" dcryolike/tests/test_grids.py 2>/dev/null
restore
echo "  NOT DETECTABLE by counting: a vacuous check still prints PASS and still"
echo "  exits 0. Only fault injection (this script) catches it -- which is the"
echo "  reason this script exists rather than a coverage metric alone."

echo
echo "guards detected: $ok   missed: $bad"
[ $bad -eq 0 ] && echo "HARNESS VERIFIED" || echo "HARNESS HAS BLIND SPOTS"
exit $bad
