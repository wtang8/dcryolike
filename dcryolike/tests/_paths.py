"""Put dcryolike and the cryolike reference on sys.path, relative to this file.

Imported for its side effect by every test, so the suite runs from any checkout
location. Set CRYOLIKE_SRC to point at the reference package if it is not a
sibling of dcryolike/.
"""
import os, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_CRYOLIKE = Path(os.environ.get("CRYOLIKE_SRC", _ROOT / "cryolike" / "src"))

for _p in (_ROOT / "dcryolike", _CRYOLIKE):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

if not (_CRYOLIKE / "cryolike").is_dir():
    sys.stderr.write(
        f"\n  cryolike reference package not found at {_CRYOLIKE}\n"
        f"  clone it beside dcryolike/:\n"
        f"    git clone https://github.com/flatironinstitute/CryoLike.git "
        f"{_ROOT / 'cryolike'}\n"
        f"  or set CRYOLIKE_SRC to its src/ directory.\n\n")
