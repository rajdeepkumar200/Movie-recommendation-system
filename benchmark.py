import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

# The real script lives in scripts/, but it needs src/ on the path. Easiest
# way to share code without copy-pasting: just exec it from here.
exec((ROOT / "scripts" / "benchmark.py").read_text())
