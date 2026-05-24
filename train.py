# Same idea as web.py - just a thin wrapper so `python train.py` works.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from movie_recsys.train import main  # noqa: E402

if __name__ == "__main__":
    main()
