# Convenience launcher so you can just `python web.py` from the repo root
# without futzing with PYTHONPATH.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from movie_recsys.web_app import main  # noqa: E402

if __name__ == "__main__":
    main()
