import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from movie_recsys.recommend import main  # noqa: E402

if __name__ == "__main__":
    main()
