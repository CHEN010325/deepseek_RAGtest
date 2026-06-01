from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rageval.ui.app import main


if __name__ == "__main__":
    raise SystemExit(main())
