from pathlib import Path
import runpy
import sys

LEGACY_DIR = Path(__file__).resolve().parent / "legacy"
sys.path.insert(0, str(LEGACY_DIR))
runpy.run_path(str(LEGACY_DIR / Path(__file__).name), run_name="__main__")
