from pathlib import Path
import runpy

_LEGACY_PATH = Path(__file__).resolve().parent / "legacy" / Path(__file__).name
_NAMESPACE = runpy.run_path(str(_LEGACY_PATH))
globals().update({key: value for key, value in _NAMESPACE.items() if not key.startswith("__")})
