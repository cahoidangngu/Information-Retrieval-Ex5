"""Small helpers: output dirs, logging, environment info."""
import platform, sys, time
from pathlib import Path
from config import ROOT


def dirs(smoke):
    proc = ROOT / "data" / "processed" / ("smoke" if smoke else "full")
    res = ROOT / ("results_smoke" if smoke else "results")
    proc.mkdir(parents=True, exist_ok=True); res.mkdir(parents=True, exist_ok=True)
    (res / "figures").mkdir(exist_ok=True)
    return proc, res


class Logger:
    def __init__(self, path):
        self.path = Path(path)

    def __call__(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf8") as f:
            f.write(line + "\n")


def env_info():
    import numpy, scipy, sklearn, pandas
    return (f"python {sys.version.split()[0]} | numpy {numpy.__version__} | scipy {scipy.__version__} | "
            f"sklearn {sklearn.__version__} | pandas {pandas.__version__} | {platform.platform()}")
