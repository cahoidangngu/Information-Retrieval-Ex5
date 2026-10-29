"""Stage-2 rankers. Each fit_* returns (scorer weights/intercept, train_time_s, peak_rss_mb)."""
import ctypes, platform, threading, time, warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import LinearSVC
from config import SEED


def _rss_mb():
    if platform.system() != "Windows":
        return float("nan")

    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + \
                   [(n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage")]
    pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
    k32 = ctypes.windll.kernel32; k32.GetCurrentProcess.restype = ctypes.c_void_p
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
    ctypes.windll.psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return pmc.WorkingSetSize / 1024 ** 2


class PeakMemory:
    """Samples process working set every 20 ms (whole process incl. liblinear's C heap)."""
    def __enter__(self):
        self.peak = _rss_mb(); self._stop = False
        def run():
            while not self._stop:
                self.peak = max(self.peak, _rss_mb()); time.sleep(0.02)
        self._t = threading.Thread(target=run, daemon=True); self._t.start(); return self

    def __exit__(self, *a):
        self._stop = True; self._t.join(); self.peak = max(self.peak, _rss_mb())


def _timed(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        with PeakMemory() as pm:
            t0 = time.perf_counter(); m = fn(); dt = time.perf_counter() - t0
    return m, dt, pm.peak, any(issubclass(x.category, ConvergenceWarning) for x in w)


def fit_linreg(X, y):
    m, dt, mem, cw = _timed(lambda: LinearRegression().fit(X, y))
    return {"w": m.coef_.ravel(), "b": float(m.intercept_), "train_s": dt, "peak_mb": mem, "no_converge": cw}


def fit_logreg(X, y_bin, C):
    m, dt, mem, cw = _timed(lambda: LogisticRegression(C=C, max_iter=2000, random_state=SEED).fit(X, y_bin))
    return {"w": m.coef_.ravel(), "b": float(m.intercept_[0]), "train_s": dt, "peak_mb": mem, "no_converge": cw}


def fit_ranksvm(Xd, yd, w, C):
    """Linear SVM on pair differences, no intercept. Primal solver (n_pairs >> n_features),
    squared-hinge loss (scikit-learn/liblinear default; the classic Ranking SVM uses hinge)."""
    mk = lambda: LinearSVC(C=C, fit_intercept=False, dual=False, loss="squared_hinge",
                           tol=1e-4, max_iter=1000, random_state=SEED).fit(Xd, yd, sample_weight=w)
    m, dt, mem, cw = _timed(mk)
    return {"w": m.coef_.ravel(), "b": 0.0, "train_s": dt, "peak_mb": mem, "no_converge": cw}


def score(model, X):
    return X @ model["w"] + model["b"]


def predict_time(model, X, reps=5):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); score(model, X); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def fit_ranksvm_hinge(Xd, yd, w, C, max_iter=1000):
    """Standard (L1) hinge Ranking SVM, as in the course slides' objective
    1/2||w||^2 + C sum max(0, 1 - w^T(x_i - x_j)): LinearSVC(loss='hinge', dual=True), no intercept."""
    mk = lambda: LinearSVC(C=C, fit_intercept=False, dual=True, loss="hinge",
                           tol=1e-4, max_iter=max_iter, random_state=SEED).fit(Xd, yd, sample_weight=w)
    m, dt, mem, cw = _timed(mk)
    return {"w": m.coef_.ravel(), "b": 0.0, "train_s": dt, "peak_mb": mem, "no_converge": cw, "n_iter": int(np.max(m.n_iter_))}
