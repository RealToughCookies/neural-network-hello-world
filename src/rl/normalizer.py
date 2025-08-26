from dataclasses import dataclass
import numpy as np

@dataclass
class RunningNorm:
    eps: float = 1e-5
    count: float = 0.0
    mean: np.ndarray | None = None
    var:  np.ndarray | None = None
    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32)
        if self.mean is None:
            self.mean = x.copy()
            self.var  = np.ones_like(x, dtype=np.float32)
            self.count = 1.0
            return
        n1 = self.count
        n2 = 1.0
        delta = x - self.mean
        tot = n1 + n2
        new_mean = self.mean + delta / tot
        self.var = ((n1 * self.var) + (n2 * (x - new_mean) * (x - new_mean))) / max(tot, 1.0)
        self.mean = new_mean
        self.count = tot
    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None: return x.astype(np.float32)
        return ((x - self.mean) / np.sqrt(self.var + self.eps)).astype(np.float32)
    def state_dict(self):
        return dict(eps=self.eps, count=self.count,
                    mean=None if self.mean is None else self.mean.tolist(),
                    var=None if self.var is None else self.var.tolist())
    def load_state_dict(self, d):
        self.eps   = d.get("eps", 1e-5)
        self.count = float(d.get("count", 0.0))
        self.mean  = None if d.get("mean") is None else np.array(d["mean"], dtype=np.float32)
        self.var   = None if d.get("var")  is None else np.array(d["var"],  dtype=np.float32)