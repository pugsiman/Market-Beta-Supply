import pandas as pd
import numpy as np

class Beta:
    def __init__(self, x: np.array, y: np.array):
        self.x = np.atleast_2d(x).T
        self.y = np.atleast_2d(y).T
        self.n_obs = x.shape[0]
        self.x_mat = np.hstack([np.ones((self.x.shape[0], 1)), self.x])

    def ols(self):
        return np.ravel(self._ols(self.x, self.y))[-1]

    def welch(self, delta: float = 3, rho=2 / 256) -> float:
        bm_min, bm_max = (1 - delta) * self.x, (1 + delta) * self.x
        lower, upper = np.minimum(bm_min, bm_max), np.maximum(bm_min, bm_max)
        y_winsorized = np.atleast_2d(np.clip(self.y, lower, upper))
        weights = np.exp(-rho * np.arange(self.n_obs)[::-1]) if rho else None
        return np.ravel(self._ols(self.x_mat, y_winsorized, weights=weights))[1]

    def _ols(self, x, y, weights=None):
        if weights is None:
            weights = np.ones(x.shape[0])
        weights = np.diag(weights)

        return np.linalg.inv(x.T @ weights @ x) @ x.T @ weights @ y

