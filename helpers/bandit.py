import numpy as np
import pickle
import config
import os


class LinearTS:
    def __init__(self, dim: int, lam: float = 1.0, sigma: float = 1.0):
        self.d = dim
        self.sigma = sigma
        self.A_inv = (1/lam)*np.eye(dim, dtype=np.float32)
        self.b = np.zeros(dim, dtype=np.float32)
        self.rng = np.random.default_rng()

    def select(self, pool: dict[str, np.ndarray]) -> str:
        mu = self.A_inv@self.b
        theta = self.rng.multivariate_normal(mu, self.sigma**2*self.A_inv)
        return max(pool.items(), key=lambda kv: kv[1]@theta)[0]

    def update(self, phi_vec: np.ndarray, r: float):
        Av = self.A_inv@phi_vec
        denom = 1+phi_vec@Av
        self.A_inv -= np.outer(Av, Av)/denom
        self.b += r*phi_vec


def save_bandit_state(bandit: "LinearTS"):
    with open(config.BANDIT_STATE_PATH, "wb") as f:
        pickle.dump({"A_inv": bandit.A_inv, "b": bandit.b}, f)


def load_bandit_state(dim: int) -> "LinearTS":
    bandit = LinearTS(dim)
    if os.path.exists(config.BANDIT_STATE_PATH):
        with open(config.BANDIT_STATE_PATH, "rb") as f:
            state = pickle.load(f)
        bandit.A_inv = state["A_inv"]
        bandit.b = state["b"]
    return bandit
