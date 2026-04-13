import numpy as np
import pandas as pd

def central_moments_matrix(X: np.ndarray, max_k: int = 4):
    """
    X: matriz (T, n) de retornos terminales históricos.
    Retorna:
      mu (n,)
      moments: dict k -> (n,) con momentos centrales k
      cov (n,n)
    """
    mu = X.mean(axis=0)
    centered = X - mu

    moments = {}
    for k in range(1, max_k + 1):
        if k == 1:
            moments[k] = mu.copy()  # primer momento = media
        else:
            moments[k] = np.mean(centered**k, axis=0)

    cov = np.cov(X.T, ddof=0)  # cov poblacional para consistencia
    return mu, moments, cov

def build_targets(terminal_returns_df: pd.DataFrame):
    """
    terminal_returns_df: DataFrame (T, n)
    """
    X = terminal_returns_df.to_numpy()
    mu, moments, cov = central_moments_matrix(X, max_k=4)

    targets = {
        "mu": mu,
        "moments": moments,  # moments[1]=mu, moments[2..4] centrales
        "cov": cov,
        "asset_names": terminal_returns_df.columns.tolist()
    }
    return targets