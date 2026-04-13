import numpy as np
import pandas as pd

def filter_by_coverage(prices: pd.DataFrame, min_coverage: float = 0.98) -> pd.DataFrame:
    """
    Filtra activos con suficientes datos en el período.
    coverage = porcentaje de días con dato no-nulo.
    """
    coverage = prices.notna().mean(axis=0)
    keep = coverage[coverage >= min_coverage].index.tolist()
    return prices[keep].dropna(how="all")

def compute_daily_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """
    Retornos diarios:
    - log: log(P_t/P_{t-1})
    - simple: P_t/P_{t-1} - 1
    """
    prices = prices.dropna(how="all")
    if method == "log":
        rets = np.log(prices / prices.shift(1))
    else:
        rets = prices / prices.shift(1) - 1.0
    return rets.dropna(how="all")

def compute_terminal_returns(daily_returns: pd.DataFrame, H: int) -> pd.DataFrame:
    """
    Retorno terminal a H días:
    - si returns son log: sumatoria rolling (equivale a log retorno H)
    - si returns son simples: producto rolling - 1
    Detecta por magnitud: asume log si hay negativos y valores pequeños.
    """
    term = daily_returns.rolling(H).sum()
    return term.dropna(how="all")