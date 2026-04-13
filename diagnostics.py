import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .objective import compute_model_stats

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_objective_history(history, out_csv):
    df = pd.DataFrame(history)
    df.to_csv(out_csv, index=False)
    return df

def matching_errors(x, p, targets):
    """
    Retorna errores por activo y por covarianza.
    """
    m1, m2, m3, m4, cov = compute_model_stats(x, p)

    err = {
        "m1": m1 - targets["moments"][1],
        "m2": m2 - targets["moments"][2],
        "m3": m3 - targets["moments"][3],
        "m4": m4 - targets["moments"][4],
        "cov": cov - targets["cov"],
    }
    return err

def plot_terminal_histogram(x, p, asset_names, out_path, bins=40):
    """
    Histograma del retorno terminal simulado (mezcla discreta).
    Para graficar, muestreamos escenarios según p.
    """
    ensure_dir(os.path.dirname(out_path))
    rng = np.random.default_rng(123)
    n, N = x.shape
    # sample de escenarios
    S = 15000
    idx = rng.choice(np.arange(N), size=S, p=p)
    Xs = x[:, idx]  # (n,S)

    for i, name in enumerate(asset_names):
        plt.figure()
        plt.hist(Xs[i, :], bins=bins)
        plt.title(f"Retorno terminal simulado (H días) - {name}")
        plt.xlabel("Return")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path.replace(".png", f"_{name}.png"), dpi=180)
        plt.close()