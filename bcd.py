from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .objective import mm_objective, safe_weights


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def project_simplex(p: np.ndarray) -> np.ndarray:
    """Proyecta p al simplex: p >= 0, sum(p) = 1."""
    p = np.asarray(p, dtype=float).copy()
    p[p < 0.0] = 0.0
    s = p.sum()
    if not np.isfinite(s) or s <= 0.0:
        p[:] = 1.0 / len(p)
    else:
        p /= s
    return p


def init_scenarios_from_history(hist_term: np.ndarray, N: int, seed: int = 123) -> np.ndarray:
    """
    Inicializa X (n, N) muestreando N filas al azar de hist_term (T, n).
    Añade ruido gaussiano pequeño para romper simetría y evitar
    el punto estacionario trivial donde todos los escenarios son iguales.
    """
    rng = np.random.default_rng(seed)
    T, n = hist_term.shape
    idx = rng.integers(0, T, size=N)
    Xs = hist_term[idx, :].copy()   # (N, n)

    # Ruido pequeño proporcional a la std de cada activo (1% de std)
    stds = hist_term.std(axis=0, keepdims=True)     # (1, n)
    noise = rng.standard_normal(Xs.shape) * stds * 0.01
    Xs = Xs + noise

    return Xs.T.copy()   # (n, N)


def _build_bounds_from_history(
    hist_term: np.ndarray,
    n: int,
    N: int,
    q_lo: float = 0.005,
    q_hi: float = 0.995,
) -> list[tuple[float, float]]:
    """
    Bounds para x_ij: cada activo i queda acotado en [q_lo, q_hi] histórico.
    Se amplía a q=0.005/0.995 (vs original 0.01/0.99) para capturar
    eventos extremos que el MM necesita replicar (alta kurtosis IPSA).
    """
    q_low  = np.quantile(hist_term, q_lo, axis=0)
    q_high = np.quantile(hist_term, q_hi, axis=0)
    q_low  = np.where(np.isfinite(q_low),  q_low,  -1.0)
    q_high = np.where(np.isfinite(q_high), q_high,  1.0)

    bounds = []
    for i in range(n):
        lo, hi = float(q_low[i]), float(q_high[i])
        if lo >= hi:
            lo, hi = hi - 1e-6, lo + 1e-6
        for _ in range(N):
            bounds.append((lo, hi))
    return bounds


# ---------------------------------------------------------------------------
# Solver BCD principal
# ---------------------------------------------------------------------------

def bcd_solve_mm(
    hist_term: np.ndarray,
    targets: dict,
    N: int,
    bcd_max_iter: int = 50,
    tol: float = 1e-7,
    w_moments: float = 1.0,
    w_cov: float = 1.0,
    moment_weights: dict | None = None,
    seed: int = 123,
    n_starts: int = 3,          # multi-start: número de semillas
    min_iter: int = 5,          # mínimo de iteraciones antes de evaluar convergencia
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict], tuple]:
    """
    Block Coordinate Descent para Matching-Moment.

    Mejoras sobre versión original:
    ─────────────────────────────────────────────────────────
    1. min_iter: nunca cortar antes de `min_iter` iteraciones.
       Evita la salida prematura que producía 1 sola iteración.

    2. Criterio de parada robusto: requiere convergencia en 3
       iteraciones consecutivas (no solo en 1).

    3. Multi-start: ejecuta BCD desde `n_starts` semillas
       distintas y retiene la solución con menor F objetivo.
       Mitiga el problema de mínimos locales (no-convexidad).

    4. Actualización parcial de x: si L-BFGS-B falla, se
       acepta el resultado igualmente si mejora el objetivo.

    5. Logging detallado: cada iteración registra si los
       optimizadores convergieron y el valor de F.

    Parámetros
    ──────────
    hist_term    : (T, n) retornos terminales históricos
    targets      : dict con 'moments' y 'cov' (salida de build_targets)
    N            : número de escenarios a generar
    bcd_max_iter : máximo de iteraciones BCD por start
    tol          : tolerancia de convergencia (cambio relativo en F)
    w_moments    : peso global del término de momentos
    w_cov        : peso global del término de covarianza
    moment_weights: pesos por momento {'k1','k2','k3','k4'}
    seed         : semilla base para reproducibilidad
    n_starts     : número de puntos de partida distintos (multi-start)
    min_iter     : mínimo de iteraciones antes de evaluar convergencia
    verbose      : imprimir progreso en consola

    Retorna
    ───────
    x        : (n, N) escenarios óptimos
    p        : (N,)   probabilidades óptimas
    history  : lista de dicts con métricas por iteración
    (W1, W2) : pesos usados en la función objetivo
    """
    if moment_weights is None:
        moment_weights = {"k1": 1.0, "k2": 1.0, "k3": 1.0, "k4": 1.0}

    W1, W2 = safe_weights(
        targets,
        w_moments=w_moments,
        w_cov=w_cov,
        moment_w=moment_weights,
    )

    n = hist_term.shape[1]
    bounds_x = _build_bounds_from_history(hist_term, n=n, N=N)
    cons_p   = [{"type": "eq", "fun": lambda pp: np.sum(pp) - 1.0}]
    bnds_p   = [(0.0, 1.0)] * N

    best_x       = None
    best_p       = None
    best_f       = np.inf
    best_history = []

    # ── Multi-start ────────────────────────────────────────────────────────
    for start_idx in range(n_starts):
        current_seed = seed + start_idx * 1000
        if verbose:
            print(f"\n[BCD] Start {start_idx + 1}/{n_starts}  (seed={current_seed})")

        x = init_scenarios_from_history(hist_term, N=N, seed=current_seed)
        p = np.ones(N, dtype=float) / N

        def obj_p(p_vec):
            return mm_objective(x, p_vec, targets, W1, W2)

        def obj_x(x_vec):
            return mm_objective(x_vec.reshape(n, N), p, targets, W1, W2)

        f_prev    = mm_objective(x, p, targets, W1, W2)
        history   = [{"iter": 0, "objective": float(f_prev),
                      "start": start_idx + 1, "conv_p": False, "conv_x": False}]
        consec_ok = 0   # iteraciones consecutivas que cumplen tolerancia

        for it in range(1, bcd_max_iter + 1):

            # ── Bloque A: optimizar p (fijando x) ──────────────────────────
            res_p = minimize(
                fun=obj_p,
                x0=p.copy(),
                method="SLSQP",
                bounds=bnds_p,
                constraints=cons_p,
                options={"maxiter": 1000, "ftol": 1e-13, "disp": False},
            )
            conv_p = bool(res_p.success and np.all(np.isfinite(res_p.x)))
            if conv_p:
                p = project_simplex(res_p.x)
            else:
                # Aceptar si mejora aunque no haya convergencia formal
                if np.all(np.isfinite(res_p.x)):
                    p_candidate = project_simplex(res_p.x)
                    if mm_objective(x, p_candidate, targets, W1, W2) < mm_objective(x, p, targets, W1, W2):
                        p = p_candidate
                        conv_p = True   # mejora parcial aceptada

            # ── Bloque B: optimizar x (fijando p) ──────────────────────────
            res_x = minimize(
                fun=obj_x,
                x0=x.reshape(-1),
                method="L-BFGS-B",
                bounds=bounds_x,
                options={"maxiter": 600, "ftol": 1e-13,
                         "gtol": 1e-8, "disp": False},
            )
            conv_x = bool(res_x.success and np.all(np.isfinite(res_x.x)))
            if conv_x:
                x = res_x.x.reshape(n, N)
            else:
                # Aceptar si mejora aunque no haya convergencia formal
                if np.all(np.isfinite(res_x.x)):
                    x_candidate = res_x.x.reshape(n, N)
                    if mm_objective(x_candidate, p, targets, W1, W2) < mm_objective(x, p, targets, W1, W2):
                        x = x_candidate
                        conv_x = True   # mejora parcial aceptada

            # ── Evaluar progreso ────────────────────────────────────────────
            f_now     = mm_objective(x, p, targets, W1, W2)
            delta_rel = abs(f_prev - f_now) / (abs(f_prev) + 1e-12)

            history.append({
                "iter":      it,
                "objective": float(f_now),
                "start":     start_idx + 1,
                "conv_p":    conv_p,
                "conv_x":    conv_x,
                "delta_rel": float(delta_rel),
            })

            if verbose:
                status = f"  it={it:3d}  F={f_now:.6f}  Δrel={delta_rel:.2e}"
                status += f"  conv_p={'Y' if conv_p else 'N'}  conv_x={'Y' if conv_x else 'N'}"
                print(status)

            # ── Criterio de parada ──────────────────────────────────────────
            # Solo evaluar DESPUÉS de min_iter y solo si
            # se cumplen 3 iteraciones consecutivas bajo tolerancia.
            if it >= min_iter and delta_rel < tol:
                consec_ok += 1
                if consec_ok >= 3:
                    if verbose:
                        print(f"  [BCD] Convergió en iteración {it} (3 consec. bajo tol)")
                    break
            else:
                consec_ok = 0

            f_prev = f_now

        # ── Guardar mejor start ─────────────────────────────────────────────
        f_final = float(mm_objective(x, p, targets, W1, W2))
        if verbose:
            print(f"  [BCD] Start {start_idx + 1} finalizado  F={f_final:.6f}")

        if f_final < best_f:
            best_f       = f_final
            best_x       = x.copy()
            best_p       = p.copy()
            best_history = history

    if verbose:
        print(f"\n[BCD] Mejor F={best_f:.6f}  (start con seed={seed + int(np.argmin([h[-1]['objective'] for h in [best_history]])) * 1000})")

    return best_x, best_p, best_history, (W1, W2)
