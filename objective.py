from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Estadísticas del modelo (escenarios + probabilidades)
# ---------------------------------------------------------------------------

def compute_model_stats(
    x: np.ndarray,
    p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula los momentos centrales y la covarianza del modelo MM.

    Parámetros
    ──────────
    x : (n, N)  escenarios — n activos, N escenarios
    p : (N,)    probabilidades en el simplex

    Retorna
    ───────
    m1  : (n,)    medias ponderadas
    m2  : (n,)    varianzas ponderadas (momento central 2)
    m3  : (n,)    momento central 3 (skewness sin normalizar)
    m4  : (n,)    momento central 4 (kurtosis sin normalizar)
    cov : (n, n)  matriz de covarianza ponderada
    """
    n, N = x.shape
    p    = np.asarray(p, dtype=float).reshape(-1)
    p    = p / p.sum()   # garantizar que suma 1 dentro del cálculo

    m1  = x @ p                    # (n,)
    xc  = x - m1[:, None]          # (n, N)  centrado

    m2  = (xc ** 2) @ p            # (n,)
    m3  = (xc ** 3) @ p            # (n,)
    m4  = (xc ** 4) @ p            # (n,)
    cov = (xc * p[None, :]) @ xc.T # (n, n)

    return m1, m2, m3, m4, cov


# ---------------------------------------------------------------------------
# Cálculo de pesos de normalización
# ---------------------------------------------------------------------------

def safe_weights(
    targets: dict,
    w_moments: float,
    w_cov: float,
    moment_w: dict,
) -> tuple[dict, np.ndarray]:
    """
    Pesos de normalización para la función objetivo MM.

    Problema de la versión original
    ────────────────────────────────
    La versión anterior usaba  w / (Mik² + eps).
    Cuando Mik ≈ 0 (frecuente en m3 y m1 de algunos activos),
    los pesos explotaban a valores gigantescos (~1e12), haciendo
    el problema muy mal condicionado numéricamente.

    Solución implementada
    ──────────────────────
    Normalizar por la ESCALA del momento, definida como:
        escala_k = max(std_cross(Mk), valor_absoluto_mediano) + eps

    Esto captura la magnitud típica del momento a través de los
    n activos, y es robusta cuando algunos activos tienen Mik ≈ 0.

    Adicionalmente, W2 (covarianza) se normaliza por la escala
    off-diagonal de la matriz histórica, con diagonal en 0.

    Parámetros
    ──────────
    targets    : dict con 'moments' {1..4} y 'cov'
    w_moments  : peso escalar global del término de momentos
    w_cov      : peso escalar global del término de covarianza
    moment_w   : dict {'k1','k2','k3','k4'} con pesos relativos por momento

    Retorna
    ───────
    W1 : dict {1,2,3,4} → array (n,)  pesos por activo y momento
    W2 : array (n, n)                  pesos para covarianza off-diagonal
    """
    eps = 1e-8

    M1 = np.asarray(targets["moments"][1], dtype=float)
    M2 = np.asarray(targets["moments"][2], dtype=float)
    M3 = np.asarray(targets["moments"][3], dtype=float)
    M4 = np.asarray(targets["moments"][4], dtype=float)
    C  = np.asarray(targets["cov"],        dtype=float)

    def _escala(M: np.ndarray) -> np.ndarray:
        """
        Escala por activo: max(std_cross, mediana_abs) + eps.
        Para cada activo devuelve un escalar positivo que representa
        la magnitud típica del momento en ese activo.
        """
        std_global  = float(np.std(M))                        # dispersión cross-asset
        med_abs     = np.abs(M) + eps                         # magnitud por activo
        global_ref  = max(std_global, float(np.median(np.abs(M)))) + eps
        # Escala individual: mezcla de referencia global y magnitud propia
        # El max evita que activos con M≈0 reciban escala≈0
        return np.maximum(med_abs, global_ref * 0.1)

    scale1 = _escala(M1)
    scale2 = _escala(M2)
    scale3 = _escala(M3)
    scale4 = _escala(M4)

    W1 = {
        1: w_moments * moment_w["k1"] / (scale1 ** 2),
        2: w_moments * moment_w["k2"] / (scale2 ** 2),
        3: w_moments * moment_w["k3"] / (scale3 ** 2),
        4: w_moments * moment_w["k4"] / (scale4 ** 2),
    }

    # Escala de covarianza: usar la escala off-diagonal de C
    C_off = C.copy()
    np.fill_diagonal(C_off, 0.0)
    abs_C_off = np.abs(C_off)
    scale_cov = np.maximum(abs_C_off, float(np.std(C_off[C_off != 0])) + eps)

    W2 = w_cov / (scale_cov ** 2 + eps)
    np.fill_diagonal(W2, 0.0)   # no penalizar varianzas en este término

    return W1, W2


# ---------------------------------------------------------------------------
# Función objetivo MM
# ---------------------------------------------------------------------------

def mm_objective(
    x: np.ndarray,
    p: np.ndarray,
    targets: dict,
    W1: dict,
    W2: np.ndarray,
) -> float:
    """
    Función objetivo de Matching-Moment (Høyland & Wallace, 2001).

    F(x, p) = Σᵢ Σₖ W1[k][i] · (mᵢₖ − Mᵢₖ)²
            + Σᵢ<l W2[i,l] · (cᵢₗ − Cᵢₗ)²

    Parámetros
    ──────────
    x       : (n, N) escenarios
    p       : (N,)   probabilidades
    targets : dict con momentos históricos objetivo
    W1      : dict {1..4} → (n,) pesos por momento y activo
    W2      : (n, n)  pesos para covarianza (diagonal = 0)

    Retorna
    ───────
    val : float  valor del objetivo (≥ 0, óptimo = 0)
    """
    m1, m2, m3, m4, cov = compute_model_stats(x, p)

    M1 = targets["moments"][1]
    M2 = targets["moments"][2]
    M3 = targets["moments"][3]
    M4 = targets["moments"][4]
    C  = targets["cov"]

    val  = 0.0
    val += float(np.sum(W1[1] * (m1 - M1) ** 2))
    val += float(np.sum(W1[2] * (m2 - M2) ** 2))
    val += float(np.sum(W1[3] * (m3 - M3) ** 2))
    val += float(np.sum(W1[4] * (m4 - M4) ** 2))

    # Solo términos i < l (triángulo superior) para no duplicar
    diffC = cov - C
    mask  = np.triu(np.ones_like(diffC, dtype=bool), k=1)
    val  += float(np.sum(W2[mask] * (diffC[mask] ** 2)))

    return val
