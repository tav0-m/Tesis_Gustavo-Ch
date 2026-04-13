from __future__ import annotations

import os
import yaml
import numpy as np
import pandas as pd

from src.data.download import download_adj_close
from src.mm.targets import build_targets
from src.mm.bcd import bcd_solve_mm
from src.mm.diagnostics import matching_errors

from src.analysis.diagnostics_plots import (
    ensure_dir,
    plot_hist_vs_sim_terminal,
    plot_hist_grid_small_multiples,
    plot_cov_heatmaps,
    plot_corr_heatmaps,
    plot_convergence_bcd,
    plot_error_bars_by_asset,
    plot_qq_plots,
    plot_fan_chart,
    plot_violin_hist_vs_sim,
    plot_radar_errors_moments,
    plot_cov_error_heatmap,
    plot_scenario_probabilities,
    plot_moments_scatter_impl,
    plot_moments_panel,
    export_metrics_summary,
)


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

def compute_daily_returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Retornos diarios simples a partir de precios ajustados.
    Aplica ffill+bfill para rellenar NaN aislados (feriados) antes
    de descartar columnas — evita perder activos por un solo día sin dato.
    """
    rets = prices.pct_change().dropna(how="all")
    rets = rets.ffill().bfill()

    max_nan_ratio = 0.02
    nan_ratio  = rets.isna().mean(axis=0)
    cols_ok    = nan_ratio[nan_ratio <= max_nan_ratio].index.tolist()
    cols_drop  = nan_ratio[nan_ratio > max_nan_ratio].index.tolist()

    if cols_drop:
        print(f"  [AVISO] Columnas descartadas (>{max_nan_ratio:.0%} NaN): {cols_drop}")

    return rets[cols_ok]


def historical_terminal_returns(daily_returns: pd.DataFrame, H: int) -> np.ndarray:
    """
    Retornos terminales históricos con ventanas móviles de H días.
    terminal_t = prod_{k=0..H-1}(1+r_{t+k}) - 1
    Retorna array (T-H, n).
    """
    R      = daily_returns.values
    T0, n  = R.shape
    if T0 <= H:
        raise ValueError(f"Insuficientes observaciones: T={T0} debe ser > H={H}")
    T   = T0 - H
    out = np.empty((T, n), dtype=float)
    for t in range(T):
        out[t, :] = np.prod(1.0 + R[t:t+H, :], axis=0) - 1.0
    return out


def compute_terminal_returns_simulated(
    X_scenarios: np.ndarray,
    p: np.ndarray,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Samplea directamente de la distribución discreta (X_scenarios, p).
    X_scenarios: (N, n) retornos terminales H-días — no retornos diarios.
    """
    p_norm = np.asarray(p, dtype=float)
    p_norm = p_norm / p_norm.sum()
    idx    = rng.choice(len(p_norm), size=n_paths, replace=True, p=p_norm)
    return X_scenarios[idx, :]


def print_matching_quality(err: dict, assets: list) -> None:
    print("\n" + "=" * 72)
    print("  CALIDAD DE MATCHING (error absoluto por activo y momento)")
    print("=" * 72)
    print(f"{'Activo':<20} {'|err_m1|':>10} {'|err_m2|':>10} {'|err_m3|':>10} {'|err_m4|':>10}")
    print("-" * 72)
    for i, a in enumerate(assets):
        print(f"{a:<20} {err['m1'][i]:>10.5f} {err['m2'][i]:>10.5f} "
              f"{err['m3'][i]:>10.6f} {err['m4'][i]:>10.6f}")
    print("=" * 72)
    print(f"\n  Error medio |m1|: {np.mean(np.abs(err['m1'])):.5f}   "
          f"max: {np.max(np.abs(err['m1'])):.5f}")
    print(f"  Error medio |m2|: {np.mean(np.abs(err['m2'])):.5f}   "
          f"max: {np.max(np.abs(err['m2'])):.5f}\n")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def main() -> None:

    # ── 1) Configuración ────────────────────────────────────────────────────
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg["output"]["out_dir"]
    fig_dir = os.path.join(out_dir, "figures")
    tab_dir = os.path.join(out_dir, "tables")
    ensure_dir(out_dir)
    ensure_dir(fig_dir)
    ensure_dir(tab_dir)

    tickers  = cfg["universe"]["tickers"]
    min_cov  = float(cfg["universe"].get("min_coverage", 0.95))
    start    = cfg["data"]["start"]
    end      = cfg["data"]["end"]
    interval = cfg["data"].get("interval", "1d")

    H       = int(cfg["returns"]["horizon_H"])
    n_paths = int(cfg["returns"].get("n_paths", 10000))

    N_scenarios    = int(cfg["mm"]["N_scenarios"])
    seed           = int(cfg["mm"].get("seed", 123))
    bcd_max_iter   = int(cfg["mm"].get("bcd_max_iter", 50))
    tol            = float(cfg["mm"].get("tol", 1e-7))
    min_iter       = int(cfg["mm"].get("min_iter", 5))
    n_starts       = int(cfg["mm"].get("n_starts", 3))
    w_moments      = float(cfg["mm"].get("w_moments", 1.0))
    w_cov          = float(cfg["mm"].get("w_cov", 1.0))
    moment_weights = cfg["mm"].get(
        "moment_weights",
        {"k1": 1.0, "k2": 1.0, "k3": 1.0, "k4": 1.0}
    )
    bins = int(cfg.get("plots", {}).get("bins", 30))

    rng = np.random.default_rng(seed)

    # ── 2) Descargar precios ────────────────────────────────────────────────
    print("\n[1/7] Descargando precios ajustados...")
    prices, report = download_adj_close(
        tickers       = tickers,
        start         = start,
        end           = end,
        interval      = interval,
        min_coverage  = min_cov,
        drop_bad      = True,
        max_retries   = 3,
        retry_sleep_s = 2.0,
        show_warnings = False,
        fill_method   = "ffill",
    )
    print(f"  Universo final ({len(report.tickers_ok)} activos): {report.tickers_ok}")
    if report.tickers_bad:
        print(f"  Tickers descartados: {report.tickers_bad}")
    prices.to_csv(os.path.join(out_dir, "adj_close_prices.csv"))

    # ── 3) Retornos diarios ─────────────────────────────────────────────────
    print("\n[2/7] Calculando retornos diarios simples...")
    daily_returns            = compute_daily_returns_from_prices(prices)
    daily_returns.index.name = "Date"
    daily_returns.to_csv(os.path.join(out_dir, "daily_returns.csv"))
    assets = list(daily_returns.columns)
    print(f"  Shape retornos: {daily_returns.shape}   Activos: {len(assets)}")

    # ── 4) Retornos terminales históricos ───────────────────────────────────
    print(f"\n[3/7] Calculando retornos terminales históricos H={H}...")
    hist_term    = historical_terminal_returns(daily_returns, H=H)
    hist_term_df = pd.DataFrame(hist_term, columns=assets)
    hist_term_df.to_csv(
        os.path.join(out_dir, f"hist_terminal_returns_H{H}.csv"), index=False
    )
    print(f"  Ventanas terminales: {hist_term.shape[0]}   Activos: {hist_term.shape[1]}")

    # ── 5) Targets (momentos históricos) ────────────────────────────────────
    print("\n[4/7] Calculando momentos y covarianza objetivo (targets)...")
    targets = build_targets(hist_term_df)
    kurt_media = np.mean(
        targets["moments"][4] / (targets["moments"][2]**2)
    )
    print(f"  Kurtosis media: {kurt_media:.2f}")

    # ── 6) BCD Matching-Moment ──────────────────────────────────────────────
    print(f"\n[5/7] Ejecutando BCD Matching-Moment")
    print(f"  N_escenarios={N_scenarios}  max_iter={bcd_max_iter}  "
          f"n_starts={n_starts}  tol={tol}")
    x, p, history, _ = bcd_solve_mm(
        hist_term      = hist_term,
        targets        = targets,
        N              = N_scenarios,
        bcd_max_iter   = bcd_max_iter,
        tol            = tol,
        w_moments      = w_moments,
        w_cov          = w_cov,
        moment_weights = moment_weights,
        seed           = seed,
        n_starts       = n_starts,
        min_iter       = min_iter,
        verbose        = True,
    )

    # x: (n, N) → X: (N, n) para guardar en CSV
    X = x.T

    n_nonzero = int(np.sum(p > 1e-6))
    print(f"\n  F final                     : {history[-1]['objective']:.6f}")
    print(f"  Iteraciones                 : {len(history)}")
    print(f"  Escenarios activos (p>1e-6) : {n_nonzero} / {len(p)}")
    print(f"  p_max={p.max():.4f}  p_min={p.min():.6f}  std(p)={p.std():.4f}")

    # ── 7) Guardar resultados MM ────────────────────────────────────────────
    print("\n[6/7] Guardando resultados MM...")
    pd.DataFrame(X, columns=assets).to_csv(
        os.path.join(out_dir, "mm_scenarios_x.csv"), index=False
    )
    pd.DataFrame({"p": p}).to_csv(
        os.path.join(out_dir, "mm_probabilities_p.csv"), index=False
    )
    pd.DataFrame(history).to_csv(
        os.path.join(out_dir, "objective_history.csv"), index=False
    )

    err = matching_errors(x, p, targets)
    pd.DataFrame({"err_m1": err["m1"]}, index=assets).to_csv(
        os.path.join(tab_dir, "err_m1.csv")
    )
    pd.DataFrame({"err_m2": err["m2"]}, index=assets).to_csv(
        os.path.join(tab_dir, "err_m2.csv")
    )
    pd.DataFrame({"err_m3": err["m3"]}, index=assets).to_csv(
        os.path.join(tab_dir, "err_m3.csv")
    )
    pd.DataFrame({"err_m4": err["m4"]}, index=assets).to_csv(
        os.path.join(tab_dir, "err_m4.csv")
    )
    print_matching_quality(err, assets)

    # ── 8) Retornos terminales simulados ────────────────────────────────────
    print("[7/7] Simulando retornos terminales desde distribución MM...")
    terminal_arr = compute_terminal_returns_simulated(
        X_scenarios = X,
        p           = p,
        n_paths     = n_paths,
        rng         = rng,
    )
    terminal_sim = pd.DataFrame(terminal_arr, columns=assets)
    terminal_sim.to_csv(
        os.path.join(out_dir, f"terminal_returns_H{H}.csv"), index=False
    )
    print(f"  Paths generados: {terminal_sim.shape[0]}   Activos: {terminal_sim.shape[1]}")

    # ── 9) Gráficos de diagnóstico ──────────────────────────────────────────
    print("\nGenerando gráficos de diagnóstico...")

    # A) KDE histórico vs simulado — 1 PNG por activo
    plot_hist_vs_sim_terminal(
        daily_returns_csv        = os.path.join(out_dir, "daily_returns.csv"),
        terminal_returns_sim_csv = os.path.join(out_dir, f"terminal_returns_H{H}.csv"),
        H        = H,
        out_dir  = fig_dir,
        bins     = bins,
        date_col = "Date",
        prefix   = f"hist_terminal_H{H}",
    )

    # B) Grid panel — todos los activos en una figura
    plot_hist_grid_small_multiples(
        daily_returns_csv        = os.path.join(out_dir, "daily_returns.csv"),
        terminal_returns_sim_csv = os.path.join(out_dir, f"terminal_returns_H{H}.csv"),
        H        = H,
        out_dir  = fig_dir,
        bins     = 25,
        ncols    = 4,
        date_col = "Date",
        prefix   = "hist_grid",
    )

    # C) Heatmaps covarianza
    plot_cov_heatmaps(
        daily_returns_csv       = os.path.join(out_dir, "daily_returns.csv"),
        mm_scenarios_x_csv      = os.path.join(out_dir, "mm_scenarios_x.csv"),
        mm_probabilities_p_csv  = os.path.join(out_dir, "mm_probabilities_p.csv"),
        out_dir  = fig_dir,
        date_col = "Date",
        prefix   = "cov",
    )

    # D) Heatmaps correlación (histórico, MM, combinado triángulo sup/inf)
    plot_corr_heatmaps(
        daily_returns_csv       = os.path.join(out_dir, "daily_returns.csv"),
        mm_scenarios_x_csv      = os.path.join(out_dir, "mm_scenarios_x.csv"),
        mm_probabilities_p_csv  = os.path.join(out_dir, "mm_probabilities_p.csv"),
        out_dir        = fig_dir,
        date_col       = "Date",
        prefix         = "corr",
        triangle_upper = True,
    )

    # E) Convergencia BCD multi-start
    plot_convergence_bcd(
        objective_history_csv = os.path.join(out_dir, "objective_history.csv"),
        out_dir               = fig_dir,
        fname                 = "convergence_bcd.png",
    )

    # F) Errores de matching — barras apiladas por momento y activo
    plot_error_bars_by_asset(
        err_tables_dir = tab_dir,
        out_dir        = fig_dir,
        assets         = assets,
        fname          = "errors_by_asset.png",
    )

    # G) QQ-plots histórico y MM vs normal — grid por activo
    plot_qq_plots(
        daily_returns_csv        = os.path.join(out_dir, "daily_returns.csv"),
        terminal_returns_sim_csv = os.path.join(out_dir, f"terminal_returns_H{H}.csv"),
        H        = H,
        out_dir  = fig_dir,
        date_col = "Date",
        prefix   = "qq",
        ncols    = 4,
    )

    # H) Fan chart — bandas de percentiles con anotaciones
    plot_fan_chart(
        daily_returns_csv = os.path.join(out_dir, "daily_returns.csv"),
        H           = H,
        out_dir     = fig_dir,
        n_paths     = 5000,
        date_col    = "Date",
        asset_index = 0,
        asset_name  = assets[0].replace(".SN", "") if assets else None,
    )

    # I) Violin plots — distribuciones completas por activo
    plot_violin_hist_vs_sim(
        daily_returns_csv        = os.path.join(out_dir, "daily_returns.csv"),
        terminal_returns_sim_csv = os.path.join(out_dir, f"terminal_returns_H{H}.csv"),
        H        = H,
        out_dir  = fig_dir,
        date_col = "Date",
        prefix   = "violin",
    )

    # J) Radar de errores por momento
    plot_radar_errors_moments(
        err_tables_dir = tab_dir,
        out_dir        = fig_dir,
        assets         = assets,
        fname          = "radar_errors_moments.png",
    )

    # K) Heatmap error de correlación — solo si plot_corr_heatmaps generó cov_diff.csv
    cov_diff_path = os.path.join(fig_dir, "cov_diff.csv")
    if os.path.exists(cov_diff_path):
        plot_cov_error_heatmap(
            cov_diff_csv = cov_diff_path,
            out_dir      = fig_dir,
            fname        = "cov_error_heatmap.png",
        )
    else:
        print("  [AVISO] cov_diff.csv no encontrado — saltando heatmap de error")

    # L) Distribución de probabilidades + curva de Lorenz
    plot_scenario_probabilities(
        mm_probabilities_p_csv = os.path.join(out_dir, "mm_probabilities_p.csv"),
        out_dir                = fig_dir,
        fname                  = "scenario_probabilities.png",
        top_n                  = 30,
    )

    # M) Scatter momentos histórico vs MM con R²
    m1_hist = np.array(targets["moments"][1])
    m2_hist = np.array(targets["moments"][2])
    m3_hist = np.array(targets["moments"][3])
    m4_hist = np.array(targets["moments"][4])
    plot_moments_scatter_impl(
        m1_hist                = m1_hist,
        m2_hist                = m2_hist,
        m3_hist                = m3_hist,
        m4_hist                = m4_hist,
        mm_scenarios_x_csv     = os.path.join(out_dir, "mm_scenarios_x.csv"),
        mm_probabilities_p_csv = os.path.join(out_dir, "mm_probabilities_p.csv"),
        n_assets               = len(assets),
        out_dir                = fig_dir,
        H                      = H,
    )

    # N) Panel de momentos 4×1 — el gráfico principal de la tesis
    plot_moments_panel(
        hist_term_csv          = os.path.join(out_dir, f"hist_terminal_returns_H{H}.csv"),
        mm_scenarios_x_csv     = os.path.join(out_dir, "mm_scenarios_x.csv"),
        mm_probabilities_p_csv = os.path.join(out_dir, "mm_probabilities_p.csv"),
        out_dir                = fig_dir,
        H                      = H,
        fname                  = "moments_panel.png",
    )

    # O) Tabla resumen MAE/RMSE por momento
    export_metrics_summary(
        err_tables_dir = tab_dir,
        out_dir        = tab_dir,
        assets         = assets,
        fname          = "metrics_summary.csv",
    )

    print(f"\nOutputs guardados en: {out_dir}")
    print("Pipeline completado correctamente.")


if __name__ == "__main__":
    main()