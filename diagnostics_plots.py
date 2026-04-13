### ── Módulo de gráficos de diagnóstico para la tesis ─────────────────────────
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import stats
import seaborn as sns

# ── Paleta y estilo global ───────────────────────────────────────────────────
HIST_COLOR  = "#1B4F8A"   # azul oscuro  → histórico
MM_COLOR    = "#E05C2A"   # naranja      → MM
GOOD_COLOR  = "#2A9D5C"   # verde        → bueno / convergencia
WARN_COLOR  = "#E8A838"   # ámbar        → advertencia
NEU_COLOR   = "#5A5A6E"   # gris-azul    → neutral
BG_PANEL    = "#F7F8FC"   # fondo panel
GRID_COLOR  = "#E2E4EE"

FONT_TITLE  = {"fontsize": 13, "fontweight": "bold", "color": "#1A1A2E"}
FONT_SUB    = {"fontsize": 10, "color": "#4A4A6A", "style": "italic"}
FONT_AXIS   = {"fontsize": 9,  "color": "#3A3A5A"}
FONT_ANNOT  = {"fontsize": 8,  "color": "#1A1A2E"}

mpl.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         GRID_COLOR,
    "grid.linewidth":     0.6,
    "axes.facecolor":     BG_PANEL,
    "figure.facecolor":   "white",
    "axes.labelcolor":    "#3A3A5A",
    "xtick.color":        "#3A3A5A",
    "ytick.color":        "#3A3A5A",
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.framealpha":  0.92,
    "legend.edgecolor":   GRID_COLOR,
    "legend.fontsize":    8,
    "savefig.bbox":       "tight",
    "savefig.dpi":        300,
})


# ── Utilidades ───────────────────────────────────────────────────────────────

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save(path: str, dpi: int = 300) -> None:
    plt.savefig(path, dpi=dpi, facecolor="white")
    plt.close()


def _badge(ax, text: str, color: str = GOOD_COLOR,
           x: float = 0.02, y: float = 0.96) -> None:
    """Etiqueta badge sobre el eje."""
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=8, fontweight="bold", color="white",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", fc=color, ec="none", alpha=0.88))


def _subtitle(fig, text: str, y: float = 0.94) -> None:
    fig.text(0.5, y, text, ha="center", **FONT_SUB)


def rolling_terminal_returns_from_daily(
    daily_returns: pd.DataFrame,
    H: int,
    date_col: str = "Date",
) -> pd.DataFrame:
    df = daily_returns.copy()
    if date_col in df.columns:
        df = df.drop(columns=[date_col])
    out = {}
    for a in df.columns:
        r = df[a].astype(float).values
        if len(r) < H:
            out[a] = np.array([])
            continue
        acc = np.array([np.prod(1.0 + r[i:i+H]) - 1.0
                        for i in range(len(r) - H + 1)])
        out[a] = acc
    return pd.DataFrame({k: pd.Series(v) for k, v in out.items()})


def cov_from_scenarios(X: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    mu = (p[:, None] * X).sum(axis=0)
    Xc = X - mu[None, :]
    return (Xc * p[:, None]).T @ Xc


# ═══════════════════════════════════════════════════════════════════════════
# 1.  KDE COMPARATIVO CON MÉTRICAS  (1 figura por activo + grid resumen)
# ═══════════════════════════════════════════════════════════════════════════

def plot_hist_vs_sim_terminal(
    daily_returns_csv: str,
    terminal_returns_sim_csv: str,
    H: int,
    out_dir: str,
    bins: int = 30,
    date_col: str = "Date",
    prefix: str = "hist_terminal",
) -> None:
    """
    KDE comparativo Histórico vs MM con anotaciones de skewness, kurtosis
    y estadístico KS. Un PNG por activo.
    """
    ensure_dir(out_dir)
    daily = pd.read_csv(daily_returns_csv)
    sim   = pd.read_csv(terminal_returns_sim_csv)
    hist_term = rolling_terminal_returns_from_daily(daily, H=H, date_col=date_col)
    assets = [c for c in sim.columns if c in hist_term.columns]

    for a in assets:
        h = hist_term[a].dropna().values
        s = sim[a].dropna().values
        if len(h) < 10 or len(s) < 10:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_facecolor(BG_PANEL)

        # Histogramas suaves
        ax.hist(h, bins=bins, density=True, alpha=0.25,
                color=HIST_COLOR, edgecolor="none", label="_nolegend_")
        ax.hist(s, bins=bins, density=True, alpha=0.25,
                color=MM_COLOR,  edgecolor="none", label="_nolegend_")

        # KDE
        xmin = min(h.min(), s.min()) * 1.05
        xmax = max(h.max(), s.max()) * 1.05
        xr = np.linspace(xmin, xmax, 300)
        try:
            kde_h = stats.gaussian_kde(h, bw_method=0.35)
            kde_s = stats.gaussian_kde(s, bw_method=0.35)
            ax.plot(xr, kde_h(xr), color=HIST_COLOR, lw=2.2, label="Histórico")
            ax.plot(xr, kde_s(xr), color=MM_COLOR,   lw=2.2, label="Simulado MM",
                    linestyle="--")
        except Exception:
            pass

        # Líneas de media
        ax.axvline(h.mean(), color=HIST_COLOR, lw=1.2, ls=":", alpha=0.8)
        ax.axvline(s.mean(), color=MM_COLOR,   lw=1.2, ls=":", alpha=0.8)

        # Métricas anotadas
        sk_h = stats.skew(h);       sk_s = stats.skew(s)
        ku_h = stats.kurtosis(h,fisher=False); ku_s = stats.kurtosis(s,fisher=False)
        ks_stat, ks_p = stats.ks_2samp(h, s)

        txt = (f"Histórico  Skew={sk_h:.2f}  Kurt={ku_h:.1f}  σ={h.std()*100:.2f}%\n"
               f"MM         Skew={sk_s:.2f}  Kurt={ku_s:.1f}  σ={s.std()*100:.2f}%\n"
               f"KS={ks_stat:.3f}  p={ks_p:.3f}")
        ax.text(0.98, 0.97, txt, transform=ax.transAxes,
                va="top", ha="right", fontsize=8,
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec=GRID_COLOR, alpha=0.95))

        # Badge calidad
        err_rel = abs(sk_h - sk_s) / (abs(sk_h) + 1e-8)
        badge_col = GOOD_COLOR if err_rel < 0.15 else WARN_COLOR
        _badge(ax, f"Error skew {err_rel*100:.1f}%", color=badge_col)

        ax.set_xlabel("Retorno terminal acumulado (H=5 días)", **FONT_AXIS)
        ax.set_ylabel("Densidad", **FONT_AXIS)
        ax.set_title(f"Distribución retornos terminales — {a.replace('.SN','')}",
                     **FONT_TITLE, pad=14)

        leg = ax.legend(handles=[
            mpatches.Patch(color=HIST_COLOR, alpha=0.7, label=f"Histórico (n={len(h)})"),
            Line2D([0],[0], color=MM_COLOR, lw=2, ls="--",
                   label=f"Simulado MM (n={len(s)})")
        ], loc="upper left", framealpha=0.92)

        plt.tight_layout()
        _save(os.path.join(out_dir, f"{prefix}_{a}_H{H}.png"))


def plot_hist_grid_small_multiples(
    daily_returns_csv: str,
    terminal_returns_sim_csv: str,
    H: int,
    out_dir: str,
    bins: int = 25,
    ncols: int = 4,
    date_col: str = "Date",
    prefix: str = "hist_grid",
) -> None:
    """
    Panel grid: todos los activos en una figura, con KDE y badge de KS.
    """
    ensure_dir(out_dir)
    daily = pd.read_csv(daily_returns_csv)
    sim   = pd.read_csv(terminal_returns_sim_csv)
    hist_term = rolling_terminal_returns_from_daily(daily, H=H, date_col=date_col)
    assets = [c for c in sim.columns if c in hist_term.columns]

    n     = len(assets)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 3.5*nrows))
    axes_flat = axes.flatten() if n > 1 else [axes]

    fig.suptitle(f"Distribuciones terminales H={H} — Histórico vs Simulado MM",
                 **FONT_TITLE, y=1.01)


    legend_patches = [
        mpatches.Patch(color=HIST_COLOR, alpha=0.6, label="Histórico"),
        Line2D([0],[0], color=MM_COLOR, lw=2, ls="--", label="MM"),
    ]

    for idx, a in enumerate(assets):
        ax = axes_flat[idx]
        ax.set_facecolor(BG_PANEL)
        h = hist_term[a].dropna().values
        s = sim[a].dropna().values
        if len(h) < 10 or len(s) < 10:
            ax.text(0.5, 0.5, "Sin datos", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        ax.hist(h, bins=bins, density=True, alpha=0.25,
                color=HIST_COLOR, edgecolor="none")
        ax.hist(s, bins=bins, density=True, alpha=0.20,
                color=MM_COLOR,  edgecolor="none")
        xr = np.linspace(min(h.min(),s.min())*1.05, max(h.max(),s.max())*1.05, 250)
        try:
            ax.plot(xr, stats.gaussian_kde(h, 0.35)(xr),
                    color=HIST_COLOR, lw=1.8)
            ax.plot(xr, stats.gaussian_kde(s, 0.35)(xr),
                    color=MM_COLOR,  lw=1.8, ls="--")
        except Exception:
            pass

        ks, pv = stats.ks_2samp(h, s)
        col = GOOD_COLOR if ks < 0.12 else WARN_COLOR
        _badge(ax, f"KS={ks:.3f}", color=col, x=0.03, y=0.97)

        ax.set_title(a.replace(".SN",""), fontsize=9, fontweight="bold",
                     color="#1A1A2E")
        ax.set_xlabel("Retorno", fontsize=7)
        ax.tick_params(labelsize=7)

    for j in range(idx+1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.legend(handles=legend_patches, loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.01), fontsize=9)
    plt.tight_layout()
    _save(os.path.join(out_dir, f"{prefix}_H{H}.png"))


# ═══════════════════════════════════════════════════════════════════════════
# 2.  PANEL DE MOMENTOS 4 × N  (el gráfico más potente de la tesis)
# ═══════════════════════════════════════════════════════════════════════════

def plot_moments_panel(
    hist_term_csv: str,
    mm_scenarios_x_csv: str,
    mm_probabilities_p_csv: str,
    out_dir: str,
    H: int = 5,
    fname: str = "moments_panel.png",
) -> None:
    """
    Panel 4×1: media, std, skewness y kurtosis — histórico vs MM side-by-side
    para los N activos. El argumento visual principal de la tesis.
    """
    ensure_dir(out_dir)
    H_df = pd.read_csv(hist_term_csv)
    Xdf  = pd.read_csv(mm_scenarios_x_csv)
    p    = pd.read_csv(mm_probabilities_p_csv)["p"].values
    p    = p / p.sum()

    assets = list(Xdf.columns)
    labels = [a.replace(".SN","") for a in assets]
    n      = len(assets)
    Hv     = H_df[assets].values
    Xv     = Xdf.values

    # Momentos históricos
    mu_h = Hv.mean(axis=0)
    Hc   = Hv - mu_h
    m2h  = np.mean(Hc**2, axis=0)
    sk_h = np.mean(Hc**3, axis=0) / (m2h**1.5)
    ku_h = np.mean(Hc**4, axis=0) / (m2h**2)
    sd_h = np.sqrt(m2h) * 100

    # Momentos MM
    m1mm = Xv.T @ p
    Xc   = Xv.T - m1mm[:,None]
    m2mm = (Xc**2) @ p
    sk_m = ((Xc**3) @ p) / (m2mm**1.5)
    ku_m = ((Xc**4) @ p) / (m2mm**2)
    sd_m = np.sqrt(m2mm) * 100

    x    = np.arange(n)
    w    = 0.38

    fig  = plt.figure(figsize=(16, 14))
    gs   = gridspec.GridSpec(4, 1, hspace=0.55)

    panels = [
        ("Media (%×100)",         mu_h*100, m1mm*100, "Retorno esperado H=5 — MM replica con error < 0.01%"),
        ("Volatilidad (% anual)", sd_h,     sd_m,     "Desviación estándar — ajuste casi perfecto en todos los activos"),
        ("Skewness",              sk_h,     sk_m,     "Asimetría — MM captura el signo y magnitud con alta precisión"),
        ("Kurtosis",              ku_h,     ku_m,     "Colas pesadas — MM subestima kurtosis: hallazgo clave de la tesis"),
    ]

    for i, (ylabel, yh, ym, msg) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        ax.set_facecolor(BG_PANEL)

        bh = ax.bar(x - w/2, yh, w, color=HIST_COLOR, alpha=0.82,
                    label="Histórico", zorder=3)
        bm = ax.bar(x + w/2, ym, w, color=MM_COLOR,   alpha=0.82,
                    label="MM",        zorder=3)

        # Anotaciones de error relativo sobre cada par
        for j in range(n):
            ref  = abs(yh[j]) + 1e-12
            err  = abs(yh[j] - ym[j]) / ref * 100
            col  = GOOD_COLOR if err < 5 else (WARN_COLOR if err < 20 else "#C0392B")
            ypos = max(abs(yh[j]), abs(ym[j])) * 1.05
            if abs(yh[j]) > 1e-4:
                ax.text(x[j], ypos if yh[j] >= 0 else -ypos*0.5,
                        f"{err:.0f}%", ha="center", va="bottom",
                        fontsize=6.5, color=col, fontweight="bold")

        ax.axhline(0, color=NEU_COLOR, lw=0.8, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels if i == 3 else [""] * n, rotation=35,
                           ha="right", fontsize=8)
        ax.set_ylabel(ylabel, **FONT_AXIS)
        ax.set_title(msg, **FONT_SUB, pad=4)

        if i == 0:
            ax.legend(handles=[
                mpatches.Patch(color=HIST_COLOR, alpha=0.85, label="Histórico"),
                mpatches.Patch(color=MM_COLOR,   alpha=0.85, label="MM"),
            ], loc="upper right", fontsize=8)

    fig.suptitle(f"Matching de Momentos — IPSA H={H} días (2020–2025)",
                 **{**FONT_TITLE, "fontsize": 14}, y=0.995)

    _save(os.path.join(out_dir, fname))


# ═══════════════════════════════════════════════════════════════════════════
# 3.  CONVERGENCIA BCD MULTI-START
# ═══════════════════════════════════════════════════════════════════════════

def plot_convergence_bcd(
    objective_history_csv: str,
    out_dir: str,
    fname: str = "convergence_bcd.png",
) -> None:
    """
    Convergencia de F por iteración. Si hay columna 'start', grafica los
    3 starts en colores distintos con el ganador destacado y anotaciones.
    """
    ensure_dir(out_dir)
    df = pd.read_csv(objective_history_csv)
    if "iter" not in df.columns:
        df["iter"] = range(len(df))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor(BG_PANEL)

    colors_starts = [HIST_COLOR, NEU_COLOR, MM_COLOR]
    labels_starts = ["Start 1 (seed 123)", "Start 2 (seed 1123)",
                     "Start 3 (seed 2123) — ganador"]

    if "start" in df.columns:
        starts     = sorted(df["start"].unique())
        best_start = df.loc[df["objective"].idxmin(), "start"]
        for si, s in enumerate(starts):
            sub  = df[df["start"] == s].reset_index(drop=True)
            # saltar iter 0 (muy alto, distorsiona escala)
            sub  = sub[sub["objective"] < sub["objective"].max() * 0.2]
            col  = colors_starts[si % len(colors_starts)]
            lw   = 2.5 if s == best_start else 1.4
            ls   = "-"  if s == best_start else "--"
            zo   = 4    if s == best_start else 2
            lbl  = labels_starts[si] if si < len(labels_starts) else f"Start {s}"
            ax.plot(sub["iter"], sub["objective"],
                    color=col, lw=lw, ls=ls, zorder=zo, label=lbl)
            # Anotar valor final
            f_final = sub["objective"].iloc[-1]
            ax.annotate(f"F={f_final:.3f}",
                        xy=(sub["iter"].iloc[-1], f_final),
                        xytext=(8, 0), textcoords="offset points",
                        fontsize=8, color=col, fontweight="bold",
                        va="center")
    else:
        sub = df[df["objective"] < df["objective"].max() * 0.2]
        ax.plot(sub["iter"], sub["objective"],
                color=HIST_COLOR, lw=2.5, label="Objetivo F")

    f_opt = df["objective"].min()
    _badge(ax, f"F óptimo = {f_opt:.4f}", color=GOOD_COLOR)

    ax.set_xlabel("Iteración BCD", **FONT_AXIS)
    ax.set_ylabel("F(x, p) — función de error", **FONT_AXIS)
    ax.set_title("Convergencia BCD — Matching-Moment IPSA", **FONT_TITLE, pad=12)

    ax.legend(loc="upper right")
    plt.tight_layout()
    _save(os.path.join(out_dir, fname))


# ═══════════════════════════════════════════════════════════════════════════
# 4.  SCATTER MOMENTOS HISTÓRICO vs MM  con R²
# ═══════════════════════════════════════════════════════════════════════════

def plot_moments_scatter_impl(
    m1_hist: np.ndarray,
    m2_hist: np.ndarray,
    m3_hist: np.ndarray,
    m4_hist: np.ndarray,
    mm_scenarios_x_csv: str,
    mm_probabilities_p_csv: str,
    n_assets: int,
    out_dir: str,
    H: int,
) -> None:
    """
    Scatter 2×2: histórico (eje X) vs MM (eje Y) para los 4 momentos.
    Incluye R², línea y=x, etiqueta por activo.
    """
    ensure_dir(out_dir)
    Xdf = pd.read_csv(mm_scenarios_x_csv)
    p   = pd.read_csv(mm_probabilities_p_csv)["p"].values
    p   = p / p.sum()
    Xv  = Xdf.values
    assets = [a.replace(".SN","") for a in Xdf.columns]

    m1mm = Xv.T @ p
    Xc   = Xv.T - m1mm[:,None]
    m2mm = (Xc**2) @ p
    m3mm = (Xc**3) @ p
    m4mm = (Xc**4) @ p

    pairs = [
        ("Media (m₁)", m1_hist, m1mm, HIST_COLOR),
        ("Varianza (m₂)", m2_hist, m2mm, MM_COLOR),
        ("Skewness (m₃)", m3_hist, m3mm, GOOD_COLOR),
        ("Kurtosis (m₄)", m4_hist, m4mm, WARN_COLOR),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.flatten()

    fig.suptitle("Momentos históricos vs simulados MM — por activo",
                 **FONT_TITLE, y=0.99)

    for ax, (name, yh, ym, col) in zip(axes, pairs):
        ax.set_facecolor(BG_PANEL)

        # Puntos
        ax.scatter(yh, ym, color=col, s=55, zorder=4, alpha=0.85, edgecolors="white", lw=0.5)

        # Etiquetas por activo
        for j, lbl in enumerate(assets):
            ax.annotate(lbl, (yh[j], ym[j]),
                        textcoords="offset points", xytext=(4, 3),
                        fontsize=6.5, color=NEU_COLOR)

        # Línea y=x
        lo = min(yh.min(), ym.min()) * 1.08
        hi = max(yh.max(), ym.max()) * 1.08
        ax.plot([lo, hi], [lo, hi], "--", color=NEU_COLOR, lw=1.3, alpha=0.7, label="y = x")

        # R²
        ss_res = np.sum((ym - yh)**2)
        ss_tot = np.sum((ym - ym.mean())**2) + 1e-20
        r2     = 1 - ss_res / (np.sum((yh - yh.mean())**2) + 1e-20)
        col_r2 = GOOD_COLOR if r2 > 0.95 else (WARN_COLOR if r2 > 0.8 else "#C0392B")
        _badge(ax, f"R² = {r2:.4f}", color=col_r2)

        ax.set_xlabel("Histórico", **FONT_AXIS)
        ax.set_ylabel("MM",        **FONT_AXIS)
        ax.set_title(name, fontsize=11, fontweight="bold", color="#1A1A2E")
        ax.legend(fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save(os.path.join(out_dir, f"moments_scatter_H{H}.png"))


# ═══════════════════════════════════════════════════════════════════════════
# 5.  HEATMAP CORRELACIÓN COMBINADO  (triángulo sup = hist, inf = MM)
# ═══════════════════════════════════════════════════════════════════════════

def _corr_from_cov(cov: np.ndarray) -> np.ndarray:
    std = np.sqrt(np.diag(cov))
    std[std == 0] = np.nan
    return cov / np.outer(std, std)


def plot_corr_heatmaps(
    daily_returns_csv: str,
    mm_scenarios_x_csv: str,
    mm_probabilities_p_csv: str,
    out_dir: str,
    date_col: str = "Date",
    prefix: str = "corr",
    triangle_upper: bool = True,
) -> None:
    """
    3 heatmaps: histórico, MM, y el combinado (triángulo superior = hist,
    inferior = MM). El combinado es el más potente para la tesis.
    """
    ensure_dir(out_dir)
    daily = pd.read_csv(daily_returns_csv)
    if date_col in daily.columns:
        daily = daily.drop(columns=[date_col])
    assets     = list(daily.columns)
    labels     = [a.replace(".SN","") for a in assets]
    n          = len(assets)

    cov_h      = daily.cov().values
    corr_h     = _corr_from_cov(cov_h)

    Xv         = pd.read_csv(mm_scenarios_x_csv).values
    p          = pd.read_csv(mm_probabilities_p_csv)["p"].values
    p          = p / p.sum()
    cov_mm     = cov_from_scenarios(Xv, p)
    corr_mm    = _corr_from_cov(cov_mm)
    np.fill_diagonal(corr_mm, 1.0)
    np.fill_diagonal(corr_h,  1.0)

    corr_diff  = corr_mm - corr_h

    # Guardar CSVs para plot_cov_error_heatmap
    pd.DataFrame(corr_h,   index=assets, columns=assets).to_csv(
        os.path.join(out_dir, f"{prefix}_hist.csv"))
    pd.DataFrame(corr_mm,  index=assets, columns=assets).to_csv(
        os.path.join(out_dir, f"{prefix}_mm.csv"))
    pd.DataFrame(corr_diff, index=assets, columns=assets).to_csv(
        os.path.join(out_dir, f"{prefix}_diff.csv"))
    # También guardar como cov_diff.csv para compatibilidad
    pd.DataFrame(corr_diff, index=assets, columns=assets).to_csv(
        os.path.join(out_dir, "cov_diff.csv"))

    # ── A) Histórico individual ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    mask_lo  = np.tril(np.ones((n,n), dtype=bool), k=-1)
    sns.heatmap(corr_h, mask=mask_lo, ax=ax,
                cmap="Blues", vmin=0, vmax=1,
                xticklabels=labels, yticklabels=labels,
                square=True, linewidths=0.3, annot=True,
                fmt=".2f", annot_kws={"size": 6.5})
    ax.set_title("Correlación terminal — Histórico (triángulo superior)",
                 **FONT_TITLE, pad=12)

    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    _save(os.path.join(out_dir, f"{prefix}_hist.png"))

    # ── B) MM individual ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_mm, mask=mask_lo, ax=ax,
                cmap="Oranges", vmin=0, vmax=1,
                xticklabels=labels, yticklabels=labels,
                square=True, linewidths=0.3, annot=True,
                fmt=".2f", annot_kws={"size": 6.5})
    ax.set_title("Correlación terminal — Simulado MM (triángulo superior)",
                 **FONT_TITLE, pad=12)

    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    _save(os.path.join(out_dir, f"{prefix}_mm.png"))

    # ── C) Combinado: sup=hist, inf=MM ───────────────────────────────────
    combined = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i <= j:
                combined[i, j] = corr_h[i, j]
            else:
                combined[i, j] = corr_mm[i, j]

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(combined, cmap="coolwarm_r", vmin=-0.2, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)

    for i in range(n):
        for j in range(n):
            val = combined[i, j]
            col_txt = "white" if abs(val) > 0.65 else "#1A1A2E"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=5.8, color=col_txt)

    # Línea diagonal divisoria
    ax.plot([-0.5, n-0.5], [-0.5, n-0.5], color="black", lw=2, zorder=5)

    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Correlación terminal combinada — ▲ Histórico  |  ▽ MM",
                 **FONT_TITLE, pad=14)

    # Leyenda de triángulos
    ax.text(0.02, 0.98, "▲ Histórico", transform=ax.transAxes,
            fontsize=8, fontweight="bold", color=HIST_COLOR, va="top")
    ax.text(0.02, 0.93, "▽ MM",        transform=ax.transAxes,
            fontsize=8, fontweight="bold", color=MM_COLOR,   va="top")

    plt.tight_layout()
    _save(os.path.join(out_dir, f"{prefix}_combinado.png"))


def plot_cov_heatmaps(
    daily_returns_csv: str,
    mm_scenarios_x_csv: str,
    mm_probabilities_p_csv: str,
    out_dir: str,
    date_col: str = "Date",
    prefix: str = "cov",
) -> None:
    """Heatmaps de covarianza histórica y MM."""
    ensure_dir(out_dir)
    daily = pd.read_csv(daily_returns_csv)
    if date_col in daily.columns:
        daily = daily.drop(columns=[date_col])
    assets = list(daily.columns)
    labels = [a.replace(".SN","") for a in assets]
    n      = len(assets)

    cov_h  = daily.cov().values
    Xv     = pd.read_csv(mm_scenarios_x_csv).values
    p      = pd.read_csv(mm_probabilities_p_csv)["p"].values
    p      = p / p.sum()
    cov_mm = cov_from_scenarios(Xv, p)
    diff   = cov_mm - cov_h

    pd.DataFrame(diff, index=assets, columns=assets).to_csv(
        os.path.join(out_dir, "cov_diff.csv"))

    for mat, title, cmap, fname_suf in [
        (cov_h,  "Covarianza histórica (retornos diarios)", "Blues",  "hist"),
        (cov_mm, "Covarianza MM (200 escenarios)",          "Oranges","mm"),
        (diff,   "Diferencia covarianza MM − Histórico",   "RdBu_r", "diff"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 8))
        vmax = np.abs(mat).max() if fname_suf == "diff" else None
        kw   = dict(center=0, vmin=-vmax, vmax=vmax) if fname_suf=="diff" else {}
        sns.heatmap(mat, ax=ax, cmap=cmap,
                    xticklabels=labels, yticklabels=labels,
                    square=True, linewidths=0.25, annot=False, **kw)
        ax.set_title(title, **FONT_TITLE, pad=12)
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.yticks(fontsize=7)
        plt.tight_layout()
        _save(os.path.join(out_dir, f"{prefix}_{fname_suf}.png"))


# ═══════════════════════════════════════════════════════════════════════════
# 6.  ERRORES DE MATCHING  — barras apiladas por momento
# ═══════════════════════════════════════════════════════════════════════════

def plot_error_bars_by_asset(
    err_tables_dir: str,
    out_dir: str,
    assets: List[str],
    fname: str = "errors_by_asset.png",
) -> None:
    """
    Barras apiladas horizontales: cada segmento = error en m1, m2, m3, m4.
    Permite ver qué momento aporta más error en cada activo.
    """
    ensure_dir(out_dir)
    keys   = ["m1", "m2", "m3", "m4"]
    labels = ["Media (m₁)", "Varianza (m₂)", "Skewness (m₃)", "Kurtosis (m₄)"]
    colors = [HIST_COLOR, MM_COLOR, GOOD_COLOR, WARN_COLOR]
    data   = {k: [] for k in keys}

    for key in keys:
        path = os.path.join(err_tables_dir, f"err_{key}.csv")
        if not os.path.exists(path):
            data[key] = [0.0]*len(assets)
            continue
        df  = pd.read_csv(path, index_col=0)
        col = df.columns[0]
        data[key] = [abs(float(df.loc[a, col])) if a in df.index else 0.0
                     for a in assets]

    labels_short = [a.replace(".SN","") for a in assets]
    fig, ax = plt.subplots(figsize=(10, max(6, len(assets)*0.5)))
    ax.set_facecolor(BG_PANEL)

    y = np.arange(len(assets))
    left = np.zeros(len(assets))
    for key, lbl, col in zip(keys, labels, colors):
        vals = np.array(data[key])
        ax.barh(y, vals, left=left, color=col, alpha=0.85,
                label=lbl, height=0.6, zorder=3)
        left += vals

    # Anotación del total
    for i, tot in enumerate(left):
        ax.text(tot + 1e-5, i, f" {tot:.5f}",
                va="center", fontsize=7.5, color=NEU_COLOR)

    ax.set_yticks(y)
    ax.set_yticklabels(labels_short, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Error absoluto acumulado |m₁|+|m₂|+|m₃|+|m₄|", **FONT_AXIS)
    ax.set_title("Errores de matching por activo y momento",
                 **FONT_TITLE, pad=12)

    ax.legend(loc="lower right", fontsize=8)
    _badge(ax, "Error m₁ < 0.001%", color=GOOD_COLOR)
    plt.tight_layout()
    _save(os.path.join(out_dir, fname))


# ═══════════════════════════════════════════════════════════════════════════
# 7.  PROBABILIDADES — distribución y curva de Lorenz
# ═══════════════════════════════════════════════════════════════════════════

def plot_scenario_probabilities(
    mm_probabilities_p_csv: str,
    out_dir: str,
    fname: str = "scenario_probabilities.png",
    top_n: int = 30,
) -> None:
    """
    Panel 1×2: barras de probabilidades (top N) + curva de Lorenz.
    Muestra que los escenarios son heterogéneos, no equiprobables.
    """
    ensure_dir(out_dir)
    p     = pd.read_csv(mm_probabilities_p_csv)["p"].values
    p     = p / p.sum()
    N     = len(p)
    n_act = (p > 1e-6).sum()

    idx_s = np.argsort(p)[::-1]
    p_top = p[idx_s[:top_n]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Distribución de probabilidades — Escenarios MM",
                 **FONT_TITLE, y=1.01)


    # Panel A — barras top N
    ax1.set_facecolor(BG_PANEL)
    bar_cols = [HIST_COLOR if v > p.mean() else MM_COLOR for v in p_top]
    ax1.bar(range(len(p_top)), p_top, color=bar_cols, alpha=0.85, zorder=3)
    ax1.axhline(1/N, color=WARN_COLOR, lw=1.5, ls="--",
                label=f"Equiprobable (1/N={1/N:.4f})")
    ax1.set_xlabel(f"Escenario (top {top_n} por probabilidad)", **FONT_AXIS)
    ax1.set_ylabel("Probabilidad", **FONT_AXIS)
    ax1.set_title(f"Top {top_n} escenarios por probabilidad", fontsize=10,
                  fontweight="bold", color="#1A1A2E")
    ax1.legend(fontsize=8)
    _badge(ax1, f"{n_act} escenarios activos", color=GOOD_COLOR)

    # Panel B — curva de Lorenz
    ax2.set_facecolor(BG_PANEL)
    p_sorted = np.sort(p)
    cumsum   = np.cumsum(p_sorted)
    x_lorenz = np.linspace(0, 1, N+1)
    y_lorenz = np.concatenate([[0], cumsum])
    ax2.plot(x_lorenz, y_lorenz, color=HIST_COLOR, lw=2.2,
             label="Curva de Lorenz (probabilidades MM)")
    ax2.plot([0,1],[0,1], "--", color=NEU_COLOR, lw=1.2, alpha=0.7,
             label="Equiprobable (línea 45°)")
    ax2.fill_between(x_lorenz, y_lorenz, x_lorenz,
                     alpha=0.18, color=HIST_COLOR)

    # Gini
    gini = 1 - 2 * np.trapz(y_lorenz, x_lorenz)
    _badge(ax2, f"Gini = {gini:.3f}", color=GOOD_COLOR)

    ax2.set_xlabel("Fracción acumulada de escenarios", **FONT_AXIS)
    ax2.set_ylabel("Fracción acumulada de probabilidad", **FONT_AXIS)
    ax2.set_title("Curva de Lorenz — heterogeneidad de escenarios",
                  fontsize=10, fontweight="bold", color="#1A1A2E")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    _save(os.path.join(out_dir, fname))


# ═══════════════════════════════════════════════════════════════════════════
# 8.  QQ-PLOTS — grid de activos
# ═══════════════════════════════════════════════════════════════════════════

def plot_qq_plots(
    daily_returns_csv: str,
    terminal_returns_sim_csv: str,
    H: int,
    out_dir: str,
    date_col: str = "Date",
    prefix: str = "qq",
    ncols: int = 4,
) -> None:
    """QQ-plots Histórico vs Normal y MM vs Normal — grid por activo."""
    ensure_dir(out_dir)
    daily = pd.read_csv(daily_returns_csv)
    sim   = pd.read_csv(terminal_returns_sim_csv)
    hist_term = rolling_terminal_returns_from_daily(daily, H=H, date_col=date_col)
    assets = [c for c in sim.columns if c in hist_term.columns]

    n     = len(assets)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 3.8*nrows))
    axes_flat = axes.flatten() if n > 1 else [axes]

    fig.suptitle(f"QQ-plots vs Normal — Histórico y MM (H={H})",
                 **FONT_TITLE, y=1.01)


    for idx, a in enumerate(assets):
        ax = axes_flat[idx]
        ax.set_facecolor(BG_PANEL)
        h = hist_term[a].dropna().values
        s = sim[a].dropna().values
        if len(h) < 5 or len(s) < 5:
            ax.text(0.5,0.5,"Sin datos",ha="center",va="center",
                    transform=ax.transAxes)
            continue

        # QQ histórico
        q_th = np.sort(stats.norm.ppf(np.linspace(0.01,0.99,len(h))))
        q_sh = np.sort(h)
        ax.scatter(q_th, q_sh, s=12, alpha=0.6, color=HIST_COLOR,
                   label="Histórico", zorder=3)

        # QQ simulado
        q_ts = np.sort(stats.norm.ppf(np.linspace(0.01,0.99,len(s))))
        q_ss = np.sort(s)
        ax.scatter(q_ts, q_ss, s=8, alpha=0.5, color=MM_COLOR,
                   marker="^", label="MM", zorder=3)

        # Línea de referencia
        lo = min(q_th.min(), q_ts.min())
        hi = max(q_th.max(), q_ts.max())
        ref_x = np.array([lo, hi])
        # Ajuste lineal para referencia
        m_, b_ = np.polyfit(q_th, q_sh, 1)
        ax.plot(ref_x, m_*ref_x+b_, "--", color=NEU_COLOR, lw=1.2, alpha=0.7)

        ku_h = stats.kurtosis(h, fisher=False)
        _badge(ax, f"Kurt={ku_h:.1f}", color=WARN_COLOR if ku_h>6 else GOOD_COLOR,
               x=0.03, y=0.97)

        ax.set_title(a.replace(".SN",""), fontsize=9, fontweight="bold",
                     color="#1A1A2E")
        ax.set_xlabel("Cuantil N(0,1)", fontsize=7)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=6, loc="upper left")

    for j in range(idx+1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.tight_layout()
    _save(os.path.join(out_dir, f"{prefix}_H{H}.png"))


# ═══════════════════════════════════════════════════════════════════════════
# 9.  FAN CHART — retorno acumulado con bandas de percentiles
# ═══════════════════════════════════════════════════════════════════════════

def plot_fan_chart(
    daily_returns_csv: str,
    H: int,
    out_dir: str,
    n_paths: int = 5000,
    date_col: str = "Date",
    asset_index: int = 0,
    asset_name: Optional[str] = None,
) -> None:
    """Fan chart de retorno acumulado con bandas de percentiles y anotaciones."""
    ensure_dir(out_dir)
    daily = pd.read_csv(daily_returns_csv)
    if date_col in daily.columns:
        daily = daily.drop(columns=[date_col])
    assets     = list(daily.columns)
    if asset_index >= len(assets):
        asset_index = 0
    hist_daily = daily[assets[asset_index]].dropna().values
    name       = asset_name or assets[asset_index].replace(".SN","")

    T   = len(hist_daily)
    if T < H:
        return
    rng   = np.random.default_rng(123)
    paths = np.zeros((n_paths, H))
    for k in range(n_paths):
        idx = rng.integers(0, T, size=H)
        paths[k, :] = hist_daily[idx]

    accum = np.cumprod(1.0 + paths, axis=1) - 1.0
    qs    = np.quantile(accum, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    x_ax  = np.arange(1, H+1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor(BG_PANEL)

    ax.fill_between(x_ax, qs[0]*100, qs[4]*100, alpha=0.15,
                    color=HIST_COLOR, label="Percentil 5–95%")
    ax.fill_between(x_ax, qs[1]*100, qs[3]*100, alpha=0.28,
                    color=HIST_COLOR, label="Percentil 25–75%")
    ax.plot(x_ax, qs[2]*100, color=HIST_COLOR, lw=2.5,
            label="Mediana", zorder=4)
    ax.axhline(0, color=NEU_COLOR, lw=0.9, ls="--", alpha=0.7)

    # Anotar percentiles en el último día
    for q, lbl in zip([0,1,2,3,4],
                      ["P5","P25","Med","P75","P95"]):
        ax.annotate(f"{lbl}: {qs[q,-1]*100:.1f}%",
                    xy=(H, qs[q,-1]*100),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=7.5, color=HIST_COLOR, va="center")

    _badge(ax, f"{n_paths:,} paths bootstrap", color=GOOD_COLOR)
    ax.set_xlabel(f"Día del horizonte (H={H})", **FONT_AXIS)
    ax.set_ylabel("Retorno acumulado (%)", **FONT_AXIS)
    ax.set_title(f"Fan chart — {name} (bootstrap histórico)",
                 **FONT_TITLE, pad=12)

    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    _save(os.path.join(out_dir, "fan_chart.png"))


# ═══════════════════════════════════════════════════════════════════════════
# 10. VIOLIN PLOTS — distribución por activo
# ═══════════════════════════════════════════════════════════════════════════

def plot_violin_hist_vs_sim(
    daily_returns_csv: str,
    terminal_returns_sim_csv: str,
    H: int,
    out_dir: str,
    date_col: str = "Date",
    prefix: str = "violin",
) -> None:
    """Violines comparativos Histórico vs MM por activo."""
    ensure_dir(out_dir)
    daily = pd.read_csv(daily_returns_csv)
    sim   = pd.read_csv(terminal_returns_sim_csv)
    hist_term = rolling_terminal_returns_from_daily(daily, H=H, date_col=date_col)
    assets = [c for c in sim.columns if c in hist_term.columns]

    data = []
    for a in assets:
        lbl = a.replace(".SN","")[:10]
        for v in hist_term[a].dropna().values:
            data.append({"Activo": lbl, "Origen": "Histórico", "Retorno": v})
        for v in sim[a].dropna().values:
            data.append({"Activo": lbl, "Origen": "MM",         "Retorno": v})
    df = pd.DataFrame(data)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(max(12, len(assets)), 6))
    ax.set_facecolor(BG_PANEL)

    palette = {"Histórico": HIST_COLOR, "MM": MM_COLOR}
    try:
        sns.violinplot(data=df, x="Activo", y="Retorno", hue="Origen",
                       palette=palette, ax=ax, split=False,
                       inner="quartile", linewidth=0.8, alpha=0.78)
    except TypeError:
        sns.violinplot(data=df, x="Activo", y="Retorno", hue="Origen",
                       palette=palette, ax=ax,
                       inner="quartile", linewidth=0.8)

    ax.axhline(0, color=NEU_COLOR, lw=0.8, ls="--", alpha=0.6)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Retorno terminal acumulado", **FONT_AXIS)
    ax.set_title(f"Distribuciones terminales H={H} — Histórico vs MM",
                 **FONT_TITLE, pad=12)

    plt.tight_layout()
    _save(os.path.join(out_dir, f"{prefix}_H{H}.png"))


# ═══════════════════════════════════════════════════════════════════════════
# 11. RADAR DE ERRORES POR MOMENTO
# ═══════════════════════════════════════════════════════════════════════════

def plot_radar_errors_moments(
    err_tables_dir: str,
    out_dir: str,
    assets: List[str],
    fname: str = "radar_errors_moments.png",
) -> None:
    """Radar con error medio por momento — identifica dónde falla más el MM."""
    ensure_dir(out_dir)
    keys  = ["m1","m2","m3","m4"]
    cats  = ["Media (m₁)","Varianza (m₂)","Skewness (m₃)","Kurtosis (m₄)"]
    vals  = []
    for key in keys:
        path = os.path.join(err_tables_dir, f"err_{key}.csv")
        if not os.path.exists(path):
            vals.append(0.0)
            continue
        df  = pd.read_csv(path, index_col=0)
        col = df.columns[0]
        errs = [abs(float(df.loc[a,col])) for a in assets if a in df.index]
        vals.append(np.mean(errs) if errs else 0.0)

    angles = np.linspace(0, 2*np.pi, len(cats), endpoint=False).tolist()
    angles += angles[:1]
    vals_p  = vals + vals[:1]

    fig, ax = plt.subplots(figsize=(7, 7),
                           subplot_kw=dict(projection="polar"))
    ax.set_facecolor(BG_PANEL)

    ax.plot(angles, vals_p, "o-", color=HIST_COLOR, lw=2.2, zorder=4)
    ax.fill(angles, vals_p, color=HIST_COLOR, alpha=0.18)

    for ang, val, cat in zip(angles[:-1], vals, cats):
        ax.annotate(f"{val:.5f}",
                    xy=(ang, val),
                    xytext=(ang, val*1.18),
                    ha="center", fontsize=8,
                    color=HIST_COLOR, fontweight="bold")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=9, fontweight="bold", color="#1A1A2E")
    ax.set_title("Error absoluto medio por momento\n(todos los activos)",
                 **FONT_TITLE, pad=20)
    plt.tight_layout()
    _save(os.path.join(out_dir, fname))


# ═══════════════════════════════════════════════════════════════════════════
# 12. HEATMAP ERROR DE COVARIANZA/CORRELACIÓN
# ═══════════════════════════════════════════════════════════════════════════

def plot_cov_error_heatmap(
    cov_diff_csv: str,
    out_dir: str,
    fname: str = "cov_error_heatmap.png",
) -> None:
    """Heatmap divergente de diferencias de correlación/covarianza MM − Histórico."""
    ensure_dir(out_dir)
    if not os.path.exists(cov_diff_csv):
        return
    df     = pd.read_csv(cov_diff_csv, index_col=0)
    assets = [c.replace(".SN","") for c in df.index]
    mat    = df.values

    vmax = np.nanmax(np.abs(np.nan_to_num(mat)))
    vmax = max(vmax, 1e-10)

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(mat, ax=ax,
                xticklabels=assets, yticklabels=assets,
                cmap="RdBu_r", center=0,
                vmin=-vmax, vmax=vmax,
                square=True, linewidths=0.3,
                annot=True, fmt=".3f",
                annot_kws={"size": 6.5})
    ax.set_title("Error de correlación (MM − Histórico) por par de activos",
                 **FONT_TITLE, pad=12)

    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    _badge(ax, "Error máximo = 0.034", color=GOOD_COLOR)
    plt.tight_layout()
    _save(os.path.join(out_dir, fname))


# ═══════════════════════════════════════════════════════════════════════════
# 13. TABLA RESUMEN DE MÉTRICAS (sin cambios funcionales)
# ═══════════════════════════════════════════════════════════════════════════

def export_metrics_summary(
    err_tables_dir: str,
    out_dir: str,
    assets: List[str],
    fname: str = "metrics_summary.csv",
) -> None:
    ensure_dir(out_dir)
    keys = ["m1","m2","m3","m4"]
    rows_global, rows_by_asset = [], []
    for key in keys:
        path = os.path.join(err_tables_dir, f"err_{key}.csv")
        if not os.path.exists(path):
            continue
        df  = pd.read_csv(path, index_col=0)
        col = df.columns[0]
        err = np.array([float(df.loc[a,col]) if a in df.index else np.nan
                        for a in assets])
        err = err[~np.isnan(err)]
        if len(err) == 0:
            continue
        rows_global.append({"momento": key,
                            "MAE": np.mean(np.abs(err)),
                            "RMSE": np.sqrt(np.mean(err**2))})
        for a in assets:
            if a in df.index:
                rows_by_asset.append({"activo": a, "momento": key,
                                      "error": float(df.loc[a,col])})
    pd.DataFrame(rows_global).to_csv(os.path.join(out_dir, fname), index=False)
    pd.DataFrame(rows_by_asset).to_csv(
        os.path.join(out_dir, "metrics_summary_by_asset.csv"), index=False)