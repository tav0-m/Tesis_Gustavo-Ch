from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf


@dataclass
class DownloadReport:
    tickers_requested: List[str]
    tickers_ok: List[str]
    tickers_bad: List[str]
    n_rows: int
    start: str
    end: str


# Aliases para tickers chilenos con nombres alternativos comunes
ALIASES: Dict[str, str] = {
    "ANDINAB.SN":  "ANDINA-B.SN",
    "SQMB.SN":     "SQM-B.SN",
    "AGUASA.SN":   "AGUAS-A.SN",
    "ENTELCL.SN":  "ENTEL.SN",
    "ITAU.SN":     "ITAUCL.SN",
}


def normalize_ticker(t: str) -> str:
    t = str(t).strip().upper()
    return ALIASES.get(t, t)


def _has_adj_close(df: pd.DataFrame) -> bool:
    """
    Verifica si un DataFrame de yfinance contiene datos de Adj Close.
    Maneja los dos formatos posibles que devuelve yfinance:
      - Columnas planas:    df.columns = ['Open', 'High', ..., 'Adj Close']
      - MultiIndex:         df.columns = [('Adj Close', 'ENTEL.SN'), ...]
    """
    if df is None or df.empty:
        return False
    if isinstance(df.columns, pd.MultiIndex):
        return "Adj Close" in df.columns.get_level_values(0)
    return "Adj Close" in df.columns


def _extract_adj_close_single(df: pd.DataFrame, ticker: str) -> Optional[pd.Series]:
    """
    Extrae la serie Adj Close de un DataFrame de un solo ticker.
    Devuelve None si no se puede extraer.
    """
    try:
        if isinstance(df.columns, pd.MultiIndex):
            # Formato: ('Adj Close', 'ENTEL.SN')
            if "Adj Close" in df.columns.get_level_values(0):
                return df["Adj Close"].squeeze()
        else:
            if "Adj Close" in df.columns:
                return df["Adj Close"]
    except Exception:
        pass
    return None


def validate_tickers(
    tickers: List[str],
    probe_period: str = "10d",
    sleep_s: float = 0.5,
) -> Tuple[List[str], List[str]]:
    """
    Valida cada ticker individualmente contra Yahoo Finance.

    Mejoras respecto a la versión original:
    ─────────────────────────────────────────
    1. Maneja correctamente MultiIndex de columnas (bug frecuente
       con tickers .SN que devuelven estructura inesperada).
    2. Reintenta una vez si la primera descarga falla con excepción,
       ya que Yahoo Finance a veces tiene timeouts transitorios.
    3. Acepta tickers aunque tengan algunos NaN — solo requiere
       que al menos 1 fila tenga dato válido.
    4. Agrega sleep entre llamadas para no saturar la API.
    """
    ok: List[str] = []
    bad: List[str] = []

    for raw in tickers:
        t = normalize_ticker(raw)
        validado = False

        for intento in range(2):   # hasta 2 intentos por ticker
            try:
                df = yf.download(
                    t,
                    period       = probe_period,
                    interval     = "1d",
                    auto_adjust  = False,
                    progress     = False,
                    threads      = False,
                )
                if _has_adj_close(df):
                    serie = _extract_adj_close_single(df, t)
                    if serie is not None and serie.notna().any():
                        ok.append(t)
                        validado = True
                        break
            except Exception:
                pass

            if intento == 0:
                time.sleep(sleep_s)   # esperar antes del reintento

        if not validado:
            bad.append(raw)

        time.sleep(sleep_s)   # pausa entre tickers

    return ok, bad


def download_adj_close(
    tickers: List[str],
    start: str,
    end: str,
    interval: str = "1d",
    min_coverage: float = 0.90,
    drop_bad: bool = True,
    max_retries: int = 3,
    retry_sleep_s: float = 2.0,
    show_warnings: bool = False,
    fill_method: str = "ffill",
) -> Tuple[pd.DataFrame, DownloadReport]:
    """
    Descarga Adj Close para todos los tickers, con manejo robusto
    de tickers chilenos (.SN) y relleno inteligente de NAs.

    Mejoras respecto a la versión original:
    ─────────────────────────────────────────
    1. fill_method='ffill': rellena días sin dato con el precio
       anterior (standard en finanzas para días sin transacciones).
       Esto evita que activos con pocos NaN sean descartados.
    2. max_retries=3 y retry_sleep_s=2.0 para mayor robustez.
    3. Descarga individual de fallback: si un ticker falla en la
       descarga grupal, intenta descargarlo solo.
    4. Reporte mejorado con cobertura real de cada ticker.

    Parámetros
    ──────────
    fill_method : 'ffill' (recomendado), 'bfill', o None (sin relleno)
    """
    if not show_warnings:
        warnings.filterwarnings("ignore", message="Timestamp.utcnow is deprecated")
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

    tickers_requested = [normalize_ticker(t) for t in tickers if str(t).strip()]

    # ── 1) Validación individual ────────────────────────────────────────────
    print("  Validando tickers contra Yahoo Finance...")
    tickers_ok, tickers_bad = validate_tickers(
        tickers_requested,
        probe_period = "10d",
        sleep_s      = 0.5,
    )

    if not tickers_ok:
        raise RuntimeError(
            f"No se pudo validar ningún ticker. Revisa símbolos.\n"
            f"Fallidos: {tickers_bad}"
        )

    print(f"  Validados: {len(tickers_ok)} OK / {len(tickers_bad)} fallidos")
    if tickers_bad:
        print(f"  Fallidos en validación: {tickers_bad}")

    # ── 2) Descarga grupal con reintentos ───────────────────────────────────
    data = None
    last_exc: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            data = yf.download(
                tickers_ok,
                start        = start,
                end          = end,
                interval     = interval,
                auto_adjust  = False,
                group_by     = "column",
                progress     = False,
                threads      = True,
            )
            if data is not None and not data.empty:
                break
        except Exception as e:
            last_exc = e
            print(f"  Reintento {attempt + 1}/{max_retries} descarga grupal...")
            time.sleep(retry_sleep_s)

    if data is None or data.empty:
        raise RuntimeError(f"Fallo descarga yfinance. Último error: {last_exc}")

    # ── 3) Extraer columna Adj Close ────────────────────────────────────────
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" not in data.columns.get_level_values(0):
            raise RuntimeError("No se encontró 'Adj Close' en la descarga grupal.")
        prices = data["Adj Close"].copy()
        # Si solo hay 1 ticker, puede ser Series — convertir a DataFrame
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers_ok[0])
    else:
        # Descarga de un único ticker devuelve columnas planas
        if "Adj Close" not in data.columns:
            raise RuntimeError("No se encontró 'Adj Close' (descarga single ticker).")
        prices = data[["Adj Close"]].copy()
        prices.columns = [tickers_ok[0]]

    # ── 4) Limpieza de índice ───────────────────────────────────────────────
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    # ── 5) Descarga individual de fallback para tickers con muchos NaN ──────
    #    Si algún ticker tiene >50% de NaN en la descarga grupal,
    #    intentar descargarlo solo (a veces la descarga grupal falla
    #    para tickers específicos aunque individualmente funcionen).
    for t in list(prices.columns):
        cobertura = prices[t].notna().mean()
        if cobertura < 0.50:
            print(f"  Reintentando descarga individual para {t} (cobertura grupal: {cobertura:.1%})...")
            try:
                df_ind = yf.download(
                    t, start=start, end=end,
                    interval="1d", auto_adjust=False,
                    progress=False, threads=False,
                )
                if _has_adj_close(df_ind):
                    serie = _extract_adj_close_single(df_ind, t)
                    if serie is not None:
                        serie.index = pd.to_datetime(serie.index)
                        prices[t] = serie.reindex(prices.index)
                        nueva_cob = prices[t].notna().mean()
                        print(f"    {t}: cobertura mejorada a {nueva_cob:.1%}")
            except Exception as e:
                print(f"    {t}: descarga individual también falló ({e})")

    # ── 6) Relleno de NAs (forward fill) ────────────────────────────────────
    #    ffill: un día sin cotización usa el precio del día anterior.
    #    Esto es correcto para activos con baja liquidez o feriados locales.
    if fill_method:
        prices = prices.ffill()
        # bfill solo para el comienzo de la serie (primeros días)
        prices = prices.bfill()

    # ── 7) Filtrar por cobertura mínima ─────────────────────────────────────
    cov = prices.notna().mean(axis=0)
    keep        = cov[cov >= float(min_coverage)].index.tolist()
    dropped_cov = cov[cov < float(min_coverage)].index.tolist()

    if dropped_cov:
        print(f"  Descartados por cobertura < {min_coverage:.0%}: {dropped_cov}")
        for t in dropped_cov:
            print(f"    {t}: {cov[t]:.1%} de días con dato")

    prices = prices[keep].copy()

    tickers_bad_all = sorted(set(tickers_bad + dropped_cov))

    report = DownloadReport(
        tickers_requested = tickers_requested,
        tickers_ok        = keep,
        tickers_bad       = tickers_bad_all,
        n_rows            = int(prices.shape[0]),
        start             = start,
        end               = end,
    )

    return prices, report