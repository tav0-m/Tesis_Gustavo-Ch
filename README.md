# Tesis

Repositorio de apoyo para la tesis **“Generación de Escenarios Financieros para el IPSA mediante el método Matching-Moment con BCD: Optimización de Portafolios bajo criterios de Markowitz y CVaR”**. Este proyecto desarrolla un pipeline reproducible en Python para descargar, transformar y modelar datos financieros del mercado accionario chileno, con foco en la construcción de escenarios discretos que representen de forma más realista la distribución de retornos de los activos del IPSA.

A diferencia de enfoques tradicionales basados en supuestos de normalidad, este trabajo busca capturar propiedades empíricas relevantes de los retornos financieros, como la asimetría, la curtosis y la dependencia entre activos. Para ello, se implementa el método **Matching-Moment (MM)** optimizado mediante **Block Coordinate Descent (BCD)**, generando escenarios y probabilidades que luego sirven como base para el análisis de riesgo y futuras aplicaciones de optimización de portafolios bajo criterios como **Markowitz** y **CVaR**.

---

## Objetivo del proyecto

El objetivo principal es calibrar un modelo de generación de escenarios financieros sobre un universo de activos del IPSA para replicar, con la mayor fidelidad posible, las principales características estadísticas observadas en los retornos históricos.

En particular, el proyecto busca representar adecuadamente:

- media
- varianza
- asimetría
- curtosis
- estructura de correlación entre activos

Además, este repositorio funciona como respaldo computacional de la tesis, permitiendo reproducir resultados, revisar métricas de ajuste y extender el análisis hacia modelos comparativos y optimización de portafolios.

---

## Contexto de la tesis

La tesis utiliza precios ajustados de **15 activos del IPSA** para el período **2020–2025**. A partir de esta base, se construyen retornos diarios y retornos terminales acumulados a un horizonte de **5 días hábiles (H = 5)**, que constituyen la base estadística para la calibración del modelo.

La configuración principal del modelo considera:

- **15 activos**
- **200 escenarios discretos**
- **5 inicializaciones**
- **150 iteraciones máximas**
- horizonte terminal de **5 días**

---

## Metodología general

El flujo del proyecto se divide en cuatro etapas principales:

1. **Extracción y preparación de datos**
   - descarga de precios
   - limpieza de series
   - cálculo de retornos diarios
   - construcción de retornos terminales

2. **Construcción de objetivos estadísticos**
   - momentos históricos
   - matrices de covarianza y correlación
   - métricas de comparación

3. **Calibración del modelo MM**
   - generación de escenarios discretos
   - optimización de escenarios y probabilidades
   - uso de BCD con estrategia multi-start

4. **Diagnóstico y visualización**
   - medición de errores por momento
   - comparación histórico vs. simulado
   - análisis de distribuciones
   - análisis de correlación y covarianza
   - generación de gráficos y tablas resumen

---

## Estructura del proyecto

```bash
Tesis_Gustavo-Ch/
│
├── .venv/
│
├── outputs/
│   └── figures/
│       ├── convergence_bcd.png
│       ├── corr_combinado.png
│       ├── corr_diff_1.png
│       ├── corr_diff.csv
│       ├── corr_hist_1.png
│       ├── corr_hist.csv
│       ├── corr_mm_1.png
│       ├── corr_mm.csv
│       ├── cov_diff.csv
│       ├── cov_diff.png
│       ├── cov_error_heatmap.png
│       ├── cov_hist_1.png
│       ├── cov_hist.csv
│       ├── cov_mm.csv
│       ├── cov_mm.png
│       ├── errors_by_asset.png
│       ├── fan_chart.png
│       ├── hist_grid_H5.png
│       ├── hist_terminal_H5_*.png
│       └── ...
│
├── tables/
│   ├── err_m1.csv
│   ├── err_m2.csv
│   ├── err_m3.csv
│   ├── err_m4.csv
│   ├── metrics_summary.csv
│   └── metrics_summary_by_asset.csv
│
├── src/
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── diagnostics_plots.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py
│   │   └── transform.py
│   │
│   └── mm/
│       ├── __init__.py
│       ├── bcd.py
│       ├── diagnostics.py
│       ├── objective.py
│       └── targets.py
│
├── adj_close_prices.csv
├── daily_returns.csv
├── hist_terminal_returns_H5.csv
├── mm_probabilities_p.csv
├── mm_scenarios_x.csv
├── objective_history.csv
├── terminal_returns_H5.csv
│
├── config.yaml
├── generate_plots.py
├── requirements.txt
├── run.py
└── README.md
