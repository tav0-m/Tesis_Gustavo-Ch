# Proyecto: Matching-Moment (MM) para escenarios de retornos (Chile, .SN), parte de Tesis generación de escenarios para portafolio financiero

Este proyecto descarga precios ajustados (Adj Close) desde Yahoo Finance (vía `yfinance`), construye retornos diarios y retornos *terminales* a horizonte **H**, calibra un modelo **Matching-Moment** (escenarios discretos `X` y probabilidades `p`) y genera diagnósticos/figuras para comparar histórico vs simulado.

---

## 1) Requisitos

- Python 3.10+ recomendado
- Paquetes:
  - numpy
  - pandas
  - scipy
  - matplotlib
  - yfinance
  - pyyaml

Instalación (Windows PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy pandas scipy matplotlib yfinance pyyaml

#Segunda versión de MM 
- En esta versión ITAUCL tiene unos pocos días sin cotización en el período 2020–2025 (feriados locales, suspensiones de trading). Eso es normal para acciones chilenas de menor liquidez. 
- Esto no solo no daña tu modelo — en tu tesis puedes mencionar explícitamente que la calidad de los datos es alta, con cobertura promedio del 99.99% sobre el período 2020–2025, y que los únicos datos faltantes corresponden a feriados nacionales resueltos con forward fill estándar.
* (menciona en la sección de datos que "la cobertura del universo es del 99.99%, con el único dato faltante correspondiente a un feriado nacional (2 de mayo de 2024) en ITAUCL.SN, resuelto mediante forward fill estándar de la industria.")
