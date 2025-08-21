# Medical Malpractice Claims — Interactive Dashboard (Dash/Plotly)

The project includes an exploratory data analysis and
an interactive dashboard for exploring medical malpractice claims at scale.

**Data source (not included):**
Kaggle — *Medical Malpractice Insurance Dataset*  
https://www.kaggle.com/datasets/gabrielsantello/medical-malpractice-insurance-dataset

Download the CSV/XLSX and put it in `data/` (recommended name: `medicalmalpractice.csv`).

## Setup (Python 3.8)
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r malpractice_requirements.txt
python src/app.py
```
Open http://127.0.0.1:8050

## Notes
- **Large data friendly**: k-means fit uses ≤20k rows; silhouette uses ≤5k rows. Tune
  `MAX_ROWS_KMEANS` / `MAX_ROWS_SILH` near the top of `src/app.py` if needed.
- **Figures auto-save**: PNGs are written to `assets/` via Plotly (kaleido) and Matplotlib (Agg).
- **Dashboard Tabs**: Overview, EDA (hist/box/QQ/corr), Clustering (k-means with PCA), Regression (OLS).
- This repo **does not redistribute** the dataset. Please follow the dataset's terms of use on Kaggle.
