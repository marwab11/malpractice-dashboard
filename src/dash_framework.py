# app.py — Medical Malpractice Dashboard (PCA-safe KMeans)
# Author: Marwa Bahr (2025)

from pathlib import Path
import os
import numpy as np
import pandas as pd

from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score, r2_score

from statsmodels.graphics.gofplots import qqplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

def load_dataset(preferred="medicalmalpractice.csv"):
    cands = [DATA_DIR/preferred] + sorted(DATA_DIR.glob("*.csv")) + sorted(DATA_DIR.glob("*.xlsx")) + \
            [BASE_DIR/preferred] + sorted(BASE_DIR.glob("*.csv")) + sorted(BASE_DIR.glob("*.xlsx"))
    for fp in cands:
        if fp.exists():
            print(f"[INFO] Loading dataset: {fp}")
            if fp.suffix.lower() == ".xlsx":
                return pd.read_excel(fp)
            return pd.read_csv(fp)
    raise FileNotFoundError(f"Put a CSV/XLSX in {DATA_DIR} (e.g., {DATA_DIR/'medicalmalpractice.csv'}).")

def numeric_columns(df): return df.select_dtypes(include=[np.number]).columns.tolist()
def categorical_columns(df): 
    nums=set(numeric_columns(df)); 
    return [c for c in df.columns if c not in nums]

def save_plotly(fig, name):
    try:
        fig.write_image(str(ASSETS_DIR/name), scale=2)
    except Exception as e:
        print(f"[WARN] Could not save {name}: {e}")

def save_matplotlib(fig, name):
    try:
        fig.savefig(str(ASSETS_DIR/name), dpi=200, bbox_inches="tight"); plt.close(fig)
    except Exception as e:
        print(f"[WARN] Could not save {name}: {e}")

# ---------- Data ----------
df = load_dataset()
df = df.apply(lambda s: pd.to_numeric(s, errors="ignore")).dropna(axis=1, how="all")
df = df.rename(columns=lambda c: str(c).strip())

num_cols = numeric_columns(df)
cat_cols = categorical_columns(df)
if not num_cols:
    raise ValueError("No numeric columns detected.")

# ---------- Perf knobs ----------
MAX_ROWS_KMEANS = 20000
MAX_ROWS_SILH = 5000
RANDOM_SEED = 42

# ---------- App ----------
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Medical Malpractice Claims — Dashboard"

def card(h, v):
    return html.Div([html.Div(h, style={"color":"#6b7280"}),
                     html.Div(v, style={"fontWeight":"600", "fontSize":"18px"})],
                    style={"padding":"12px 16px","border":"1px solid #eee","borderRadius":"12px","background":"#fff","boxShadow":"0 1px 3px rgba(0,0,0,0.05)"})

app.layout = html.Div([
    html.H1("Medical Malpractice Claims — Interactive Dashboard"),
    html.Div([card("Rows", f"{len(df):,}"),
              card("Columns", f"{df.shape[1]:,}"),
              card("Numeric cols", f"{len(num_cols)}"),
              card("Categorical cols", f"{len(cat_cols)}")],
             style={"display":"grid","gridTemplateColumns":"repeat(4,1fr)","gap":"12px","marginBottom":"16px"}),
    dcc.Tabs(id="tabs", value="tab-eda", children=[
        dcc.Tab(label="Overview", value="tab-overview"),
        dcc.Tab(label="EDA", value="tab-eda"),
        dcc.Tab(label="Clustering (k-means)", value="tab-kmeans"),
        dcc.Tab(label="Regression", value="tab-reg"),
        dcc.Tab(label="Downloads", value="tab-dl"),
    ]),
    html.Div(id="tab-content", style={"marginTop":"16px"}),
], style={"maxWidth":"1100px","margin":"0 auto","padding":"16px 20px","fontFamily":"system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif","background":"#fafafa"})

def layout_overview():
    return html.Div([
        html.H3("Dataset preview"),
        dcc.Graph(figure=px.imshow(df[num_cols].corr(), text_auto=False, title="Correlation heatmap (numeric)")),
        html.Pre(df.head(10).to_string(index=False), style={"whiteSpace":"pre-wrap","background":"#fff","padding":"12px","borderRadius":"12px","border":"1px solid #eee"}),
    ])

def layout_eda():
    default_num = num_cols[0]
    default_color = cat_cols[0] if cat_cols else None
    controls = html.Div([
        html.Div([html.Label("Numeric column"),
                  dcc.Dropdown(options=[{"label":c,"value":c} for c in num_cols], value=default_num, id="eda-num")], style={"flex":"1"}),
        html.Div([html.Label("Color (optional, categorical)"),
                  dcc.Dropdown(options=[{"label":c,"value":c} for c in cat_cols], value=default_color, id="eda-color", clearable=True)], style={"flex":"1","marginLeft":"12px"}),
    ], style={"display":"flex","marginBottom":"12px"})
    return html.Div([controls,
                     dcc.Graph(id="hist-fig"),
                     dcc.Graph(id="box-fig"),
                     dcc.Graph(id="qq-fig"),
                     dcc.Graph(id="corr-fig")])

def layout_kmeans():
    default_feats = num_cols[:3]
    return html.Div([
        html.Div([
            html.Div([html.Label("Features"),
                      dcc.Dropdown(options=[{"label":c,"value":c} for c in num_cols],
                                   value=default_feats, id="km-feats", multi=True)], style={"flex":"2"}),
            html.Div([html.Label("k (clusters)"),
                      dcc.Slider(id="km-k", min=2, max=8, step=1, value=3, marks={i:str(i) for i in range(2,9)})],
                     style={"flex":"1","padding":"8px 12px"}),
        ], style={"display":"flex","gap":"12px","marginBottom":"8px"}),
        dcc.Loading(children=[dcc.Graph(id="km-fig")], type="dot"),
        html.Div(id="km-metrics", style={"marginTop":"6px"}),
    ])

def layout_reg():
    default_target = num_cols[0]
    default_feats = [c for c in num_cols if c != default_target][:3]
    return html.Div([
        html.Div([
            html.Div([html.Label("Target (numeric)"),
                      dcc.Dropdown(options=[{"label":c,"value":c} for c in num_cols],
                                   value=default_target, id="reg-target")], style={"flex":"1"}),
            html.Div([html.Label("Features"),
                      dcc.Dropdown(options=[{"label":c,"value":c} for c in num_cols],
                                   value=default_feats, id="reg-feats", multi=True)], style={"flex":"2","marginLeft":"12px"}),
        ], style={"display":"flex","gap":"12px","marginBottom":"8px"}),
        dcc.Graph(id="reg-scatter"),
        html.Pre(id="reg-summary", style={"whiteSpace":"pre-wrap","background":"#fff","padding":"12px","borderRadius":"12px","border":"1px solid #eee"}),
    ])

def layout_downloads():
    return html.Div([
        html.P("Download cleaned dataset and the latest generated figures."),
        html.Button("Download cleaned CSV", id="dl-clean-btn"),
        dcc.Download(id="dl-clean"),
        html.Div(style={"height":"10px"}),
        dcc.Markdown("Saved images (in `assets/`):"),
        html.Ul([html.Li(name) for name in sorted([p.name for p in ASSETS_DIR.glob('*.png')]) or ["(none yet — interact with charts to save)"]])
    ])

@app.callback(Output("tab-content","children"), Input("tabs","value"))
def render_tab(tab):
    return {"tab-overview": layout_overview(),
            "tab-eda": layout_eda(),
            "tab-kmeans": layout_kmeans(),
            "tab-reg": layout_reg(),
            "tab-dl": layout_downloads()}.get(tab, html.Div("Unknown tab."))

# ---------- EDA ----------
@app.callback(
    Output("hist-fig","figure"),
    Output("box-fig","figure"),
    Output("qq-fig","figure"),
    Output("corr-fig","figure"),
    Input("eda-num","value"),
    Input("eda-color","value"),
)
def update_eda(num_col, color_col):
    if num_col is None or num_col not in df.columns: num_col = num_cols[0]
    hist_fig = px.histogram(df, x=num_col, color=color_col, nbins=40, marginal="box", title=f"Distribution of {num_col}")
    save_plotly(hist_fig, f"hist_{num_col}.png")

    if color_col and color_col in df.columns and color_col not in num_cols:
        box_fig = px.box(df, x=color_col, y=num_col, points="outliers", title=f"{num_col} by {color_col}")
    else:
        box_fig = px.box(df, y=num_col, points="outliers", title=f"{num_col} (box plot)")
    save_plotly(box_fig, f"box_{num_col}.png")

    fig = plt.figure()
    try:
        qqplot(df[num_col].dropna(), line="s", ax=plt.gca()); plt.title(f"Q-Q plot — {num_col}")
    except Exception as e:
        plt.text(0.5,0.5,f"Q-Q error: {e}", ha="center")
    save_matplotlib(fig, f"qq_{num_col}.png")

    corr = df[num_cols].corr()
    corr_fig = px.imshow(corr, text_auto=False, title="Correlation heatmap (numeric)")
    save_plotly(corr_fig, "corr_heatmap.png")
    return hist_fig, box_fig, go.Figure(), corr_fig

# ---------- KMeans (PCA-safe, sampled) ----------
@app.callback(
    Output("km-fig","figure"),
    Output("km-metrics","children"),
    Input("km-feats","value"),
    Input("km-k","value"),
)
def run_kmeans(cols, k):
    if not cols:
        return px.scatter(title="Select at least one feature."), "Select at least one feature."
    if len(cols) > 6:
        cols = cols[:6]

    X_full = df[cols].dropna().to_numpy()
    n_full, d = X_full.shape
    if n_full < int(k):
        return px.scatter(title=f"Not enough rows after dropna for k={k}."), f"Not enough rows after dropna for k={k}."

    # Sample for speed
    rng = np.random.RandomState(RANDOM_SEED)
    fit_n = min(MAX_ROWS_KMEANS, n_full)
    idx_fit = rng.choice(n_full, size=fit_n, replace=False) if n_full > fit_n else np.arange(n_full)
    X = X_full[idx_fit]

    # Scale
    Xs = StandardScaler().fit_transform(X)

    # Choose PCA components safely
    max_components = min(2, Xs.shape[0], Xs.shape[1])
    n_comp = max(1, max_components)  # 1 or 2
    pca = PCA(n_components=n_comp, random_state=RANDOM_SEED, svd_solver="auto")
    Xp = pca.fit_transform(Xs)

    # KMeans
    kmeans = KMeans(n_clusters=int(k), random_state=RANDOM_SEED, n_init=10)
    labels = kmeans.fit_predict(Xs)

    # Silhouette sample
    sil_n = min(MAX_ROWS_SILH, Xs.shape[0])
    idx_sil = rng.choice(Xs.shape[0], size=sil_n, replace=False) if Xs.shape[0] > sil_n else np.arange(Xs.shape[0])
    sil = silhouette_score(Xs[idx_sil], labels[idx_sil])

    # Plot (2D or 1D fallback)
    if n_comp == 2:
        fig = px.scatter(x=Xp[:,0], y=Xp[:,1], color=labels.astype(str),
                         title=f"KMeans (k={k}) — PCA projection",
                         labels={"x":"PC1","y":"PC2","color":"cluster"})
    else:
        fig = px.scatter(x=Xp[:,0], y=np.zeros_like(Xp[:,0]), color=labels.astype(str),
                         title=f"KMeans (k={k}) — PCA(1D) projection",
                         labels={"x":"PC1","y":"", "color":"cluster"})
    save_plotly(fig, f"kmeans_k{k}_fit{fit_n}_sil{sil_n}.png")

    msg = (f"Rows used for fit: **{fit_n:,}** of {n_full:,}  •  "
           f"Rows used for silhouette: **{sil_n:,}**  •  "
           f"Silhouette score: **{sil:.3f}**  •  "
           f"PCA dims: {n_comp} (explained variance: " +
           ", ".join(f"{v:.2f}" for v in (pca.explained_variance_ratio_ if n_comp>1 else [pca.explained_variance_ratio_])) + ")")
    return fig, dcc.Markdown(msg)

# ---------- Regression ----------
@app.callback(
    Output("reg-scatter","figure"),
    Output("reg-summary","children"),
    Input("reg-target","value"),
    Input("reg-feats","value"),
)
def run_regression(target, feats):
    if target is None or not feats:
        return go.Figure(), "Pick a target and at least one feature."
    feats = [c for c in feats if c != target]
    X = df[feats].dropna().to_numpy()
    y = df.loc[df[feats].dropna().index, target].to_numpy().reshape(-1,1)
    if len(y) == 0:
        return go.Figure(), "No rows left after dropping NAs."

    model = LinearRegression().fit(X, y)
    yhat = model.predict(X).ravel()
    r2 = r2_score(y, yhat)

    fig = px.scatter(x=y.ravel(), y=yhat, labels={"x":"Actual","y":"Predicted"}, title=f"Linear Regression — R²={r2:.3f}")
    fig.add_trace(go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()], mode="lines", name="ideal"))
    save_plotly(fig, f"reg_{target}.png")

    coef_table = pd.DataFrame({"feature": feats, "coef": model.coef_.ravel()}).sort_values("coef", key=abs, ascending=False)
    summary = f"Intercept: {model.intercept_.item():.4f}\n" + coef_table.to_string(index=False)
    return fig, summary

@app.callback(Output("dl-clean","data"), Input("dl-clean-btn","n_clicks"), prevent_initial_call=True)
def download_clean(n):
    clean_path = ASSETS_DIR / "cleaned.csv"
    df.to_csv(clean_path, index=False)
    return dcc.send_file(str(clean_path))

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
