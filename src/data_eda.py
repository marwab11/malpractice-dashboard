# analysis_static.py  — your original analysis, with ALL figures saved to assets/
# Notes:
# - Uses a non-interactive backend (Agg) so this runs in terminals/servers.
# - Creates assets/ if missing and saves each plot with a clear filename.
# - No Dash here (we’ll move the Dash app to a separate file next).

import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

# Use non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D proj)
from prettytable import PrettyTable
from statsmodels.graphics.gofplots import qqplot
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import shapiro, pearsonr, ttest_ind
from pingouin import partial_corr

# --------------------------
# Global setup & paths
# --------------------------
np.random.seed(5764)

BASE_DIR = Path(__file__).resolve().parent.parent      # .../malpractice-dashboard
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

print(f"[paths] DATA_DIR = {DATA_DIR}")
print(f"[paths] ASSETS_DIR = {ASSETS_DIR}")

# Helpers to save figures/grids consistently
def save_current(name: str):
    """Save the current Matplotlib figure and close it."""
    out = ASSETS_DIR / name
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")

def save_fig(fig, name: str):
    """Save a specific Matplotlib figure instance and close it."""
    out = ASSETS_DIR / name
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")

def save_grid(grid, name: str):
    """Save seaborn FacetGrid/JointGrid/ClusterGrid objects."""
    out = ASSETS_DIR / name
    grid.fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(grid.fig)
    print(f"[saved] {out}")

# --------------------------
# Robust data loader
# --------------------------
def load_dataset(preferred="medicalmalpractice.csv"):
    candidates = []
    candidates.append(DATA_DIR / preferred)
    candidates += sorted(DATA_DIR.glob("*.csv"))
    candidates += sorted(DATA_DIR.glob("*.xlsx"))
    candidates.append(BASE_DIR / preferred)
    candidates += sorted(BASE_DIR.glob("*.csv"))
    candidates += sorted(BASE_DIR.glob("*.xlsx"))

    for fp in candidates:
        if fp.exists():
            print(f"[INFO] Loading data from: {fp}")
            if fp.suffix.lower() == ".xlsx":
                return pd.read_excel(fp)
            return pd.read_csv(fp)

    raise FileNotFoundError(
        f"No dataset found. Put your file in {DATA_DIR} (e.g., {DATA_DIR / 'medicalmalpractice.csv'})."
    )

df = load_dataset()

# --------------------------
# Quick peek
# --------------------------
print("Dataset (Not Cleaned):")
print(df.head(), "\n")
print("Dataset Info:")
print(df.info(), "\n")

# Take first 5 columns as numeric preview (adjust if needed)
df_columns = df.columns[:5]
df_numeric = df[df_columns].apply(pd.to_numeric, errors="coerce")
print("Dataset Description:")
print(df_numeric.describe(), "\n")

# Safe random sample (max 1000 or dataset size)
sample_n = min(1000, len(df))
df_subset_1000 = df.sample(n=sample_n, random_state=5764).reset_index(drop=True)

# These lists are for later (when we build the Dash UI)
drop1 = {c for c in ["Amount", "Age", "Specialty"] if c in df.columns}
drop2 = {c for c in ["Amount", "Age"] if c in df.columns}
category = [c for c in df.columns if c not in drop1]
category2 = [c for c in df.columns if c not in drop2]

sns.set_theme(style="darkgrid")

# --------------------------
# DATA CLEANING
# --------------------------
if df.isna().any().any() or df.isnull().any().any():
    df = df.dropna()
    print("Missing values were found and removed. Cleaned head:\n")
    print(round(df.head(), 2))
else:
    print("Dataset has no missing values found and is clean.\n")

# --------------------------
# OUTLIER Detection & Boxplot
# --------------------------
amount = df['Amount']
Q1 = np.percentile(amount, 25)
Q3 = np.percentile(amount, 75)
IQR = Q3 - Q1
low_outliers = Q1 - 1.5 * IQR
high_outliers = Q3 + 1.5 * IQR
print(f"Q1/Q3: ${Q1:.2f} / ${Q3:.2f} | IQR = ${IQR:.2f}")
print(f"Outlier threshold: < ${low_outliers:.2f} or > ${high_outliers:.2f}")

plt.figure(figsize=(8, 6))
sns.boxplot(x=amount)
plt.xlabel('Amount (USD)', {'fontname': 'serif', 'color': 'blue', 'size': 16})
plt.title('Boxplot of Amount', {'fontname': 'serif', 'color': 'darkred', 'size': 20})
plt.tight_layout()
save_current("boxplot_amount.png")

# --------------------------
# PCA (on selected columns)
# --------------------------
numerical_columns = ['Amount', 'Age']
selected_categorical_columns = ['Severity', 'Private Attorney', 'Marital Status']
subset_df = df[numerical_columns + selected_categorical_columns].copy()

# Standardize numerical features
scaler = StandardScaler()
subset_df[numerical_columns] = scaler.fit_transform(subset_df[numerical_columns])

# (Optionally) treat the categorical codes as numeric here
X = subset_df

scaler = StandardScaler()
X_std = scaler.fit_transform(X)
print("Standardized Feature Space (first 5 rows):")
print(pd.DataFrame(X_std, columns=X.columns).head().round(2))

pca = PCA(n_components=X_std.shape[1])
pca.fit(X_std)
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)
n_components_for_90_var = np.argmax(cumulative_var >= 0.90) + 1
features_to_remove = X.shape[1] - n_components_for_90_var
print(f"Number of features to remove for 90% variance: {features_to_remove}")

plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(explained_var) + 1, 1), 100 * cumulative_var,
         lw=3, marker='o', linestyle='-')
plt.axhline(y=90, color='black', linestyle='--', label='90% Explained Variance')
plt.axvline(x=n_components_for_90_var, color='red', linestyle='--',
            label=f'{n_components_for_90_var} Components')
plt.xlabel('Number of Components', {'fontname': 'serif', 'color': 'blue', 'size': 16})
plt.ylabel('Cumulative Explained Variance (%)', {'fontname': 'serif', 'color': 'blue', 'size': 16})
plt.title('Cumulative Explained Variance vs Components',
          {'fontname': 'serif', 'color': 'darkred', 'size': 20})
plt.legend(); plt.grid(True); plt.tight_layout()
save_current("pca_cumulative_explained_variance.png")

# --------------------------
# NORMALITY (Shapiro) + QQ plots
# --------------------------
def shapiro_test(x, title):
    stats, p = shapiro(x)
    print('=' * 50)
    print(f'Shapiro test — {title}: statistics = {stats:.3f}, p = {p:.3f}')
    print('Looks Normal' if p > 0.05 else 'Not Normal')

shapiro_test(df['Amount'], 'Amount')
shapiro_test(df['Age'], 'Age')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
qqplot(df['Amount'], line='s', color='blue', ax=axes[0])
axes[0].set_title('QQ Plot — Amount')
qqplot(df['Age'], line='s', color='blue', ax=axes[1])
axes[1].set_title('QQ Plot — Age')
plt.tight_layout()
save_fig(fig, "qqplots_amount_age.png")

# --------------------------
# Z-score transform + 2x2 plots
# --------------------------
mean_amount, std_amount = df['Amount'].mean(), df['Amount'].std()
mean_age, std_age = df['Age'].mean(), df['Age'].std()
df_subset_1000['z_score_amount'] = (df_subset_1000['Amount'] - mean_amount) / std_amount
df_subset_1000['z_score_age'] = (df_subset_1000['Age'] - mean_age) / std_age

plt.figure(figsize=(16, 10))

plt.subplot(2, 2, 1)
sns.lineplot(x=range(1, len(df_subset_1000) + 1), y=df_subset_1000['Amount'], label='Amount')
sns.lineplot(x=range(1, len(df_subset_1000) + 1), y=df_subset_1000['Age'], label='Age')
plt.title('Subset of Amount & Age (Random Sample)'); plt.legend()

plt.subplot(2, 2, 2)
sns.lineplot(x=range(1, len(df_subset_1000) + 1), y=df_subset_1000['z_score_amount'], label='Amount (z)')
sns.lineplot(x=range(1, len(df_subset_1000) + 1), y=df_subset_1000['z_score_age'], label='Age (z)')
plt.title('Transformed (z-score) — Amount & Age'); plt.legend()

plt.subplot(2, 2, 3)
sns.histplot(data=df_subset_1000, x='Amount', bins=20, alpha=0.7)
sns.histplot(data=df_subset_1000, x='Age', bins=20, alpha=0.7)
plt.title('Histogram — Amount & Age')

plt.subplot(2, 2, 4)
sns.histplot(data=df_subset_1000, x='z_score_amount', bins=20, alpha=0.7, label='Amount (z)')
sns.histplot(data=df_subset_1000, x='z_score_age', bins=20, alpha=0.7, label='Age (z)')
plt.title('Histogram — z-scores'); plt.legend()

plt.tight_layout()
save_current("zscore_and_histograms.png")

# --------------------------
# Pearson heatmap + pairplot
# --------------------------
corr = df_numeric.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Pearson Correlation Heatmap')
plt.tight_layout()
save_current("corr_heatmap.png")

pair_g = sns.pairplot(df_numeric.dropna(), height=2)
pair_g.fig.suptitle('Scatter Plot Matrix', y=1.02)
save_grid(pair_g, "pairplot_matrix.png")

# PrettyTable of correlations
table = PrettyTable()
table.title = 'Pearson Correlation Coefficients'
table.field_names = ['Variable'] + list(corr.columns)
for v1 in corr.index:
    row = [v1] + [f'{corr.loc[v1, v2]:.2f}' for v2 in corr.columns]
    table.add_row(row)
print("Pearson Correlation Coefficients:")
print(table)

# --------------------------
# T-tests & partial correlations
# --------------------------
corr_amt_pa, _ = pearsonr(df['Amount'], df['Private Attorney'])
print(f"Corr(Amount, Private Attorney) = {corr_amt_pa:.2f}")
t_stat, p_val = ttest_ind(df['Amount'], df['Private Attorney'])
print(f"T-stat = {t_stat:.2f}  |  p = {p_val:.2f}")

pc_amt_pa = partial_corr(df, x='Amount', y='Private Attorney', covar='Severity')['r'][0]
print(f"Partial corr(Amount, Private Attorney | Severity) = {pc_amt_pa:.2f}")

t_stat2, p_val2 = ttest_ind(df['Amount'], df['Private Attorney'])
print(f"T-stat (partial corr proxy) = {t_stat2:.2f}  |  p = {p_val2:.2f}")

corr_amt_sev, _ = pearsonr(df['Amount'], df['Severity'])
print(f"Corr(Amount, Severity) = {corr_amt_sev:.2f}")
t_stat3, p_val3 = ttest_ind(df['Amount'], df['Severity'])
print(f"T-stat = {t_stat3:.2f}  |  p = {p_val3:.2f}")

pc_amt_sev = partial_corr(df, x='Amount', y='Severity', covar='Private Attorney')['r'][0]
print(f"Partial corr(Amount, Severity | Private Attorney) = {pc_amt_sev:.2f}")

# --------------------------
# KDE pairplot & 2D KDE
# --------------------------
kde_g = sns.pairplot(df_subset_1000, kind="kde", diag_kind="kde")
kde_g.fig.suptitle('Pair Plot with KDE', y=1.02)
save_grid(kde_g, "pairplot_kde.png")

plt.figure(figsize=(10, 8))
sns.kdeplot(x='Amount', y='Age', data=df_subset_1000, fill=True, linewidth=0.3, cbar=True, alpha=0.6)
plt.title('2D KDE — Amount vs Age')
plt.tight_layout()
save_current("kde_amount_age.png")

# --------------------------
# Pie charts
# --------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
gender_counts = df['Gender'].value_counts()
axes[0].pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, explode=(0.01, 0.01))
axes[0].set_title('Gender Count'); axes[0].legend(gender_counts.index, loc='upper right')

pa_counts = df['Private Attorney'].value_counts()
axes[1].pie(pa_counts, labels=pa_counts.index, autopct='%1.1f%%', startangle=90, explode=(0.01, 0.01))
axes[1].set_title('Private Attorney Count'); axes[1].legend(pa_counts.index, loc='upper right')

ins_counts = df['Insurance'].value_counts()
axes[2].pie(ins_counts, labels=ins_counts.index, autopct='%1.1f%%', startangle=90)
axes[2].set_title('Insurance Type'); axes[2].legend(ins_counts.index, loc='upper left')

plt.tight_layout()
save_fig(fig, "piecharts_gender_attorney_insurance.png")

# --------------------------
# Mixed subplots (bar/hist/strip etc.)
# --------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 14))
sns.barplot(x='Severity', y='Amount', data=df, ax=axes[0, 0]); axes[0, 0].set_title('Amount by Severity')
sns.histplot(data=df, x='Amount', bins=10, hue='Private Attorney', ax=axes[0, 1], multiple='dodge', kde=True)
axes[0, 1].set_title('Amount by Private Attorney')
sns.stripplot(x='Marital Status', y='Amount', data=df, jitter=True, palette='viridis', ax=axes[0, 2])
axes[0, 2].set_title('Amount by Marital Status')
sns.barplot(y='Specialty', x='Amount', data=df, ax=axes[1, 0]); axes[1, 0].set_title('Amount by Specialty')
sns.histplot(data=df, x='Amount', bins=10, hue='Insurance', ax=axes[1, 1], multiple='stack', kde=True)
axes[1, 1].set_title('Amount by Insurance')
sns.histplot(data=df, x='Amount', bins=10, hue='Gender', ax=axes[1, 2], multiple='dodge', kde=True)
axes[1, 2].set_title('Amount by Gender')
plt.tight_layout()
save_fig(fig, "subplots_mixed.png")

# --------------------------
# Distribution (replaces deprecated distplot)
# --------------------------
plt.figure(figsize=(12, 6))
sns.histplot(df['Amount'], kde=True, bins=30, color='skyblue')
plt.title('Distribution of Amount')
plt.tight_layout()
save_current("dist_amount.png")

# --------------------------
# Count plot (Marital Status)
# --------------------------
marital_status_mapping = {0: 'Divorced', 1: 'Single', 2: 'Married', 3: 'Widowed', 4: 'Unknown'}
df['Marital Status Label'] = df['Marital Status'].map(marital_status_mapping)

plt.figure(figsize=(12, 6))
sns.countplot(y='Marital Status Label', data=df, palette='viridis')
plt.title('Count by Marital Status')
plt.tight_layout()
save_current("count_marital_status.png")

# --------------------------
# lmplot (regression)
# --------------------------
lm = sns.lmplot(data=df, x='Amount', y='Age', scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})
lm.fig.suptitle('Linear Regression: Age vs Amount', y=1.02)
save_grid(lm, "lm_amount_age.png")

# --------------------------
# Boxen plot
# --------------------------
plt.figure(figsize=(12, 8))
sns.boxenplot(data=df, x='Severity', y='Amount', hue='Private Attorney', palette='Set3')
plt.title('Amount by Severity × Attorney (boxen)')
plt.tight_layout()
save_current("boxen_severity_attorney.png")

# --------------------------
# Area plot
# --------------------------
area_plot_data = df.groupby(['Insurance', 'Private Attorney']).agg({'Amount': 'mean'}).reset_index()
pivoted = area_plot_data.pivot(index='Insurance', columns='Private Attorney', values='Amount').fillna(0)
ax = pivoted.plot.area(stacked=True, alpha=0.6, figsize=(12, 8))
ax.set_title('Mean Amount by Insurance × Attorney (stacked area)')
plt.tight_layout()
save_current("area_insurance_attorney.png")

# --------------------------
# Violin plot
# --------------------------
plt.figure(figsize=(12, 8))
sns.violinplot(data=df, x='Gender', y='Amount', inner='quartile', palette='muted')
plt.title('Amount by Gender (violin)')
plt.tight_layout()
save_current("violin_gender_amount.png")

# --------------------------
# Joint plot (scatter + KDE)
# --------------------------
jp = sns.jointplot(data=df, x='Amount', y='Age', hue='Private Attorney')
jp.fig.suptitle('Joint Plot — Amount vs Age', y=1.02)
save_grid(jp, "jointplot_amount_age.png")

# --------------------------
# Rug plot
# --------------------------
plt.figure(figsize=(12, 8))
sns.rugplot(data=df, x='Amount', height=0.5, color='coral')
plt.title('Rug Plot — Amount')
plt.tight_layout()
save_current("rug_amount.png")

# --------------------------
# 3D contour (toy example)
# --------------------------
x_values = np.linspace(df['Amount'].min(), df['Amount'].max(), 100)
y_values = np.linspace(df['Age'].min(), df['Age'].max(), 100)
Xg, Yg = np.meshgrid(x_values, y_values)
Z = np.exp(-(Xg - df['Amount'].mean()) ** 2 / (2 * df['Amount'].std() ** 2))
Z *= np.exp(-(Yg - df['Age'].mean()) ** 2 / (2 * df['Age'].std() ** 2))

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
for sev in sorted(df['Severity'].dropna().unique())[:9]:
    subset = df[df['Severity'] == sev]
    if subset.empty:  # safety
        continue
    Zs = np.exp(-(Xg - subset['Amount'].mean()) ** 2 / (2 * subset['Amount'].std() ** 2))
    Zs *= np.exp(-(Yg - subset['Age'].mean()) ** 2 / (2 * subset['Age'].std() ** 2))
    ax.contour3D(Xg, Yg, Zs, 50, cmap='viridis', alpha=0.5)
ax.set_title('3D Contour — Amount × Age × Severity')
save_fig(fig, "contour3d_amount_age_severity.png")

# --------------------------
# Clustermap of correlations
# --------------------------
cm = sns.clustermap(df_numeric.corr().fillna(0), cmap='coolwarm', annot=True, linewidths=.5, figsize=(14, 12))
cm.fig.suptitle('Clustermap of Selected Features', y=1.02)
save_grid(cm, "clustermap_corr.png")

# --------------------------
# Hexbin (jointplot)
# --------------------------
hexg = sns.jointplot(data=df, x='Amount', y='Age', kind='hex', gridsize=20, marginal_kws=dict(bins=20))
hexg.fig.suptitle('Hexbin — Amount vs Age', y=1.02)
save_grid(hexg, "hexbin_amount_age.png")

# --------------------------
# Swarm plot
# --------------------------
plt.figure(figsize=(10, 8))
sw = sns.swarmplot(x='Insurance', y='Amount', data=df_subset_1000, size=3, palette='viridis')
sw.set_title("Swarm — Insurance vs Amount (sample n=1000)")
plt.tight_layout()
save_current("swarm_insurance_amount.png")

print("\n[done] All figures saved under:", ASSETS_DIR)
