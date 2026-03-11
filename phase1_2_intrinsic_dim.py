"""
Intrinsic Dimensionality of the AlphaEarth Embedding Manifold

Levina & Bickel (2004) MLE intrinsic dimensionality estimation.
200K combined (balanced across 2017-2023) + 100K per year for stability.

Input:  Yearly parquet files from data/unified_conus/
Output: manifold_results/ (intrinsic dimensionality estimates, figures)

Author: Mashrekur Rahman | 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import pyarrow.parquet as pq
import json, os, gc, time, warnings
warnings.filterwarnings('ignore')

# ── Config ─────────────────────────────────────────────────────────────────
DATA_DIR = '../../data/unified_conus'
RESULTS_DIR = 'manifold_results'
OUTPUT_DIR = 'manifold_results'
FIG_DIR = f'{OUTPUT_DIR}/figures'

YEARS = list(range(2017, 2024))
SUBSAMPLE_TOTAL = 200_000
SUBSAMPLE_PER_YEAR = 100_000
SEED = 42

K_VALUES = [5, 10, 20, 30, 50, 75, 100]
K_LOCAL = 20
K_MULTISCALE = [10, 20, 50, 100]

AE_COLS = [f'A{i:02d}' for i in range(64)]
N_DIMS = 64
CONUS_EXTENT = [-125.0, -66.5, 24.5, 49.5]

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 12,
    'axes.linewidth': 0.8, 'axes.labelsize': 14,
    'axes.titlesize': 16, 'axes.titleweight': 'bold',
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print("-" * 60)
print(" INTRINSIC DIMENSIONALITY")
print("  All 7 years, 200K combined + 100K per year")
print("-" * 60)


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════

def load_data_all_years():
    """Balanced multi-year loading + per-year arrays for stability."""
    per_year = SUBSAMPLE_TOTAL // len(YEARS)
    rng = np.random.default_rng(SEED)

    fp0 = f'{DATA_DIR}/conus_{YEARS[0]}_unified.parquet'
    all_cols = pq.read_schema(fp0).names

    env_wanted = ['elevation', 'temp_mean_c', 'precip_annual_mm', 'ndvi_mean',
                  'evi_mean', 'impervious_pct', 'tree_cover_2000', 'soil_ph',
                  'lst_day_c', 'et_annual_mm']
    env_found = [c for c in env_wanted if c in all_cols]

    lc_col = None
    for cand in ['nlcd_class', 'land_cover', 'nlcd', 'landcover']:
        if cand in all_cols:
            lc_col = cand
            break

    load_cols = list(set(['longitude', 'latitude'] + AE_COLS + env_found +
                         ([lc_col] if lc_col else [])))

    print(f"\nLoading: {per_year:,}/yr × {len(YEARS)} yrs = {per_year*len(YEARS):,} combined")
    print(f"  + {SUBSAMPLE_PER_YEAR:,}/yr for stability")

    frames = []
    year_embeddings = {}

    for year in YEARS:
        fp = f'{DATA_DIR}/conus_{year}_unified.parquet'
        if not os.path.exists(fp):
            print(f"  ⚠ Missing: {fp}"); continue

        year_cols = pq.read_schema(fp).names
        use_cols = [c for c in load_cols if c in year_cols]

        df_y = pd.read_parquet(fp, columns=use_cols)
        n_total = len(df_y)

        # Combined sample
        idx_c = rng.choice(n_total, size=min(per_year, n_total), replace=False)
        frames.append(df_y.iloc[idx_c].reset_index(drop=True))

        # Per-year sample (larger)
        n_py = min(SUBSAMPLE_PER_YEAR, n_total)
        idx_py = rng.choice(n_total, size=n_py, replace=False)
        year_embeddings[year] = df_y[AE_COLS].values[idx_py].astype(np.float64)

        print(f"  {year}: {n_total:,} → {len(idx_c):,} combined + {n_py:,} per-year")
        del df_y; gc.collect()

    df = pd.concat(frames, ignore_index=True)
    E = df[AE_COLS].values.astype(np.float64)
    coords = df[['longitude', 'latitude']].values

    print(f"\n  Combined: {E.shape[0]:,} × {E.shape[1]}")
    print(f"  Env: {env_found}")
    print(f"  LC: {lc_col}")

    return E, coords, df, lc_col, env_found, year_embeddings

E, coords, df_full, lc_col, env_cols, year_embeddings = load_data_all_years()


# ═══════════════════════════════════════════════════════════════════════════
# 2. PCA PROJECTION
# ═══════════════════════════════════════════════════════════════════════════

print("\nComputing 3-D PCA projection...")
pca_model = PCA(n_components=3)
E_pca = pca_model.fit_transform(E)
var_pct = pca_model.explained_variance_ratio_[:3] * 100
print(f"  Variance: {var_pct}")
print(f"  Total (3 PCs): {var_pct.sum():.1f}%")


# ═══════════════════════════════════════════════════════════════════════════
# 3. K-NN DISTANCES
# ═══════════════════════════════════════════════════════════════════════════

def compute_knn(E, max_k=None):
    if max_k is None:
        max_k = max(K_VALUES) + 1
    print(f"\nComputing {max_k}-NN for {E.shape[0]:,} points...")
    t0 = time.time()
    nn = NearestNeighbors(n_neighbors=max_k + 1, algorithm='auto', metric='euclidean')
    nn.fit(E)
    d, idx = nn.kneighbors(E)
    print(f"  Done in {time.time()-t0:.1f}s, mean NN dist = {d[:,1].mean():.6f}")
    return d[:, 1:], idx[:, 1:]

distances, nn_indices = compute_knn(E)


# ═══════════════════════════════════════════════════════════════════════════
# 4. MLE INTRINSIC DIMENSIONALITY
# ═══════════════════════════════════════════════════════════════════════════

def mle_id(distances, k):
    """Levina & Bickel (2004): d̂(x,k) = [1/(k-1) Σ log(rₖ/rⱼ)]⁻¹"""
    eps = 1e-10
    r_k = np.maximum(distances[:, k - 1], eps)
    r_j = np.maximum(distances[:, :k - 1], eps)
    log_ratios = np.log(r_k[:, np.newaxis] / r_j)
    return np.clip(1.0 / np.maximum(log_ratios.mean(axis=1), eps), 0.5, 64.0)


print("\n" + "=" * 70)
print("INTRINSIC DIMENSIONALITY ESTIMATION (combined)")
print("-" * 60)

global_id = {}
local_id_arrays = {}

for k in K_VALUES:
    if k > distances.shape[1]:
        continue
    lid = mle_id(distances, k)
    global_id[k] = {
        'k': k, 'mean_id': float(lid.mean()), 'median_id': float(np.median(lid)),
        'std_id': float(lid.std()),
        'q25': float(np.percentile(lid, 25)), 'q75': float(np.percentile(lid, 75)),
    }
    local_id_arrays[k] = lid
    print(f"  k={k:3d}: ID = {lid.mean():.2f} ± {lid.std():.2f} (median={np.median(lid):.2f})")

local_id = local_id_arrays[K_LOCAL]

# Save
with open(f'{OUTPUT_DIR}/intrinsic_dimensionality_global.json', 'w') as f:
    json.dump(global_id, f, indent=2)

id_df = pd.DataFrame({'longitude': coords[:, 0], 'latitude': coords[:, 1], 'local_id': local_id})
for col in df_full.columns:
    if col not in AE_COLS + ['longitude', 'latitude']:
        id_df[col] = df_full[col].values
id_df.to_csv(f'{OUTPUT_DIR}/intrinsic_dimensionality_local.csv', index=False)
print("  ✓ Saved combined ID results")


# ═══════════════════════════════════════════════════════════════════════════
# 5. COMPARISON WITH PCA EFFECTIVE DIMENSIONALITY
# ═══════════════════════════════════════════════════════════════════════════

pca_ref = {}
eig_path = f'{RESULTS_DIR}/eigenvalues.csv'
if os.path.exists(eig_path):
    eig = pd.read_csv(eig_path)
    eigenvalues = eig['eigenvalue'].values
    cumvar = eig['cumulative_variance'].values
    pr = (eigenvalues.sum())**2 / (eigenvalues**2).sum()
    n90 = int(np.searchsorted(cumvar, 0.90) + 1)
    pca_ref = {'pr': pr, 'n90': n90}
    print(f"\n  Covariance analysis: PR={pr:.1f}, 90%→{n90} PCs")


# ═══════════════════════════════════════════════════════════════════════════
# 6. ID BY GROUP
# ═══════════════════════════════════════════════════════════════════════════

NLCD_LABELS = {
    11: 'Open Water', 12: 'Ice/Snow',
    21: 'Dev. Open', 22: 'Dev. Low', 23: 'Dev. Med', 24: 'Dev. High',
    31: 'Barren', 41: 'Deciduous Forest', 42: 'Evergreen Forest', 43: 'Mixed Forest',
    51: 'Dwarf Scrub', 52: 'Shrub/Scrub', 71: 'Grassland', 72: 'Sedge',
    81: 'Pasture/Hay', 82: 'Cultivated Crops', 90: 'Woody Wetlands', 95: 'Emerg. Wetlands',
}

group_col = None
print("\nID by group:")

if lc_col and lc_col in id_df.columns:
    id_df['lc_label'] = id_df[lc_col].map(NLCD_LABELS).fillna('Other')
    group_col = 'lc_label'
elif 'elevation' in id_df.columns:
    print("  Using elevation grouping")
    id_df['elev_group'] = pd.cut(id_df['elevation'],
        bins=[-100, 100, 500, 1000, 2000, 5000],
        labels=['<100m', '100-500m', '500-1000m', '1000-2000m', '>2000m'])
    group_col = 'elev_group'

if group_col:
    grp = id_df.groupby(group_col)['local_id'].agg(['mean', 'std', 'count'])
    grp = grp[grp['count'] >= 100].sort_values('mean', ascending=False)
    print(grp.to_string())


# ═══════════════════════════════════════════════════════════════════════════
# 7. PER-YEAR ID STABILITY
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PER-YEAR INTRINSIC DIMENSIONALITY")
print("-" * 60)

year_id_results = {}

for year in sorted(year_embeddings.keys()):
    E_y = year_embeddings[year]
    t0 = time.time()

    nn = NearestNeighbors(n_neighbors=max(K_VALUES) + 1, algorithm='auto', metric='euclidean')
    nn.fit(E_y)
    d_y, _ = nn.kneighbors(E_y)
    d_y = d_y[:, 1:]
    elapsed = time.time() - t0

    year_id_results[year] = {}
    for k in K_VALUES:
        if k > d_y.shape[1]:
            continue
        lid = mle_id(d_y, k)
        year_id_results[year][k] = {
            'mean_id': float(lid.mean()),
            'median_id': float(np.median(lid)),
            'std_id': float(lid.std()),
        }

    ref = year_id_results[year].get(K_LOCAL, {})
    print(f"  {year}: ID(k={K_LOCAL}) = {ref.get('mean_id',0):.2f} ± {ref.get('std_id',0):.2f}  ({elapsed:.1f}s)")
    del d_y; gc.collect()

# Summary
ids_py = [year_id_results[y][K_LOCAL]['mean_id'] for y in sorted(year_id_results.keys())]
print(f"\n  Cross-year: {np.mean(ids_py):.2f} ± {np.std(ids_py):.2f}")

rows = []
for y in sorted(year_id_results.keys()):
    for k, r in year_id_results[y].items():
        rows.append({'year': y, 'k': k, **r})
pd.DataFrame(rows).to_csv(f'{OUTPUT_DIR}/per_year_intrinsic_dimensionality.csv', index=False)
print("  ✓ Saved per-year ID")


# ═══════════════════════════════════════════════════════════════════════════
# 8. DIAGNOSTIC FIGURES
# ═══════════════════════════════════════════════════════════════════════════

# --- 8a: ID vs k ---
print("\nGenerating diagnostic figures...")

fig, ax = plt.subplots(figsize=(8, 5))
ks = sorted(global_id.keys())
means = [global_id[k]['mean_id'] for k in ks]
medians = [global_id[k]['median_id'] for k in ks]
q25 = [global_id[k]['q25'] for k in ks]
q75 = [global_id[k]['q75'] for k in ks]
ax.plot(ks, means, 'o-', color='steelblue', lw=2, ms=7, label='Mean ID', zorder=3)
ax.plot(ks, medians, 's--', color='coral', lw=1.5, ms=6, label='Median ID', zorder=3)
ax.fill_between(ks, q25, q75, alpha=0.15, color='steelblue', label='IQR')
if pca_ref:
    ax.axhline(y=pca_ref['pr'], color='red', ls=':', lw=1.5, label=f'PCA PR = {phase1_1["pr"]:.1f}')
ax.set_xlabel('Number of Neighbors (k)')
ax.set_ylabel('Intrinsic Dimensionality')
ax.set_title('MLE Intrinsic Dimensionality vs. Neighborhood Size')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_id_vs_k.png', dpi=300, facecolor='white')
fig.savefig(f'{FIG_DIR}/fig_id_vs_k.pdf', dpi=300, facecolor='white')
plt.show(); print("  ✓ ID vs k")

# --- 8b: Histogram ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(local_id, bins=80, color='steelblue', alpha=0.7, density=True, edgecolor='none')
ax.axvline(x=local_id.mean(), color='red', lw=2, label=f'Mean = {local_id.mean():.1f}')
ax.axvline(x=np.median(local_id), color='orange', lw=2, ls='--', label=f'Median = {np.median(local_id):.1f}')
if pca_ref:
    ax.axvline(x=pca_ref['pr'], color='green', lw=1.5, ls=':', label=f'PCA PR = {phase1_1["pr"]:.1f}')
ax.set_xlabel('Local Intrinsic Dimensionality'); ax.set_ylabel('Density')
ax.set_title(f'Distribution of Local ID (k={K_LOCAL})')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_id_histogram.png', dpi=300, facecolor='white')
plt.show(); print("  ✓ Histogram")

# --- 8c: CONUS map ---
fig, ax = plt.subplots(figsize=(14, 8))
vmin, vmax = np.percentile(local_id, [2, 98])
sc = ax.scatter(coords[:, 0], coords[:, 1], c=local_id, s=0.3, cmap='magma_r',
                alpha=0.7, vmin=vmin, vmax=vmax, rasterized=True)
plt.colorbar(sc, ax=ax, shrink=0.7, label='Local Intrinsic Dimensionality')
ax.set_xlim(CONUS_EXTENT[0], CONUS_EXTENT[1]); ax.set_ylim(CONUS_EXTENT[2], CONUS_EXTENT[3])
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude'); ax.set_aspect('equal')
ax.set_title(f'Local Intrinsic Dimensionality Across CONUS (k={K_LOCAL})')
ax.annotate(f'Mean = {local_id.mean():.1f} ± {local_id.std():.1f}', xy=(0.02, 0.05),
            xycoords='axes fraction', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_id_spatial_map.png', dpi=300, facecolor='white')
plt.show(); print("  ✓ CONUS map")

# --- 8d: By group ---
if group_col and group_col in id_df.columns:
    counts = id_df[group_col].value_counts()
    valid_g = counts[counts >= 100].index.tolist()
    df_plot = id_df[id_df[group_col].isin(valid_g)]
    order = df_plot.groupby(group_col)['local_id'].mean().sort_values(ascending=False).index.tolist()
    if len(order) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        bp_data = [df_plot[df_plot[group_col] == g]['local_id'].values for g in order]
        bp = ax.boxplot(bp_data, labels=order, patch_artist=True, widths=0.6,
                        showfliers=False, medianprops=dict(color='red', lw=1.5))
        for patch in bp['boxes']:
            patch.set_facecolor('steelblue'); patch.set_alpha(0.6)
        ax.set_ylabel('Local ID'); ax.set_title(f'ID by {group_col.replace("_"," ").title()} (k={K_LOCAL})')
        ax.tick_params(axis='x', rotation=45); ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        fig.savefig(f'{FIG_DIR}/fig_id_by_landcover.png', dpi=300, facecolor='white')
        plt.show(); print("  ✓ By group")

# --- 8e: Per-year stability ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
cmap_yr = plt.cm.viridis(np.linspace(0.1, 0.9, len(year_id_results)))
for i, (year, kdict) in enumerate(sorted(year_id_results.items())):
    ks_y = sorted(kdict.keys())
    means_y = [kdict[k]['mean_id'] for k in ks_y]
    ax1.plot(ks_y, means_y, 'o-', color=cmap_yr[i], lw=1.5, ms=5, label=str(year))
ax1.set_xlabel('k'); ax1.set_ylabel('Mean ID')
ax1.set_title('(a) ID vs. k by Year'); ax1.legend(fontsize=7, ncol=2); ax1.grid(True, alpha=0.3)

years_s = sorted(year_id_results.keys())
ids_bar = [year_id_results[y][K_LOCAL]['mean_id'] for y in years_s]
ax2.bar([str(y) for y in years_s], ids_bar, color=cmap_yr, alpha=0.8, edgecolor='none')
ax2.axhline(y=np.mean(ids_bar), color='red', ls='--', lw=1.5,
            label=f'Mean: {np.mean(ids_bar):.2f} ± {np.std(ids_bar):.2f}')
ax2.set_xlabel('Year'); ax2.set_ylabel(f'Mean ID (k={K_LOCAL})')
ax2.set_title(f'(b) Temporal Stability (k={K_LOCAL})'); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_id_per_year.png', dpi=300, facecolor='white')
plt.show(); print("  ✓ Per-year stability")


# ═══════════════════════════════════════════════════════════════════════════
# 9. 3D MANIFOLD VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3D MANIFOLD VISUALIZATION")
print("-" * 60)

rng_3d = np.random.default_rng(42)
n_3d = min(30_000, len(E_pca))
idx_3d = rng_3d.choice(len(E_pca), size=n_3d, replace=False)
pca_sub = E_pca[idx_3d]
id_sub = local_id[idx_3d]

# Dual view: ID + physical variable
fig = plt.figure(figsize=(18, 8))

ax1 = fig.add_subplot(121, projection='3d')
vmin_id, vmax_id = np.percentile(id_sub, [2, 98])
sc1 = ax1.scatter(pca_sub[:, 0], pca_sub[:, 1], pca_sub[:, 2], c=id_sub,
                   cmap='magma_r', s=0.5, alpha=0.4, vmin=vmin_id, vmax=vmax_id, rasterized=True)
ax1.set_xlabel(f'PC1 ({var_pct[0]:.1f}%)', fontsize=10)
ax1.set_ylabel(f'PC2 ({var_pct[1]:.1f}%)', fontsize=10)
ax1.set_zlabel(f'PC3 ({var_pct[2]:.1f}%)', fontsize=10)
ax1.set_title('(a) Manifold: Local ID', fontweight='bold', fontsize=13)
ax1.view_init(elev=25, azim=135); ax1.tick_params(labelsize=7)
plt.colorbar(sc1, ax=ax1, shrink=0.6, pad=0.1, label='Local ID')

ax2 = fig.add_subplot(122, projection='3d')
color_var = None
cmap_d = {'elevation': 'terrain', 'temp_mean_c': 'RdYlBu_r', 'evi_mean': 'YlGn'}
lbl_d = {'elevation': 'Elevation (m)', 'temp_mean_c': 'Mean Temp (°C)', 'evi_mean': 'EVI'}

for cand in ['elevation', 'temp_mean_c', 'evi_mean']:
    if cand in df_full.columns:
        vals = df_full[cand].values[idx_3d]
        if np.isfinite(vals).sum() > len(vals) * 0.5:
            color_var = cand; color_vals = vals; break

if color_var:
    valid = np.isfinite(color_vals)
    vmin_e, vmax_e = np.nanpercentile(color_vals[valid], [2, 98])
    sc2 = ax2.scatter(pca_sub[valid, 0], pca_sub[valid, 1], pca_sub[valid, 2],
                       c=color_vals[valid], cmap=cmap_d.get(color_var, 'viridis'),
                       s=0.5, alpha=0.4, vmin=vmin_e, vmax=vmax_e, rasterized=True)
    plt.colorbar(sc2, ax=ax2, shrink=0.6, pad=0.1, label=lbl_d.get(color_var, color_var))
    ax2.set_title(f'(b) Manifold: {lbl_d.get(color_var,color_var)}', fontweight='bold', fontsize=13)

ax2.set_xlabel(f'PC1 ({var_pct[0]:.1f}%)', fontsize=10)
ax2.set_ylabel(f'PC2 ({var_pct[1]:.1f}%)', fontsize=10)
ax2.set_zlabel(f'PC3 ({var_pct[2]:.1f}%)', fontsize=10)
ax2.view_init(elev=25, azim=135); ax2.tick_params(labelsize=7)

plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_3d_manifold_dual.png', dpi=300, facecolor='white')
fig.savefig(f'{FIG_DIR}/fig_3d_manifold_dual.pdf', dpi=300, facecolor='white')
plt.show(); print("  ✓ 3D manifold dual")

# Extra env var views
extra_vars = [c for c in env_cols if c != color_var and c in df_full.columns][:3]
if extra_vars:
    fig2, axes2 = plt.subplots(1, len(extra_vars), figsize=(7*len(extra_vars), 7),
                                subplot_kw={'projection': '3d'})
    if len(extra_vars) == 1:
        axes2 = [axes2]
    extra_cmaps = {'precip_annual_mm': 'Blues', 'ndvi_mean': 'YlGn', 'tree_cover_2000': 'Greens',
                   'lst_day_c': 'RdYlBu_r', 'et_annual_mm': 'PuBu', 'soil_ph': 'YlOrBr',
                   'impervious_pct': 'Greys', 'temp_mean_c': 'RdYlBu_r', 'evi_mean': 'YlGn',
                   'elevation': 'terrain'}
    for i, var in enumerate(extra_vars):
        vals = df_full[var].values[idx_3d]
        valid = np.isfinite(vals)
        if valid.sum() < 100: continue
        vv = np.nanpercentile(vals[valid], [2, 98])
        sc = axes2[i].scatter(pca_sub[valid, 0], pca_sub[valid, 1], pca_sub[valid, 2],
                               c=vals[valid], cmap=extra_cmaps.get(var, 'viridis'),
                               s=0.5, alpha=0.4, vmin=vv[0], vmax=vv[1], rasterized=True)
        axes2[i].set_title(var, fontweight='bold')
        axes2[i].view_init(elev=25, azim=135); axes2[i].tick_params(labelsize=6)
        plt.colorbar(sc, ax=axes2[i], shrink=0.5, pad=0.1)
    plt.tight_layout()
    fig2.savefig(f'{FIG_DIR}/fig_3d_manifold_envvars.png', dpi=300, facecolor='white')
    plt.show(); print("  ✓ 3D manifold env vars")


# ═══════════════════════════════════════════════════════════════════════════
# 10. MULTI-SCALE CONUS MAPS
# ═══════════════════════════════════════════════════════════════════════════

print("\nMulti-scale CONUS maps...")
k_show = [k for k in K_MULTISCALE if k in local_id_arrays]
if k_show:
    fig, axes = plt.subplots(1, len(k_show), figsize=(6*len(k_show), 5))
    if len(k_show) == 1: axes = [axes]
    all_ids = np.concatenate([local_id_arrays[k] for k in k_show])
    vmin_ms, vmax_ms = np.percentile(all_ids, [2, 98])
    for i, k in enumerate(k_show):
        lid = local_id_arrays[k]
        sc = axes[i].scatter(coords[:, 0], coords[:, 1], c=lid, s=0.2, cmap='magma_r',
                              alpha=0.7, vmin=vmin_ms, vmax=vmax_ms, rasterized=True)
        axes[i].set_xlim(CONUS_EXTENT[0], CONUS_EXTENT[1])
        axes[i].set_ylim(CONUS_EXTENT[2], CONUS_EXTENT[3])
        axes[i].set_aspect('equal')
        axes[i].set_title(f'k={k} (mean={lid.mean():.1f})', fontweight='bold')
        axes[i].tick_params(labelsize=7)
        if i == 0: axes[i].set_ylabel('Latitude')
    plt.colorbar(sc, ax=axes, shrink=0.8, label='Local ID', pad=0.02)
    plt.suptitle('Multi-Scale Intrinsic Dimensionality', fontweight='bold', fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig(f'{FIG_DIR}/fig_id_multiscale_conus.png', dpi=300, facecolor='white', bbox_inches='tight')
    plt.show(); print("  ✓ Multi-scale")


# ═══════════════════════════════════════════════════════════════════════════
# 11. PUBLICATION FIGURE — Publication figure 3
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PUBLICATION FIGURE (Publication figure 3)")
print("-" * 60)

fig = plt.figure(figsize=(18, 20))
gs = gridspec.GridSpec(3, 2, hspace=0.30, wspace=0.25, height_ratios=[1, 0.9, 0.8])

# (a) 3D — local ID
ax_a = fig.add_subplot(gs[0, 0], projection='3d')
sc_a = ax_a.scatter(pca_sub[:, 0], pca_sub[:, 1], pca_sub[:, 2], c=id_sub,
                     cmap='magma_r', s=0.4, alpha=0.35, vmin=vmin_id, vmax=vmax_id, rasterized=True)
ax_a.set_xlabel(f'PC1 ({var_pct[0]:.1f}%)', fontsize=9)
ax_a.set_ylabel(f'PC2 ({var_pct[1]:.1f}%)', fontsize=9)
ax_a.set_zlabel(f'PC3 ({var_pct[2]:.1f}%)', fontsize=9)
ax_a.set_title('(a) Manifold: Local Intrinsic Dimensionality', fontweight='bold', fontsize=12)
ax_a.view_init(elev=20, azim=140); ax_a.tick_params(labelsize=6)
plt.colorbar(sc_a, ax=ax_a, shrink=0.5, pad=0.12, label='Local ID')

# (b) 3D — physical variable
ax_b = fig.add_subplot(gs[0, 1], projection='3d')
if color_var:
    valid = np.isfinite(color_vals)
    sc_b = ax_b.scatter(pca_sub[valid, 0], pca_sub[valid, 1], pca_sub[valid, 2],
                         c=color_vals[valid], cmap=cmap_d.get(color_var, 'viridis'),
                         s=0.4, alpha=0.35, vmin=vmin_e, vmax=vmax_e, rasterized=True)
    plt.colorbar(sc_b, ax=ax_b, shrink=0.5, pad=0.12, label=lbl_d.get(color_var, color_var))
    ax_b.set_title(f'(b) Manifold: {lbl_d.get(color_var,color_var)}', fontweight='bold', fontsize=12)
ax_b.set_xlabel(f'PC1 ({var_pct[0]:.1f}%)', fontsize=9)
ax_b.set_ylabel(f'PC2 ({var_pct[1]:.1f}%)', fontsize=9)
ax_b.set_zlabel(f'PC3 ({var_pct[2]:.1f}%)', fontsize=9)
ax_b.view_init(elev=20, azim=140); ax_b.tick_params(labelsize=6)

# (c) CONUS map
ax_c = fig.add_subplot(gs[1, 0])
sc_c = ax_c.scatter(coords[:, 0], coords[:, 1], c=local_id, s=0.3, cmap='magma_r',
                     alpha=0.7, vmin=vmin, vmax=vmax, rasterized=True)
plt.colorbar(sc_c, ax=ax_c, shrink=0.8, label='Local ID')
ax_c.set_xlim(CONUS_EXTENT[0], CONUS_EXTENT[1]); ax_c.set_ylim(CONUS_EXTENT[2], CONUS_EXTENT[3])
ax_c.set_aspect('equal'); ax_c.set_xlabel('Longitude'); ax_c.set_ylabel('Latitude')
ax_c.set_title(f'(c) Local ID Across CONUS (k={K_LOCAL})', fontweight='bold', fontsize=12)
ax_c.annotate(f'Mean = {local_id.mean():.1f} ± {local_id.std():.1f}', xy=(0.02, 0.05),
              xycoords='axes fraction', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# (d) ID vs k
ax_d = fig.add_subplot(gs[1, 1])
ax_d.plot(ks, means, 'o-', color='steelblue', lw=2, ms=7, label='Mean ID', zorder=3)
ax_d.plot(ks, medians, 's--', color='coral', lw=1.5, ms=6, label='Median ID', zorder=3)
ax_d.fill_between(ks, q25, q75, alpha=0.15, color='steelblue', label='IQR')
if pca_ref:
    ax_d.axhline(y=pca_ref['pr'], color='red', ls=':', lw=1.5, label=f'PCA PR = {phase1_1["pr"]:.1f}')
ax_d.set_xlabel('Number of Neighbors (k)'); ax_d.set_ylabel('Intrinsic Dimensionality')
ax_d.set_title('(d) Global ID vs. Neighborhood Size', fontweight='bold', fontsize=12)
ax_d.legend(fontsize=8); ax_d.grid(True, alpha=0.3)

# (e) Multi-scale k=10 vs k=100
k_small, k_large = min(K_MULTISCALE), max(K_MULTISCALE)
if k_small in local_id_arrays and k_large in local_id_arrays:
    gs_e = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, 0], wspace=0.15)
    all_ms = np.concatenate([local_id_arrays[k_small], local_id_arrays[k_large]])
    vmin_ms, vmax_ms = np.percentile(all_ms, [2, 98])
    for j, (kv, sub) in enumerate([(k_small, gs_e[0]), (k_large, gs_e[1])]):
        ax_e = fig.add_subplot(sub)
        lid_k = local_id_arrays[kv]
        ax_e.scatter(coords[:, 0], coords[:, 1], c=lid_k, s=0.2, cmap='magma_r',
                     alpha=0.7, vmin=vmin_ms, vmax=vmax_ms, rasterized=True)
        ax_e.set_xlim(CONUS_EXTENT[0], CONUS_EXTENT[1])
        ax_e.set_ylim(CONUS_EXTENT[2], CONUS_EXTENT[3])
        ax_e.set_aspect('equal')
        ax_e.set_title(f'k={kv} (mean={lid_k.mean():.1f})', fontsize=10, fontweight='bold')
        ax_e.tick_params(labelsize=6)
        if j == 0: ax_e.set_ylabel('Lat', fontsize=9)
    fig.text(0.25, 0.27, '(e) Multi-Scale Comparison', ha='center', fontsize=12, fontweight='bold')

# (f) By group
ax_f = fig.add_subplot(gs[2, 1])
if group_col and group_col in id_df.columns:
    counts = id_df[group_col].value_counts()
    valid_g = counts[counts >= 100].index.tolist()
    df_p = id_df[id_df[group_col].isin(valid_g)]
    order = df_p.groupby(group_col)['local_id'].mean().sort_values(ascending=False).index.tolist()
    if order:
        bp_data = [df_p[df_p[group_col] == g]['local_id'].values for g in order]
        bp = ax_f.boxplot(bp_data, labels=order, patch_artist=True, widths=0.6,
                          showfliers=False, medianprops=dict(color='red', lw=1.5))
        for p in bp['boxes']: p.set_facecolor('steelblue'); p.set_alpha(0.6)
        ax_f.tick_params(axis='x', rotation=55, labelsize=7)
ax_f.set_ylabel('Local ID')
ax_f.set_title(f'(f) ID by {(group_col or "Group").replace("_"," ").title()}', fontweight='bold', fontsize=12)
ax_f.grid(True, alpha=0.3, axis='y')

fig.savefig(f'{FIG_DIR}/fig3_intrinsic_dimensionality.png', dpi=300, facecolor='white', bbox_inches='tight')
fig.savefig(f'{FIG_DIR}/fig3_intrinsic_dimensionality.pdf', dpi=300, facecolor='white', bbox_inches='tight')
plt.show(); print("  ✓ Figure 3 saved")


# ═══════════════════════════════════════════════════════════════════════════
# 12. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SUMMARY: INTRINSIC DIMENSIONALITY")
print("-" * 60)

print(f"\n  DATA: {len(local_id):,} combined across {len(YEARS)} years")
print(f"        {SUBSAMPLE_PER_YEAR:,} per year for stability")

print(f"\n  GLOBAL ID (combined):")
for k, r in sorted(global_id.items()):
    m = " ← ref" if k == K_LOCAL else ""
    print(f"    k={k:3d}: {r['mean_id']:.2f} ± {r['std_id']:.2f}{m}")

print(f"\n  LOCAL ID (k={K_LOCAL}): mean={local_id.mean():.2f}, median={np.median(local_id):.2f}")

if pca_ref:
    mle = global_id.get(20, {}).get('mean_id', 0)
    pr = pca_ref['pr']
    ratio = mle / pr if pr > 0 else 0
    print(f"\n  PCA COMPARISON: MLE={mle:.1f}, PR={pr:.1f}, ratio={ratio:.2f}")
    if ratio < 0.8:
        print(f"    → Manifold is CURVED")
    elif ratio > 1.2:
        print(f"    → Local complexity exceeds linear")
    else:
        print(f"    → Approximately FLAT")

print(f"\n  TEMPORAL STABILITY (k={K_LOCAL}):")
for y in sorted(year_id_results.keys()):
    r = year_id_results[y][K_LOCAL]
    print(f"    {y}: {r['mean_id']:.2f} ± {r['std_id']:.2f}")
print(f"    Cross-year: {np.mean(ids_py):.2f} ± {np.std(ids_py):.2f}")

print("\n" + "=" * 70)

del E, distances, nn_indices, year_embeddings
gc.collect()
print("Done.")
