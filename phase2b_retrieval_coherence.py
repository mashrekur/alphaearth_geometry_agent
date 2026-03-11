"""
Geometric Retrieval Coherence and Enhanced Dictionary

Evaluates whether manifold geometry predicts FAISS retrieval coherence:
  1. Environmental property variance within retrieved neighborhoods
  2. Correlation between intrinsic dimensionality and retrieval coherence
  3. Regional dimension importance profiles across CONUS subregions
  4. Confidence model from geometric features

Input:  Yearly parquet files + manifold_results/ from characterization steps
Output: manifold_results/ (coherence metrics, enhanced dictionary, figures)

Author: Mashrekur Rahman | 2026
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr, pearsonr
import pyarrow.parquet as pq
import json, os, gc, time, warnings
warnings.filterwarnings('ignore')

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

# ── Config ─────────────────────────────────────────────────────────────────
DATA_DIR = '../../data/unified_conus'
RESULTS_DIR = 'manifold_results'
OUTPUT_DIR = 'manifold_results'
FIG_DIR = f'{OUTPUT_DIR}/figures'

YEARS = list(range(2017, 2024))
SUBSAMPLE_TOTAL = 300_000
N_PROBES = 10_000
K_RETRIEVAL = 10
K_LOCAL_PCA = 100
SEED = 42

AE_COLS = [f'A{i:02d}' for i in range(64)]
N_DIMS = 64
CONUS_EXTENT = [-125.0, -66.5, 24.5, 49.5]

# Environmental variables for coherence measurement
ENV_VARS = ['elevation', 'temp_mean_c', 'precip_annual_mm', 'evi_mean',
            'ndvi_mean', 'lst_day_c', 'tree_cover_2000', 'soil_moisture',
            'et_annual_mm', 'soil_ph']

# Geographic regions for dimension importance analysis
REGIONS = {
    'Pacific NW':    {'lon': (-125, -116), 'lat': (42, 49)},
    'Great Plains':  {'lon': (-104, -95),  'lat': (35, 48)},
    'Southeast':     {'lon': (-90, -75),   'lat': (25, 36)},
    'Mountain West': {'lon': (-115, -104), 'lat': (35, 45)},
    'Northeast':     {'lon': (-80, -67),   'lat': (39, 47)},
    'Southwest':     {'lon': (-115, -104), 'lat': (31, 37)},
}

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
print("GEOMETRIC RETRIEVAL COHERENCE")
print("-" * 60)


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════

def load_data():
    per_year = SUBSAMPLE_TOTAL // len(YEARS)
    rng = np.random.default_rng(SEED)

    fp0 = f'{DATA_DIR}/conus_{YEARS[0]}_unified.parquet'
    all_cols = pq.read_schema(fp0).names
    env_found = [c for c in ENV_VARS if c in all_cols]
    load_cols = list(set(['longitude', 'latitude'] + AE_COLS + env_found))

    print(f"\nLoading: {per_year:,}/yr × {len(YEARS)} yrs")
    frames = []
    for year in YEARS:
        fp = f'{DATA_DIR}/conus_{year}_unified.parquet'
        if not os.path.exists(fp): continue
        year_cols = pq.read_schema(fp).names
        use_cols = [c for c in load_cols if c in year_cols]
        df_y = pd.read_parquet(fp, columns=use_cols)
        idx = rng.choice(len(df_y), size=min(per_year, len(df_y)), replace=False)
        frames.append(df_y.iloc[idx].reset_index(drop=True))
        print(f"  {year}: {len(df_y):,} → {len(idx):,}")
        del df_y; gc.collect()

    df = pd.concat(frames, ignore_index=True)
    E = df[AE_COLS].values.astype(np.float64)
    coords = df[['longitude', 'latitude']].values

    # Per-variable σ for normalization
    env_sigma = {}
    for c in env_found:
        s = df[c].std()
        env_sigma[c] = s if s > 0 else 1.0

    print(f"  Combined: {len(df):,}, env: {env_found}")
    return E, coords, df, env_found, env_sigma

E, coords, df_full, env_found, env_sigma = load_data()


# ═══════════════════════════════════════════════════════════════════════════
# 2. BUILD INDEX + SELECT PROBES
# ═══════════════════════════════════════════════════════════════════════════

print(f"\nBuilding k-NN index...")
t0 = time.time()
nn = NearestNeighbors(n_neighbors=K_LOCAL_PCA + 1, algorithm='auto', metric='euclidean')
nn.fit(E)
print(f"  Fitted in {time.time()-t0:.1f}s")

# Stratified probes
rng = np.random.default_rng(SEED)
if 'elevation' in df_full.columns:
    elev = df_full['elevation'].values
    bins = [-100, 100, 500, 1000, 2000, 5000]
    labels = ['<100m', '100-500m', '500-1000m', '1000-2000m', '>2000m']
    groups = pd.cut(elev, bins=bins, labels=labels)
    per_group = N_PROBES // len(labels)
    probe_indices = []
    for label in labels:
        gidx = np.where(groups == label)[0]
        probe_indices.extend(rng.choice(gidx, size=min(per_group, len(gidx)), replace=False))
    probe_indices = np.array(probe_indices)
else:
    probe_indices = rng.choice(len(E), size=N_PROBES, replace=False)
print(f"  Probes: {len(probe_indices):,}")

# Load dimension dictionary
dd_path = '../results/dimension_dictionary.csv'
dim_to_var, dim_to_cat = {}, {}
if os.path.exists(dd_path):
    dd = pd.read_csv(dd_path)
    for c in ['sp_primary', 'primary_variable']:
        if c in dd.columns: dim_to_var = dict(zip(dd['dimension'], dd[c])); break
    for c in ['sp_category', 'category']:
        if c in dd.columns: dim_to_cat = dict(zip(dd['dimension'], dd[c])); break
    print(f"  Dictionary: {len(dim_to_var)} dims")


# ═══════════════════════════════════════════════════════════════════════════
# 3. COMPUTE RETRIEVAL COHERENCE + LOCAL GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════

print(f"\nComputing retrieval coherence at {len(probe_indices):,} locations...")
t0 = time.time()

results = []

for i, pidx in enumerate(probe_indices):
    if i % 2000 == 0 and i > 0:
        print(f"  {i:,}/{len(probe_indices):,} ({time.time()-t0:.1f}s)")

    # Retrieve k neighbors
    point = E[pidx].reshape(1, -1)
    dists, nbr_idx = nn.kneighbors(point)
    nbr_idx_retrieval = nbr_idx[0, 1:K_RETRIEVAL + 1]
    nbr_idx_pca = nbr_idx[0, 1:]

    # ── Retrieval coherence ──
    # For each env variable, compute CV (std/mean) across retrieved neighbors
    coherence_scores = []
    for col in env_found:
        nbr_vals = df_full[col].values[nbr_idx_retrieval]
        valid = nbr_vals[np.isfinite(nbr_vals)]
        if len(valid) < 3:
            continue
        # Normalized spread: std of neighbors / global σ
        spread = valid.std() / env_sigma[col]
        coherence_scores.append(spread)

    # Overall coherence = mean normalized spread (lower = more coherent)
    mean_coherence = float(np.mean(coherence_scores)) if coherence_scores else np.nan

    # ── Local geometry ──
    E_local = E[nbr_idx_pca]
    pca_local = PCA(n_components=min(K_LOCAL_PCA - 1, N_DIMS))
    pca_local.fit(E_local)

    local_eigenvalues = pca_local.explained_variance_
    local_pr = float((local_eigenvalues.sum())**2 / (local_eigenvalues**2).sum())

    # MLE ID at this point
    k_id = min(20, len(nbr_idx_pca))
    nn_dists = dists[0, 1:k_id + 1]
    eps = 1e-10
    r_k = max(nn_dists[k_id - 1], eps)
    r_j = np.maximum(nn_dists[:k_id - 1], eps)
    log_ratios = np.log(r_k / r_j)
    local_id = float(np.clip(1.0 / max(log_ratios.mean(), eps), 0.5, 64.0))

    # Top 3 dominant dimensions in local PC1
    local_pc1 = pca_local.components_[0]
    top3_idx = np.argsort(np.abs(local_pc1))[::-1][:3]
    top3_dims = [AE_COLS[j] for j in top3_idx]
    top3_weights = [float(np.abs(local_pc1[j])) for j in top3_idx]

    # Dominant category
    dom_dim = AE_COLS[top3_idx[0]]
    dom_cat = dim_to_cat.get(dom_dim, '?')

    # Mean embedding distance to neighbors
    mean_emb_dist = float(dists[0, 1:K_RETRIEVAL + 1].mean())

    results.append({
        'probe_idx': int(pidx),
        'longitude': float(coords[pidx, 0]),
        'latitude': float(coords[pidx, 1]),
        'retrieval_coherence': mean_coherence,
        'local_id': local_id,
        'local_pr': local_pr,
        'dominant_dim': dom_dim,
        'dominant_cat': dom_cat,
        'top3_dims': ','.join(top3_dims),
        'top3_weights': ','.join([f'{w:.4f}' for w in top3_weights]),
        'mean_emb_dist': mean_emb_dist,
        'elevation': float(df_full['elevation'].iloc[pidx]) if 'elevation' in df_full.columns else np.nan,
    })

elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s")

coh_df = pd.DataFrame(results)
coh_df.to_csv(f'{OUTPUT_DIR}/retrieval_coherence.csv', index=False)
print(f"  ✓ Saved retrieval coherence ({len(coh_df):,} locations)")

print(f"\n  Coherence stats: mean={coh_df['retrieval_coherence'].mean():.3f}, "
      f"std={coh_df['retrieval_coherence'].std():.3f}")
print(f"  Local ID stats: mean={coh_df['local_id'].mean():.1f}, "
      f"std={coh_df['local_id'].std():.1f}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. REGIONAL DIMENSION IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("REGIONAL DIMENSION IMPORTANCE")
print("-" * 60)

# For each region, find probes within it, aggregate local PC1 weights
region_importance = {}

for region_name, bounds in REGIONS.items():
    lon_mask = (coh_df['longitude'] >= bounds['lon'][0]) & (coh_df['longitude'] <= bounds['lon'][1])
    lat_mask = (coh_df['latitude'] >= bounds['lat'][0]) & (coh_df['latitude'] <= bounds['lat'][1])
    region_mask = lon_mask & lat_mask
    region_probes = coh_df[region_mask]

    if len(region_probes) < 20:
        print(f"  {region_name}: {len(region_probes)} probes (too few, skipping)")
        continue

    # Aggregate: for each dimension, what fraction of probes have it in top-3?
    dim_counts = {d: 0 for d in AE_COLS}
    for _, row in region_probes.iterrows():
        for d in row['top3_dims'].split(','):
            d = d.strip()
            if d in dim_counts:
                dim_counts[d] += 1

    total = len(region_probes)
    dim_fractions = {d: c / total for d, c in dim_counts.items()}

    # Top 10 dimensions for this region
    top10 = sorted(dim_fractions.items(), key=lambda x: x[1], reverse=True)[:10]

    region_importance[region_name] = {
        'n_probes': len(region_probes),
        'top10': [{
            'dimension': d,
            'fraction': f,
            'variable': dim_to_var.get(d, '?'),
            'category': dim_to_cat.get(d, '?'),
        } for d, f in top10],
        'mean_coherence': float(region_probes['retrieval_coherence'].mean()),
        'mean_local_id': float(region_probes['local_id'].mean()),
    }

    print(f"\n  {region_name} (n={len(region_probes)}, coherence={region_probes['retrieval_coherence'].mean():.3f}):")
    for d, f in top10[:5]:
        var = dim_to_var.get(d, '?')
        print(f"    {d} ({var}): {f*100:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════
# 5. CONFIDENCE MODEL
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("CONFIDENCE MODEL: Geometry → Retrieval Quality")
print("-" * 60)

# Predict retrieval coherence from geometric features
valid_mask = coh_df['retrieval_coherence'].notna() & coh_df['local_id'].notna()
df_model = coh_df[valid_mask].copy()

features = ['local_id', 'local_pr', 'mean_emb_dist']
if 'elevation' in df_model.columns:
    df_model['abs_elevation'] = df_model['elevation'].abs()
    features.append('abs_elevation')

X = df_model[features].values
y = df_model['retrieval_coherence'].values

# Remove any remaining NaN
valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
X, y = X[valid], y[valid]

print(f"  Training on {len(X):,} samples, {len(features)} features")

# Spearman correlations
print(f"\n  Feature correlations with coherence (lower coherence = better):")
for i, feat in enumerate(features):
    rho, pval = spearmanr(X[:, i], y)
    print(f"    {feat:20s}: ρ = {rho:+.3f} (p = {pval:.2e})")

# Linear regression for calibration
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = model.score(X, y)
print(f"\n  Linear model R² = {r2:.4f}")
print(f"  Coefficients: {dict(zip(features, model.coef_))}")

# Bin predictions and compute calibration
n_bins = 10
pred_bins = pd.qcut(y_pred, n_bins, duplicates='drop')
calibration = df_model.iloc[np.where(valid)[0]].copy()
calibration['predicted'] = y_pred
calibration['pred_bin'] = pd.qcut(y_pred, n_bins, duplicates='drop')
cal_stats = calibration.groupby('pred_bin').agg(
    mean_predicted=('predicted', 'mean'),
    mean_actual=('retrieval_coherence', 'mean'),
    std_actual=('retrieval_coherence', 'std'),
    count=('predicted', 'count'),
).reset_index()

print(f"\n  Calibration ({n_bins} bins):")
for _, row in cal_stats.iterrows():
    print(f"    Predicted={row['mean_predicted']:.3f} → Actual={row['mean_actual']:.3f} "
          f"± {row['std_actual']:.3f} (n={int(row['count'])})")


# ═══════════════════════════════════════════════════════════════════════════
# 6. BUILD ENHANCED DICTIONARY
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("BUILDING ENHANCED GEO-DICTIONARY")
print("-" * 60)

enhanced_dict = {
    'description': 'Enhanced geometric dictionary for multi-agent geospatial reasoning',
    'geometry_summary': {
        'global_pr': 13.3,
        'mle_id_k20': 10.4,
        'mean_tangent_angle': 69.0,
        'mean_alignment_pc1': 0.17,
        'random_alignment_baseline': 0.125,
        'manifold_is_curved': True,
        'global_directions_reliable': False,
    },
    'retrieval_quality': {
        'mean_coherence': float(coh_df['retrieval_coherence'].mean()),
        'std_coherence': float(coh_df['retrieval_coherence'].std()),
        'coherence_by_complexity': {},
    },
    'confidence_model': {
        'features': features,
        'coefficients': model.coef_.tolist(),
        'intercept': float(model.intercept_),
        'r_squared': float(r2),
    },
    'regional_profiles': region_importance,
    'per_dimension': {},
}

# Coherence by complexity bins
id_bins = pd.cut(coh_df['local_id'], bins=[0, 6, 8, 10, 12, 20])
for bin_label, group in coh_df.groupby(id_bins):
    enhanced_dict['retrieval_quality']['coherence_by_complexity'][str(bin_label)] = {
        'mean_coherence': float(group['retrieval_coherence'].mean()),
        'count': len(group),
    }

# Per-dimension: aggregate across all probes where it's in top-3
for dim in AE_COLS:
    mask = coh_df['top3_dims'].str.contains(dim, na=False)
    if mask.sum() < 5:
        enhanced_dict['per_dimension'][dim] = {
            'locally_important_fraction': float(mask.sum() / len(coh_df)),
            'variable': dim_to_var.get(dim, '?'),
            'category': dim_to_cat.get(dim, '?'),
        }
        continue

    sub = coh_df[mask]
    enhanced_dict['per_dimension'][dim] = {
        'locally_important_fraction': float(mask.sum() / len(coh_df)),
        'mean_coherence_when_dominant': float(sub['retrieval_coherence'].mean()),
        'mean_local_id_when_dominant': float(sub['local_id'].mean()),
        'spatial_extent': {
            'lon_range': [float(sub['longitude'].min()), float(sub['longitude'].max())],
            'lat_range': [float(sub['latitude'].min()), float(sub['latitude'].max())],
        },
        'variable': dim_to_var.get(dim, '?'),
        'category': dim_to_cat.get(dim, '?'),
    }

with open(f'{OUTPUT_DIR}/enhanced_geo_dictionary.json', 'w') as f:
    json.dump(enhanced_dict, f, indent=2)
print(f"  ✓ Saved enhanced_geo_dictionary.json")


# ═══════════════════════════════════════════════════════════════════════════
# 7. PUBLICATION FIGURE — Publication figure 6 (4 panels)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PUBLICATION FIGURE (Publication figure 6)")
print("-" * 60)

fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(2, 2, hspace=0.30, wspace=0.28)

# ── (a) CONUS map of retrieval coherence ──
if HAS_CARTOPY:
    ax_a = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax_a.set_extent([CONUS_EXTENT[0], CONUS_EXTENT[1],
                     CONUS_EXTENT[2], CONUS_EXTENT[3]])
    ax_a.add_feature(cfeature.COASTLINE, linewidth=0.6, color='black')
    ax_a.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='gray')
    ax_a.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
    transform = ccrs.PlateCarree()
else:
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_xlim(CONUS_EXTENT[0], CONUS_EXTENT[1])
    ax_a.set_ylim(CONUS_EXTENT[2], CONUS_EXTENT[3])
    ax_a.set_aspect('equal')
    transform = None

vmin_c, vmax_c = coh_df['retrieval_coherence'].quantile([0.02, 0.98])
scatter_kw = dict(c=coh_df['retrieval_coherence'], s=2.5, cmap='RdYlGn_r',
                  vmin=vmin_c, vmax=vmax_c, alpha=0.8, rasterized=True)
if transform:
    scatter_kw['transform'] = transform
sc_a = ax_a.scatter(coh_df['longitude'], coh_df['latitude'], **scatter_kw)
plt.colorbar(sc_a, ax=ax_a, shrink=0.7, label='Retrieval Spread (σ-normalized)', pad=0.02)
ax_a.set_title('(a) FAISS Retrieval Coherence Across CONUS', fontweight='bold', fontsize=13)

# Draw region boxes
for rname, bounds in REGIONS.items():
    rect = plt.Rectangle((bounds['lon'][0], bounds['lat'][0]),
                          bounds['lon'][1] - bounds['lon'][0],
                          bounds['lat'][1] - bounds['lat'][0],
                          linewidth=1.5, edgecolor='black', facecolor='none',
                          linestyle='--', transform=transform if transform else ax_a.transData)
    ax_a.add_patch(rect)
    cx = (bounds['lon'][0] + bounds['lon'][1]) / 2
    cy = bounds['lat'][1] + 0.5
    if transform:
        ax_a.text(cx, cy, rname, fontsize=7, ha='center', fontweight='bold',
                  transform=transform,
                  bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.8))
    else:
        ax_a.text(cx, cy, rname, fontsize=7, ha='center', fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.8))

# ── (b) Coherence vs Local ID (binned) ──
ax_b = fig.add_subplot(gs[0, 1])

# Bin local ID and compute mean coherence per bin
valid = coh_df.dropna(subset=['local_id', 'retrieval_coherence'])
n_bins_plot = 15
valid['id_bin'] = pd.cut(valid['local_id'], bins=n_bins_plot)
binned = valid.groupby('id_bin').agg(
    mean_id=('local_id', 'mean'),
    mean_coh=('retrieval_coherence', 'mean'),
    std_coh=('retrieval_coherence', 'std'),
    count=('local_id', 'count'),
).dropna()

# Scatter (faint) + binned means
ax_b.scatter(valid['local_id'], valid['retrieval_coherence'],
             s=1, alpha=0.08, color='steelblue', rasterized=True)
ax_b.errorbar(binned['mean_id'], binned['mean_coh'], yerr=binned['std_coh'],
              fmt='o-', color='firebrick', lw=2, ms=6, capsize=3,
              label='Binned mean ± σ', zorder=5)

# Spearman correlation
rho_coh, p_coh = spearmanr(valid['local_id'], valid['retrieval_coherence'])
ax_b.annotate(f'Spearman ρ = {rho_coh:.3f}\n(p < {max(p_coh, 1e-300):.0e})',
              xy=(0.95, 0.05), xycoords='axes fraction', ha='right',
              fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax_b.set_xlabel('Local Intrinsic Dimensionality')
ax_b.set_ylabel('Retrieval Spread (σ-normalized)')
ax_b.set_title('(b) Geometric Complexity Predicts Retrieval Quality',
               fontweight='bold', fontsize=13)
ax_b.legend(fontsize=9, loc='upper left')
ax_b.grid(True, alpha=0.3)

# ── (c) Regional dimension importance heatmap ──
ax_c = fig.add_subplot(gs[1, 0])

# Build heatmap: rows = regions, columns = top dimensions (union of all top-5)
all_top_dims = set()
for rname, rdata in region_importance.items():
    for entry in rdata['top10'][:7]:
        all_top_dims.add(entry['dimension'])
all_top_dims = sorted(all_top_dims)

heat_data = np.zeros((len(region_importance), len(all_top_dims)))
region_names = []
for i, (rname, rdata) in enumerate(region_importance.items()):
    region_names.append(rname)
    dim_frac = {e['dimension']: e['fraction'] for e in rdata['top10']}
    for j, dim in enumerate(all_top_dims):
        heat_data[i, j] = dim_frac.get(dim, 0)

# Add variable names to dimension labels
dim_labels = [f"{d}\n({dim_to_var.get(d, '?')[:10]})" for d in all_top_dims]

im = ax_c.imshow(heat_data, cmap='YlOrRd', aspect='auto', vmin=0)
ax_c.set_xticks(range(len(all_top_dims)))
ax_c.set_xticklabels(dim_labels, fontsize=6, rotation=60, ha='right')
ax_c.set_yticks(range(len(region_names)))
ax_c.set_yticklabels(region_names, fontsize=10)
plt.colorbar(im, ax=ax_c, shrink=0.8, label='Fraction in local top-3')
ax_c.set_title('(c) Spatially-Varying Dimension Importance', fontweight='bold', fontsize=13)

# Annotate values
for i in range(heat_data.shape[0]):
    for j in range(heat_data.shape[1]):
        v = heat_data[i, j]
        if v > 0.05:
            ax_c.text(j, i, f'{v:.0%}', ha='center', va='center',
                     fontsize=6, color='white' if v > 0.2 else 'black')

# ── (d) Confidence calibration curve ──
ax_d = fig.add_subplot(gs[1, 1])

ax_d.errorbar(cal_stats['mean_predicted'], cal_stats['mean_actual'],
              yerr=cal_stats['std_actual'], fmt='o-', color='steelblue',
              lw=2, ms=8, capsize=4, label='Observed')

# Perfect calibration line
lims = [min(cal_stats['mean_predicted'].min(), cal_stats['mean_actual'].min()),
        max(cal_stats['mean_predicted'].max(), cal_stats['mean_actual'].max())]
ax_d.plot(lims, lims, 'k--', lw=1, alpha=0.5, label='Perfect calibration')

ax_d.annotate(f'R² = {r2:.3f}\nFeatures: {", ".join(features)}',
              xy=(0.05, 0.90), xycoords='axes fraction',
              fontsize=9, va='top',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax_d.set_xlabel('Predicted Retrieval Spread')
ax_d.set_ylabel('Observed Retrieval Spread')
ax_d.set_title('(d) Geometric Confidence Calibration', fontweight='bold', fontsize=13)
ax_d.legend(fontsize=9)
ax_d.grid(True, alpha=0.3)

# ── Save ──
fig.savefig(f'{FIG_DIR}/fig6_retrieval_coherence.png', dpi=300,
            facecolor='white', bbox_inches='tight')
fig.savefig(f'{FIG_DIR}/fig6_retrieval_coherence.pdf', dpi=300,
            facecolor='white', bbox_inches='tight')
plt.close(fig)
print("  ✓ Figure 6 saved")


# ═══════════════════════════════════════════════════════════════════════════
# 8. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SUMMARY: RETRIEVAL COHERENCE")
print("-" * 60)

print(f"\n  RETRIEVAL COHERENCE:")
print(f"    Mean spread: {coh_df['retrieval_coherence'].mean():.3f} ± {coh_df['retrieval_coherence'].std():.3f}")
print(f"    Geometry–coherence correlation: ρ = {rho_coh:.3f}")

print(f"\n  CONFIDENCE MODEL:")
print(f"    R² = {r2:.4f}")
print(f"    Features: {features}")

print(f"\n  REGIONAL PROFILES:")
for rname, rdata in region_importance.items():
    top3 = ', '.join([f"{e['dimension']}({e['variable'][:8]})" for e in rdata['top10'][:3]])
    print(f"    {rname:15s}: coherence={rdata['mean_coherence']:.3f}, "
          f"ID={rdata['mean_local_id']:.1f}, top={top3}")

print(f"\n  ENHANCED DICTIONARY:")
n_important = sum(1 for d in enhanced_dict['per_dimension'].values()
                  if d.get('locally_important_fraction', 0) > 0.05)
print(f"    {n_important}/64 dimensions are locally important at >5% of locations")

print(f"\n  OUTPUTS:")
for f_name in sorted(os.listdir(OUTPUT_DIR)):
    fp = os.path.join(OUTPUT_DIR, f_name)
    if os.path.isfile(fp) and ('retrieval' in f_name or 'enhanced' in f_name or 'fig6' in f_name or 'regional' in f_name):
        print(f"    {f_name} ({os.path.getsize(fp)/1024:.1f} KB)")

print("\n" + "=" * 70)
print("Complete.")
print("  → enhanced_geo_dictionary.json: full geometric metadata for agent")
print("  → Confidence model: predict retrieval quality from local geometry")
print("  → Regional profiles: spatially-varying dimension importance")

print("-" * 60)

del E, nn
gc.collect()
print("Done.")
