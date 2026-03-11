"""
Local PCA and Tangent Space Mapping

Computes local PCA at probe locations to characterize tangent space structure:
  1. Alignment between local and global principal components
  2. Tangent space angles between neighboring locations
  3. Location-dependent dimension importance
  4. Geometric dictionary mapping locations to local geometry summaries

Input:  Yearly parquet files + global eigendecomposition from covariance step
Output: manifold_results/ (local PCA results, geometric dictionary, figures)

Author: Mashrekur Rahman | 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import pyarrow.parquet as pq
import json, os, gc, time, warnings
warnings.filterwarnings('ignore')

# Use non-interactive backend — no plt.show() blocking
import matplotlib
matplotlib.use('Agg')

# ── Config ─────────────────────────────────────────────────────────────────
DATA_DIR = '../../data/unified_conus'
RESULTS_DIR = 'manifold_results'
OUTPUT_DIR = 'manifold_results'
FIG_DIR = f'{OUTPUT_DIR}/figures'

YEARS = list(range(2017, 2024))
SUBSAMPLE_TOTAL = 200_000        # balanced across years
N_PROBE_LOCATIONS = 10_000       # locations for local PCA
K_LOCAL_PCA = 100                # neighbors per probe location
K_TANGENT_PAIRS = 500            # pairs for tangent angle computation
SEED = 42

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
print(" LOCAL PCA AND TANGENT SPACE MAPPING")
print("-" * 60)


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA + GLOBAL REFERENCES
# ═══════════════════════════════════════════════════════════════════════════

def load_data():
    """Balanced multi-year sample + load global eigenvectors."""
    per_year = SUBSAMPLE_TOTAL // len(YEARS)
    rng = np.random.default_rng(SEED)

    fp0 = f'{DATA_DIR}/conus_{YEARS[0]}_unified.parquet'
    all_cols = pq.read_schema(fp0).names

    env_wanted = ['elevation', 'temp_mean_c', 'precip_annual_mm', 'ndvi_mean',
                  'evi_mean', 'impervious_pct', 'tree_cover_2000', 'soil_ph',
                  'lst_day_c', 'et_annual_mm']
    env_found = [c for c in env_wanted if c in all_cols]
    load_cols = list(set(['longitude', 'latitude'] + AE_COLS + env_found))

    print(f"\nLoading: {per_year:,}/yr × {len(YEARS)} yrs = {per_year*len(YEARS):,}")

    frames = []
    for year in YEARS:
        fp = f'{DATA_DIR}/conus_{year}_unified.parquet'
        if not os.path.exists(fp):
            continue
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
    print(f"  Combined: {E.shape[0]:,} × {E.shape[1]}")

    return E, coords, df, env_found

E, coords, df_full, env_cols = load_data()


# ── Load global eigenvectors ──
print("\nLoading global eigenvectors...")
evec_path = f'{RESULTS_DIR}/eigenvectors.csv'
eig_path = f'{RESULTS_DIR}/eigenvalues.csv'

global_evecs = None
global_var_explained = None

if os.path.exists(evec_path) and os.path.exists(eig_path):
    evec_df = pd.read_csv(evec_path, index_col=0)
    global_evecs = evec_df.values  # shape (64, 64), columns = PC1..PC64
    eig_df = pd.read_csv(eig_path)
    global_var_explained = eig_df['variance_explained'].values
    print(f"  ✓ Global eigenvectors: {global_evecs.shape}")
    print(f"  Top 3 global PCs: {global_var_explained[:3]*100}")
else:
    print("  ⚠ Covariance results not found — will skip alignment analysis")


# ── Load dimension dictionary ──
print("\nLoading dimension dictionary...")
dd_path = '../results/dimension_dictionary.csv'
dim_to_var, dim_to_rho, dim_to_cat = {}, {}, {}

if os.path.exists(dd_path):
    dd = pd.read_csv(dd_path)
    for c in ['sp_primary', 'primary_variable']:
        if c in dd.columns:
            dim_to_var = dict(zip(dd['dimension'], dd[c])); break
    for c in ['sp_rho', 'spearman_rho']:
        if c in dd.columns:
            dim_to_rho = dict(zip(dd['dimension'], dd[c])); break
    for c in ['sp_category', 'category']:
        if c in dd.columns:
            dim_to_cat = dict(zip(dd['dimension'], dd[c])); break
    print(f"  ✓ {len(dim_to_var)} dimension interpretations loaded")


# ── Load intrinsic dimensionality estimates (if available) ──
id_path = f'{RESULTS_DIR}/intrinsic_dimensionality_local.csv'
local_id_data = None
if os.path.exists(id_path):
    local_id_data = pd.read_csv(id_path)
    print(f"  ✓ Intrinsic dimensionality data: {len(local_id_data):,} points")


# ═══════════════════════════════════════════════════════════════════════════
# 2. BUILD K-NN INDEX + SELECT PROBE LOCATIONS
# ═══════════════════════════════════════════════════════════════════════════

print(f"\nBuilding k-NN index (k={K_LOCAL_PCA})...")
t0 = time.time()
nn = NearestNeighbors(n_neighbors=K_LOCAL_PCA + 1, algorithm='auto', metric='euclidean')
nn.fit(E)
print(f"  Fitted in {time.time()-t0:.1f}s")

# Select probe locations stratified by elevation
rng = np.random.default_rng(SEED)

if 'elevation' in df_full.columns:
    elev = df_full['elevation'].values
    bins = [-100, 100, 500, 1000, 2000, 5000]
    labels = ['<100m', '100-500m', '500-1000m', '1000-2000m', '>2000m']
    groups = pd.cut(elev, bins=bins, labels=labels)
    
    per_group = N_PROBE_LOCATIONS // len(labels)
    probe_indices = []
    for label in labels:
        group_idx = np.where(groups == label)[0]
        n_sample = min(per_group, len(group_idx))
        if n_sample > 0:
            probe_indices.extend(rng.choice(group_idx, size=n_sample, replace=False))
    probe_indices = np.array(probe_indices)
    print(f"  Stratified probes: {len(probe_indices):,} across {len(labels)} elevation groups")
else:
    probe_indices = rng.choice(len(E), size=N_PROBE_LOCATIONS, replace=False)
    print(f"  Random probes: {len(probe_indices):,}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. LOCAL PCA AT EACH PROBE LOCATION
# ═══════════════════════════════════════════════════════════════════════════

print(f"\nComputing local PCA at {len(probe_indices):,} probe locations (k={K_LOCAL_PCA})...")
t0 = time.time()

results = []

for i, pidx in enumerate(probe_indices):
    if i % 2000 == 0 and i > 0:
        print(f"  {i:,}/{len(probe_indices):,} ({time.time()-t0:.1f}s)")
    
    # Get k neighbors
    point = E[pidx].reshape(1, -1)
    dists, nbr_idx = nn.kneighbors(point)
    nbr_idx = nbr_idx[0, 1:]  # exclude self
    dists = dists[0, 1:]
    
    # Local PCA on the neighborhood
    E_local = E[nbr_idx]
    pca_local = PCA(n_components=min(K_LOCAL_PCA - 1, N_DIMS))
    pca_local.fit(E_local)
    
    local_var = pca_local.explained_variance_ratio_
    local_cumvar = np.cumsum(local_var)
    local_evecs = pca_local.components_  # shape (n_components, 64)
    
    # Local effective dimensionality
    local_n80 = int(np.searchsorted(local_cumvar, 0.80) + 1)
    local_n90 = int(np.searchsorted(local_cumvar, 0.90) + 1)
    local_eigenvalues = pca_local.explained_variance_
    local_pr = (local_eigenvalues.sum())**2 / (local_eigenvalues**2).sum()
    
    # ── Alignment with global PCs ──
    align_pc1 = 0.0
    align_pc2 = 0.0
    if global_evecs is not None:
        # Cosine similarity between local PC1 and global PC1
        # Use absolute value since eigenvectors can flip sign
        local_pc1 = local_evecs[0]  # shape (64,)
        local_pc2 = local_evecs[1] if local_evecs.shape[0] > 1 else np.zeros(N_DIMS)
        global_pc1 = global_evecs[:, 0]  # shape (64,)
        global_pc2 = global_evecs[:, 1]
        
        align_pc1 = float(np.abs(np.dot(local_pc1, global_pc1)))
        align_pc2 = float(np.abs(np.dot(local_pc2, global_pc2)))
    
    # ── Dominant local dimension ──
    # Which AE dimension has the largest absolute weight in local PC1?
    dominant_dim_idx = int(np.argmax(np.abs(local_evecs[0])))
    dominant_dim = AE_COLS[dominant_dim_idx]
    dominant_weight = float(local_evecs[0, dominant_dim_idx])
    
    # Top 3 contributing dimensions to local PC1
    top3_idx = np.argsort(np.abs(local_evecs[0]))[::-1][:3]
    top3_dims = [AE_COLS[j] for j in top3_idx]
    top3_weights = [float(local_evecs[0, j]) for j in top3_idx]
    
    # ── Neighborhood spread ──
    mean_dist = float(dists.mean())
    max_dist = float(dists.max())
    
    results.append({
        'probe_idx': int(pidx),
        'longitude': float(coords[pidx, 0]),
        'latitude': float(coords[pidx, 1]),
        'local_pr': float(local_pr),
        'local_n80': local_n80,
        'local_n90': local_n90,
        'local_pc1_var': float(local_var[0]),
        'local_pc2_var': float(local_var[1]) if len(local_var) > 1 else 0,
        'align_global_pc1': align_pc1,
        'align_global_pc2': align_pc2,
        'dominant_dim': dominant_dim,
        'dominant_dim_weight': dominant_weight,
        'dominant_dim_var': dim_to_var.get(dominant_dim, '?'),
        'dominant_dim_cat': dim_to_cat.get(dominant_dim, '?'),
        'top3_dims': ','.join(top3_dims),
        'top3_weights': ','.join([f'{w:.4f}' for w in top3_weights]),
        'mean_nn_dist': mean_dist,
        'max_nn_dist': max_dist,
    })

elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s ({elapsed/len(probe_indices)*1000:.1f}ms per probe)")

results_df = pd.DataFrame(results)

# Add elevation if available
if 'elevation' in df_full.columns:
    results_df['elevation'] = df_full['elevation'].values[probe_indices]

results_df.to_csv(f'{OUTPUT_DIR}/local_pca_results.csv', index=False)
print(f"  ✓ Saved {len(results_df):,} probe results")


# ═══════════════════════════════════════════════════════════════════════════
# 4. TANGENT SPACE STABILITY — PAIRWISE ANGLES
# ═══════════════════════════════════════════════════════════════════════════

print(f"\nComputing tangent space stability ({K_TANGENT_PAIRS} neighbor pairs)...")

# For each probe, compute angle between its local PC1 and its neighbors' local PC1
# This measures how quickly the tangent space rotates — high angle = curvature

# First, store local PC1 directions for all probes
local_pc1_dirs = np.zeros((len(results), N_DIMS))
for i, pidx in enumerate(probe_indices):
    point = E[pidx].reshape(1, -1)
    _, nbr_idx = nn.kneighbors(point)
    nbr_idx = nbr_idx[0, 1:]
    E_local = E[nbr_idx]
    pca_local = PCA(n_components=2)
    pca_local.fit(E_local)
    local_pc1_dirs[i] = pca_local.components_[0]

# Build a spatial k-NN among probe locations themselves
probe_coords = results_df[['longitude', 'latitude']].values
nn_probes = NearestNeighbors(n_neighbors=6, algorithm='auto', metric='euclidean')
nn_probes.fit(probe_coords)
_, probe_nbr_idx = nn_probes.kneighbors(probe_coords)

# Compute angle between each probe and its nearest probe neighbors
tangent_angles = np.zeros(len(probe_indices))
for i in range(len(probe_indices)):
    angles = []
    for j in probe_nbr_idx[i, 1:]:  # skip self
        cos_sim = np.abs(np.dot(local_pc1_dirs[i], local_pc1_dirs[j]))
        cos_sim = np.clip(cos_sim, 0, 1)
        angle_deg = np.degrees(np.arccos(cos_sim))
        angles.append(angle_deg)
    tangent_angles[i] = np.mean(angles)

results_df['tangent_angle_deg'] = tangent_angles
results_df.to_csv(f'{OUTPUT_DIR}/local_pca_results.csv', index=False)

print(f"  Tangent angle stats: mean={tangent_angles.mean():.1f}°, "
      f"std={tangent_angles.std():.1f}°, "
      f"range=[{tangent_angles.min():.1f}°, {tangent_angles.max():.1f}°]")


# ═══════════════════════════════════════════════════════════════════════════
# 5. BUILD GEOMETRIC DICTIONARY
# ═══════════════════════════════════════════════════════════════════════════

print("\nBuilding geometric dictionary...")

# Aggregate statistics per dominant dimension
dim_geo_stats = {}
for dim in AE_COLS:
    mask = results_df['dominant_dim'] == dim
    if mask.sum() < 5:
        continue
    sub = results_df[mask]
    dim_geo_stats[dim] = {
        'count': int(mask.sum()),
        'fraction': float(mask.sum() / len(results_df)),
        'mean_alignment_pc1': float(sub['align_global_pc1'].mean()),
        'mean_tangent_angle': float(sub['tangent_angle_deg'].mean()),
        'mean_local_pr': float(sub['local_pr'].mean()),
        'primary_variable': dim_to_var.get(dim, '?'),
        'category': dim_to_cat.get(dim, '?'),
        'spearman_rho': float(dim_to_rho.get(dim, 0)),
    }

# Global summary
geo_dict = {
    'global': {
        'mean_local_pr': float(results_df['local_pr'].mean()),
        'std_local_pr': float(results_df['local_pr'].std()),
        'mean_align_pc1': float(results_df['align_global_pc1'].mean()),
        'mean_align_pc2': float(results_df['align_global_pc2'].mean()),
        'mean_tangent_angle': float(tangent_angles.mean()),
        'std_tangent_angle': float(tangent_angles.std()),
        'n_probes': len(results_df),
    },
    'per_dimension': dim_geo_stats,
}

with open(f'{OUTPUT_DIR}/geometric_dictionary.json', 'w') as f:
    json.dump(geo_dict, f, indent=2)
print(f"  ✓ Saved geometric dictionary ({len(dim_geo_stats)} dimensions with dominance)")


# ═══════════════════════════════════════════════════════════════════════════
# 6. DIAGNOSTIC FIGURES
# ═══════════════════════════════════════════════════════════════════════════

print("\nGenerating diagnostic figures...")

# --- 6a: Alignment of local PC1 with global PC1 ---
fig, ax = plt.subplots(figsize=(14, 8))
sc = ax.scatter(results_df['longitude'], results_df['latitude'],
                c=results_df['align_global_pc1'], s=2, cmap='RdYlGn',
                vmin=0, vmax=1, alpha=0.8, rasterized=True)
plt.colorbar(sc, ax=ax, shrink=0.7, label='|cos(local PC1, global PC1)|')
ax.set_xlim(CONUS_EXTENT[0], CONUS_EXTENT[1])
ax.set_ylim(CONUS_EXTENT[2], CONUS_EXTENT[3])
ax.set_aspect('equal')
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
ax.set_title(f'Alignment of Local PC1 with Global PC1 (Moisture–Vegetation)')
mean_align = results_df['align_global_pc1'].mean()
ax.annotate(f'Mean alignment = {mean_align:.3f}', xy=(0.02, 0.05),
            xycoords='axes fraction', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_alignment_map.png', dpi=300, facecolor='white')
plt.close(fig)
print("  ✓ Alignment map")

# --- 6b: Tangent space angle (curvature proxy) ---
fig, ax = plt.subplots(figsize=(14, 8))
vmin_t, vmax_t = np.percentile(tangent_angles, [2, 98])
sc = ax.scatter(results_df['longitude'], results_df['latitude'],
                c=tangent_angles, s=2, cmap='hot_r',
                vmin=vmin_t, vmax=vmax_t, alpha=0.8, rasterized=True)
plt.colorbar(sc, ax=ax, shrink=0.7, label='Mean Tangent Angle (°)')
ax.set_xlim(CONUS_EXTENT[0], CONUS_EXTENT[1])
ax.set_ylim(CONUS_EXTENT[2], CONUS_EXTENT[3])
ax.set_aspect('equal')
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
ax.set_title('Tangent Space Instability (Curvature Proxy)')
ax.annotate(f'Mean = {tangent_angles.mean():.1f}° ± {tangent_angles.std():.1f}°',
            xy=(0.02, 0.05), xycoords='axes fraction', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_tangent_stability.png', dpi=300, facecolor='white')
plt.close(fig)
print("  ✓ Tangent stability map")

# --- 6c: Local PR vs global PR comparison ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(results_df['local_pr'], bins=60, color='steelblue', alpha=0.7,
         density=True, edgecolor='none')
global_pr = float(pd.read_csv(eig_path)['eigenvalue'].pipe(
    lambda x: (x.sum()**2) / (x**2).sum())) if os.path.exists(eig_path) else 0
ax1.axvline(x=results_df['local_pr'].mean(), color='red', lw=2,
            label=f'Local mean = {results_df["local_pr"].mean():.1f}')
if global_pr > 0:
    ax1.axvline(x=global_pr, color='green', lw=2, ls=':',
                label=f'Global PR = {global_pr:.1f}')
ax1.set_xlabel('Local Participation Ratio')
ax1.set_ylabel('Density')
ax1.set_title('Local vs. Global Effective Dimensionality')
ax1.legend(fontsize=9)

# Alignment histogram
ax2.hist(results_df['align_global_pc1'], bins=60, color='coral', alpha=0.7,
         density=True, edgecolor='none', label='PC1 alignment')
ax2.hist(results_df['align_global_pc2'], bins=60, color='steelblue', alpha=0.5,
         density=True, edgecolor='none', label='PC2 alignment')
ax2.set_xlabel('|cos(local PC, global PC)|')
ax2.set_ylabel('Density')
ax2.set_title('Alignment of Local with Global Principal Components')
ax2.legend(fontsize=9)
ax2.axvline(x=1/np.sqrt(N_DIMS), color='gray', ls='--', lw=1,
            label=f'Random baseline = {1/np.sqrt(N_DIMS):.3f}')
ax2.legend(fontsize=9)

plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_local_vs_global_pca.png', dpi=300, facecolor='white')
plt.close(fig)
print("  ✓ Local vs global PCA")

# --- 6d: Dominant dimension map ---
# Color each probe by which environmental CATEGORY dominates locally
CATEGORY_COLORS = {
    'Terrain': '#8B4513', 'Soil': '#DAA520', 'Vegetation': '#228B22',
    'Temperature': '#DC143C', 'Climate': '#4169E1', 'Hydrology': '#00CED1',
    'Urban': '#696969', 'Radiation': '#FFD700', '?': '#CCCCCC',
}

fig, ax = plt.subplots(figsize=(14, 8))
for cat, color in CATEGORY_COLORS.items():
    mask = results_df['dominant_dim_cat'] == cat
    if mask.sum() == 0:
        continue
    sub = results_df[mask]
    ax.scatter(sub['longitude'], sub['latitude'], c=color, s=2,
               alpha=0.7, label=f'{cat} ({mask.sum()})', rasterized=True)
ax.set_xlim(CONUS_EXTENT[0], CONUS_EXTENT[1])
ax.set_ylim(CONUS_EXTENT[2], CONUS_EXTENT[3])
ax.set_aspect('equal')
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
ax.set_title('Locally Dominant Environmental Category (Local PC1)')
ax.legend(fontsize=7, ncol=2, loc='lower right',
          markerscale=5, title='Dominant Category')
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_dominant_dims_map.png', dpi=300, facecolor='white')
plt.close(fig)
print("  ✓ Dominant dimension map")

# --- 6e: Tangent angle vs alignment scatter ---
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(results_df['align_global_pc1'], results_df['tangent_angle_deg'],
                c=results_df['local_pr'], cmap='viridis', s=3, alpha=0.5, rasterized=True)
plt.colorbar(sc, ax=ax, label='Local PR')
ax.set_xlabel('Alignment with Global PC1')
ax.set_ylabel('Tangent Angle (°)')
ax.set_title('Alignment vs. Curvature (colored by local complexity)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_alignment_vs_curvature.png', dpi=300, facecolor='white')
plt.close(fig)
print("  ✓ Alignment vs curvature scatter")


# ═══════════════════════════════════════════════════════════════════════════
# 7. PUBLICATION FIGURE — Publication figure 4
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PUBLICATION FIGURE (Publication figure 4)")
print("-" * 60)

fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(2, 3, hspace=0.30, wspace=0.25)

# (a) Alignment of local PC1 with global PC1
ax_a = fig.add_subplot(gs[0, 0])
sc_a = ax_a.scatter(results_df['longitude'], results_df['latitude'],
                     c=results_df['align_global_pc1'], s=1.5, cmap='RdYlGn',
                     vmin=0, vmax=1, alpha=0.8, rasterized=True)
plt.colorbar(sc_a, ax=ax_a, shrink=0.8, label='|cos θ|')
ax_a.set_xlim(CONUS_EXTENT[0], CONUS_EXTENT[1])
ax_a.set_ylim(CONUS_EXTENT[2], CONUS_EXTENT[3])
ax_a.set_aspect('equal'); ax_a.set_xlabel('Lon'); ax_a.set_ylabel('Lat')
ax_a.set_title('(a) Local–Global PC1 Alignment', fontweight='bold', fontsize=12)
ax_a.tick_params(labelsize=8)

# (b) Tangent space instability
ax_b = fig.add_subplot(gs[0, 1])
sc_b = ax_b.scatter(results_df['longitude'], results_df['latitude'],
                     c=tangent_angles, s=1.5, cmap='hot_r',
                     vmin=vmin_t, vmax=vmax_t, alpha=0.8, rasterized=True)
plt.colorbar(sc_b, ax=ax_b, shrink=0.8, label='Angle (°)')
ax_b.set_xlim(CONUS_EXTENT[0], CONUS_EXTENT[1])
ax_b.set_ylim(CONUS_EXTENT[2], CONUS_EXTENT[3])
ax_b.set_aspect('equal'); ax_b.set_xlabel('Lon'); ax_b.set_ylabel('Lat')
ax_b.set_title('(b) Tangent Space Instability', fontweight='bold', fontsize=12)
ax_b.tick_params(labelsize=8)

# (c) Dominant environmental category
ax_c = fig.add_subplot(gs[0, 2])
for cat, color in CATEGORY_COLORS.items():
    mask = results_df['dominant_dim_cat'] == cat
    if mask.sum() == 0:
        continue
    sub = results_df[mask]
    ax_c.scatter(sub['longitude'], sub['latitude'], c=color, s=1.5,
                 alpha=0.7, label=cat, rasterized=True)
ax_c.set_xlim(CONUS_EXTENT[0], CONUS_EXTENT[1])
ax_c.set_ylim(CONUS_EXTENT[2], CONUS_EXTENT[3])
ax_c.set_aspect('equal'); ax_c.set_xlabel('Lon'); ax_c.set_ylabel('Lat')
ax_c.set_title('(c) Locally Dominant Category', fontweight='bold', fontsize=12)
ax_c.legend(fontsize=6, ncol=2, loc='lower right', markerscale=4)
ax_c.tick_params(labelsize=8)

# (d) Local PR distribution + global reference
ax_d = fig.add_subplot(gs[1, 0])
ax_d.hist(results_df['local_pr'], bins=60, color='steelblue', alpha=0.7,
          density=True, edgecolor='none')
ax_d.axvline(x=results_df['local_pr'].mean(), color='red', lw=2,
             label=f'Local mean = {results_df["local_pr"].mean():.1f}')
if global_pr > 0:
    ax_d.axvline(x=global_pr, color='green', lw=2, ls=':',
                 label=f'Global PR = {global_pr:.1f}')
ax_d.set_xlabel('Local Participation Ratio')
ax_d.set_ylabel('Density')
ax_d.set_title('(d) Local vs. Global Dimensionality', fontweight='bold', fontsize=12)
ax_d.legend(fontsize=8)

# (e) Alignment distributions
ax_e = fig.add_subplot(gs[1, 1])
ax_e.hist(results_df['align_global_pc1'], bins=50, color='coral', alpha=0.7,
          density=True, edgecolor='none', label='PC1 (moisture)')
ax_e.hist(results_df['align_global_pc2'], bins=50, color='steelblue', alpha=0.5,
          density=True, edgecolor='none', label='PC2 (temperature)')
ax_e.axvline(x=1/np.sqrt(N_DIMS), color='gray', ls='--', lw=1, label='Random baseline')
ax_e.set_xlabel('|cos(local PC, global PC)|')
ax_e.set_ylabel('Density')
ax_e.set_title('(e) Local–Global Alignment', fontweight='bold', fontsize=12)
ax_e.legend(fontsize=8)

# (f) Alignment vs tangent angle
ax_f = fig.add_subplot(gs[1, 2])
sc_f = ax_f.scatter(results_df['align_global_pc1'], tangent_angles,
                     c=results_df['local_pr'], cmap='viridis', s=2, alpha=0.4, rasterized=True)
plt.colorbar(sc_f, ax=ax_f, shrink=0.8, label='Local PR')
ax_f.set_xlabel('PC1 Alignment')
ax_f.set_ylabel('Tangent Angle (°)')
ax_f.set_title('(f) Alignment vs. Curvature', fontweight='bold', fontsize=12)
ax_f.grid(True, alpha=0.3)

fig.savefig(f'{FIG_DIR}/fig4_local_geometry.png', dpi=300, facecolor='white', bbox_inches='tight')
fig.savefig(f'{FIG_DIR}/fig4_local_geometry.pdf', dpi=300, facecolor='white', bbox_inches='tight')
plt.close(fig)
print("  ✓ Figure 4 saved")


# ═══════════════════════════════════════════════════════════════════════════
# 8. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SUMMARY: LOCAL PCA AND TANGENT SPACES")
print("-" * 60)

print(f"\n  PROBE LOCATIONS: {len(results_df):,} (stratified by elevation)")
print(f"  NEIGHBORS PER PROBE: {K_LOCAL_PCA}")

print(f"\n  LOCAL DIMENSIONALITY:")
print(f"    Local PR: {results_df['local_pr'].mean():.1f} ± {results_df['local_pr'].std():.1f}")
print(f"    Local 90%: {results_df['local_n90'].mean():.1f} ± {results_df['local_n90'].std():.1f}")
if global_pr > 0:
    print(f"    Global PR: {global_pr:.1f}")

print(f"\n  LOCAL–GLOBAL ALIGNMENT:")
print(f"    PC1 (moisture): {results_df['align_global_pc1'].mean():.3f} ± {results_df['align_global_pc1'].std():.3f}")
print(f"    PC2 (temperature): {results_df['align_global_pc2'].mean():.3f} ± {results_df['align_global_pc2'].std():.3f}")
print(f"    Random baseline: {1/np.sqrt(N_DIMS):.3f}")

print(f"\n  TANGENT SPACE STABILITY:")
print(f"    Mean angle: {tangent_angles.mean():.1f}° ± {tangent_angles.std():.1f}°")
flat_pct = (tangent_angles < 30).sum() / len(tangent_angles) * 100
curved_pct = (tangent_angles > 60).sum() / len(tangent_angles) * 100
print(f"    Flat (<30°): {flat_pct:.1f}% of locations")
print(f"    Curved (>60°): {curved_pct:.1f}% of locations")

print(f"\n  DOMINANT LOCAL CATEGORY:")
cat_counts = results_df['dominant_dim_cat'].value_counts()
for cat, count in cat_counts.items():
    print(f"    {cat}: {count} ({count/len(results_df)*100:.1f}%)")

print(f"\n  OUTPUTS:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    fp = os.path.join(OUTPUT_DIR, f)
    if os.path.isfile(fp) and ('local_pca' in f or 'geometric' in f or 'fig4' in f or 'fig_alignment' in f or 'fig_tangent' in f or 'fig_dominant' in f):
        print(f"    {f} ({os.path.getsize(fp)/1024:.1f} KB)")

print("\n" + "=" * 70)
print("Complete. Key outputs:")
print("  → geometric_dictionary.json")
print("  → local_pca_results.csv — per-location geometric metadata")
print("  → Tangent angle map — identifies manifold boundaries")
print("  → Dominant category map — where does local physics differ from global?")
print("-" * 60)

del E, nn
gc.collect()
print("Done.")
