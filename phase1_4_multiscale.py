"""
Multi-Scale Geometric Analysis

Repeats local PCA at k = 20, 100, 500, 2000 at 10K probe locations to track:
  1. Local-global PC alignment as a function of neighborhood size
  2. Tangent space stability across scales
  3. Dominant environmental category at each scale
  4. Local dimensionality as a function of k

Identifies the crossover scale where local geometry recovers alignment
with continental-scale gradients.

Input:  Yearly parquet files + probe locations from local PCA step
Output: manifold_results/ (multiscale results, figures)

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
import pyarrow.parquet as pq
import json, os, gc, time, warnings
warnings.filterwarnings('ignore')

# Try cartopy for basemaps
HAS_CARTOPY = False
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
    print("  ✓ cartopy available — maps will have coastlines/borders")
except ImportError:
    print("  ⚠ cartopy not installed — plain scatter maps")

# ── Config ─────────────────────────────────────────────────────────────────
DATA_DIR = '../../data/unified_conus'
RESULTS_DIR = 'manifold_results'
OUTPUT_DIR = 'manifold_results'
FIG_DIR = f'{OUTPUT_DIR}/figures'

YEARS = list(range(2017, 2024))
SUBSAMPLE_TOTAL = 200_000
N_PROBES = 10_000
K_SCALES = [20, 100, 500, 2000]
MAX_K = max(K_SCALES) + 1
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
print(" MULTI-SCALE GEOMETRIC ANALYSIS")
print(f"  Scales: k = {K_SCALES}")
print("-" * 60)


# ── Map helper ─────────────────────────────────────────────────────────────

def make_map_axis(fig, subplot_spec):
    """Create a map axis with cartopy basemap if available, else plain."""
    if HAS_CARTOPY:
        ax = fig.add_subplot(subplot_spec, projection=ccrs.PlateCarree())
        ax.set_extent([CONUS_EXTENT[0], CONUS_EXTENT[1],
                       CONUS_EXTENT[2], CONUS_EXTENT[3]], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black')
        ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='gray')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
        return ax
    else:
        ax = fig.add_subplot(subplot_spec)
        ax.set_xlim(CONUS_EXTENT[0], CONUS_EXTENT[1])
        ax.set_ylim(CONUS_EXTENT[2], CONUS_EXTENT[3])
        ax.set_aspect('equal')
        return ax

def make_map_axis_plain(fig, pos):
    """For plt.subplots-style axes."""
    if HAS_CARTOPY:
        ax = fig.add_axes(pos, projection=ccrs.PlateCarree())
        ax.set_extent([CONUS_EXTENT[0], CONUS_EXTENT[1],
                       CONUS_EXTENT[2], CONUS_EXTENT[3]], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black')
        ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='gray')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
        return ax
    else:
        ax = fig.add_axes(pos)
        ax.set_xlim(CONUS_EXTENT[0], CONUS_EXTENT[1])
        ax.set_ylim(CONUS_EXTENT[2], CONUS_EXTENT[3])
        ax.set_aspect('equal')
        return ax


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════

def load_data():
    per_year = SUBSAMPLE_TOTAL // len(YEARS)
    rng = np.random.default_rng(SEED)

    fp0 = f'{DATA_DIR}/conus_{YEARS[0]}_unified.parquet'
    all_cols = pq.read_schema(fp0).names
    env_wanted = ['elevation', 'temp_mean_c', 'precip_annual_mm', 'ndvi_mean',
                  'evi_mean', 'tree_cover_2000', 'lst_day_c']
    env_found = [c for c in env_wanted if c in all_cols]
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
    print(f"  Combined: {E.shape[0]:,} × {E.shape[1]}")
    return E, coords, df, env_found

E, coords, df_full, env_cols = load_data()


# ── Load global eigenvectors ──
print("\nLoading global eigenvectors...")
evec_path = f'{RESULTS_DIR}/eigenvectors.csv'
eig_path = f'{RESULTS_DIR}/eigenvalues.csv'
global_evecs = None
if os.path.exists(evec_path):
    global_evecs = pd.read_csv(evec_path, index_col=0).values
    print(f"  ✓ {global_evecs.shape}")

# ── Load dimension dictionary ──
dd_path = '../results/dimension_dictionary.csv'
dim_to_var, dim_to_cat = {}, {}
if os.path.exists(dd_path):
    dd = pd.read_csv(dd_path)
    for c in ['sp_primary', 'primary_variable']:
        if c in dd.columns: dim_to_var = dict(zip(dd['dimension'], dd[c])); break
    for c in ['sp_category', 'category']:
        if c in dd.columns: dim_to_cat = dict(zip(dd['dimension'], dd[c])); break
    print(f"  ✓ Dictionary: {len(dim_to_var)} dims")


# ═══════════════════════════════════════════════════════════════════════════
# 2. BUILD K-NN INDEX + SELECT PROBES
# ═══════════════════════════════════════════════════════════════════════════

print(f"\nBuilding k-NN index (max_k={MAX_K})...")
t0 = time.time()
nn = NearestNeighbors(n_neighbors=MAX_K, algorithm='auto', metric='euclidean')
nn.fit(E)
print(f"  Fitted in {time.time()-t0:.1f}s")

# Stratified probe selection (same seed as local PCA for consistency)
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


# ═══════════════════════════════════════════════════════════════════════════
# 3. MULTI-SCALE LOCAL PCA
# ═══════════════════════════════════════════════════════════════════════════

print(f"\nQuerying all {len(probe_indices):,} probe neighborhoods (k={MAX_K})...")
t0 = time.time()
all_dists, all_nbr_idx = nn.kneighbors(E[probe_indices])
all_nbr_idx = all_nbr_idx[:, 1:]  # drop self
print(f"  Queried in {time.time()-t0:.1f}s")

# For each scale k, compute local PCA at all probes
scale_results = {k: [] for k in K_SCALES}

for k in K_SCALES:
    print(f"\n  Computing local PCA at k={k}...")
    t0 = time.time()

    for i, pidx in enumerate(probe_indices):
        if i % 2000 == 0 and i > 0:
            print(f"    {i:,}/{len(probe_indices):,} ({time.time()-t0:.1f}s)")

        nbrs = all_nbr_idx[i, :k]
        E_local = E[nbrs]

        n_comp = min(k - 1, N_DIMS)
        pca_local = PCA(n_components=n_comp)
        pca_local.fit(E_local)

        local_var = pca_local.explained_variance_ratio_
        local_cumvar = np.cumsum(local_var)
        local_evecs = pca_local.components_
        local_eigenvalues = pca_local.explained_variance_

        # Effective dimensionality
        local_pr = float((local_eigenvalues.sum())**2 / (local_eigenvalues**2).sum())
        local_n90 = int(np.searchsorted(local_cumvar, 0.90) + 1) if len(local_cumvar) > 0 else n_comp

        # Alignment with global PCs
        align_pc1, align_pc2 = 0.0, 0.0
        if global_evecs is not None and local_evecs.shape[0] >= 2:
            align_pc1 = float(np.abs(np.dot(local_evecs[0], global_evecs[:, 0])))
            align_pc2 = float(np.abs(np.dot(local_evecs[1], global_evecs[:, 1])))

        # Dominant dimension
        dom_idx = int(np.argmax(np.abs(local_evecs[0])))
        dom_dim = AE_COLS[dom_idx]

        scale_results[k].append({
            'probe_idx': int(pidx),
            'longitude': float(coords[pidx, 0]),
            'latitude': float(coords[pidx, 1]),
            'local_pr': local_pr,
            'local_n90': local_n90,
            'local_pc1_var': float(local_var[0]),
            'align_global_pc1': align_pc1,
            'align_global_pc2': align_pc2,
            'dominant_dim': dom_dim,
            'dominant_cat': dim_to_cat.get(dom_dim, '?'),
        })

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")

# Convert to DataFrames
scale_dfs = {}
for k in K_SCALES:
    scale_dfs[k] = pd.DataFrame(scale_results[k])
    scale_dfs[k].to_csv(f'{OUTPUT_DIR}/multiscale_k{k}.csv', index=False)
print("\n  ✓ Saved all scale results")


# ═══════════════════════════════════════════════════════════════════════════
# 4. TANGENT ANGLES PER SCALE
# ═══════════════════════════════════════════════════════════════════════════

print("\nComputing tangent angles per scale...")

# Store local PC1 directions per scale
scale_pc1_dirs = {}
for k in K_SCALES:
    dirs = np.zeros((len(probe_indices), N_DIMS))
    for i, pidx in enumerate(probe_indices):
        nbrs = all_nbr_idx[i, :k]
        E_local = E[nbrs]
        pca_local = PCA(n_components=2)
        pca_local.fit(E_local)
        dirs[i] = pca_local.components_[0]
    scale_pc1_dirs[k] = dirs

# Spatial neighbors among probes
probe_coords = coords[probe_indices]
nn_probes = NearestNeighbors(n_neighbors=6, algorithm='auto', metric='euclidean')
nn_probes.fit(probe_coords)
_, probe_nbr_idx = nn_probes.kneighbors(probe_coords)

scale_tangent_angles = {}
for k in K_SCALES:
    angles = np.zeros(len(probe_indices))
    for i in range(len(probe_indices)):
        a_list = []
        for j in probe_nbr_idx[i, 1:]:
            cos_sim = np.clip(np.abs(np.dot(scale_pc1_dirs[k][i], scale_pc1_dirs[k][j])), 0, 1)
            a_list.append(np.degrees(np.arccos(cos_sim)))
        angles[i] = np.mean(a_list)
    scale_tangent_angles[k] = angles
    scale_dfs[k]['tangent_angle_deg'] = angles
    print(f"  k={k:4d}: tangent angle = {angles.mean():.1f}° ± {angles.std():.1f}°")


# ═══════════════════════════════════════════════════════════════════════════
# 5. AGGREGATE SCALE-DEPENDENT METRICS
# ═══════════════════════════════════════════════════════════════════════════

print("\nScale-dependent summary:")
summary_rows = []
for k in K_SCALES:
    df_k = scale_dfs[k]
    ta = scale_tangent_angles[k]

    cats = df_k['dominant_cat'].value_counts(normalize=True)
    top_cat = cats.index[0] if len(cats) > 0 else '?'
    top_cat_pct = cats.iloc[0] * 100 if len(cats) > 0 else 0

    row = {
        'k': k,
        'mean_align_pc1': df_k['align_global_pc1'].mean(),
        'std_align_pc1': df_k['align_global_pc1'].std(),
        'mean_align_pc2': df_k['align_global_pc2'].mean(),
        'mean_tangent_angle': ta.mean(),
        'std_tangent_angle': ta.std(),
        'pct_flat': (ta < 30).sum() / len(ta) * 100,
        'pct_curved': (ta > 60).sum() / len(ta) * 100,
        'mean_local_pr': df_k['local_pr'].mean(),
        'mean_local_n90': df_k['local_n90'].mean(),
        'mean_pc1_var': df_k['local_pc1_var'].mean(),
        'top_category': top_cat,
        'top_category_pct': top_cat_pct,
    }
    summary_rows.append(row)

    print(f"  k={k:4d}: align_PC1={row['mean_align_pc1']:.3f}, "
          f"tangent={row['mean_tangent_angle']:.1f}°, "
          f"PR={row['mean_local_pr']:.1f}, "
          f"top={top_cat} ({top_cat_pct:.0f}%)")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f'{OUTPUT_DIR}/multiscale_summary.csv', index=False)
print("  ✓ Saved multiscale_summary.csv")

RANDOM_BASELINE = 1.0 / np.sqrt(N_DIMS)


# ═══════════════════════════════════════════════════════════════════════════
# 6. DIAGNOSTIC FIGURES
# ═══════════════════════════════════════════════════════════════════════════

print("\nGenerating diagnostic figures...")

# --- 6a: Alignment vs scale ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(summary_df['k'], summary_df['mean_align_pc1'], 'o-', color='coral',
         lw=2, ms=8, label='PC1 (moisture)')
ax1.plot(summary_df['k'], summary_df['mean_align_pc2'], 's--', color='steelblue',
         lw=2, ms=8, label='PC2 (temperature)')
ax1.axhline(y=RANDOM_BASELINE, color='gray', ls='--', lw=1, label=f'Random = {RANDOM_BASELINE:.3f}')
ax1.fill_between(summary_df['k'],
                 summary_df['mean_align_pc1'] - summary_df['std_align_pc1'],
                 summary_df['mean_align_pc1'] + summary_df['std_align_pc1'],
                 alpha=0.15, color='coral')
ax1.set_xlabel('Neighborhood Size (k)')
ax1.set_ylabel('|cos(local PC, global PC)|')
ax1.set_title('Alignment Recovery with Scale')
ax1.set_xscale('log')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

ax2.plot(summary_df['k'], summary_df['mean_tangent_angle'], 'o-', color='firebrick',
         lw=2, ms=8, label='Mean tangent angle')
ax2.fill_between(summary_df['k'],
                 summary_df['mean_tangent_angle'] - summary_df['std_tangent_angle'],
                 summary_df['mean_tangent_angle'] + summary_df['std_tangent_angle'],
                 alpha=0.15, color='firebrick')
ax2.axhline(y=45, color='gray', ls=':', lw=1, label='45° reference')
ax2.set_xlabel('Neighborhood Size (k)')
ax2.set_ylabel('Tangent Angle (°)')
ax2.set_title('Tangent Stability with Scale')
ax2.set_xscale('log')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_multiscale_alignment.png', dpi=300, facecolor='white')
plt.close(fig)
print("  ✓ Alignment vs scale")

# --- 6b: Dimensionality vs scale ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(summary_df['k'], summary_df['mean_local_pr'], 'o-', color='steelblue',
        lw=2, ms=8, label='Local PR')
ax.plot(summary_df['k'], summary_df['mean_local_n90'], 's--', color='coral',
        lw=2, ms=8, label='Local 90% PCs')

global_pr = 0
if os.path.exists(eig_path):
    evals = pd.read_csv(eig_path)['eigenvalue'].values
    global_pr = (evals.sum())**2 / (evals**2).sum()
    ax.axhline(y=global_pr, color='green', ls=':', lw=1.5, label=f'Global PR = {global_pr:.1f}')

ax.set_xlabel('Neighborhood Size (k)')
ax.set_ylabel('Effective Dimensionality')
ax.set_title('Local Dimensionality vs. Scale')
ax.set_xscale('log')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_multiscale_dimensionality.png', dpi=300, facecolor='white')
plt.close(fig)
print("  ✓ Dimensionality vs scale")

# --- 6c: Dominant category vs scale (stacked bar) ---
CATEGORY_COLORS = {
    'Terrain': '#8B4513', 'Soil': '#DAA520', 'Vegetation': '#228B22',
    'Temperature': '#DC143C', 'Climate': '#4169E1', 'Hydrology': '#00CED1',
    'Urban': '#696969', 'Radiation': '#FFD700', '?': '#CCCCCC',
}
ALL_CATS = ['Temperature', 'Vegetation', 'Hydrology', 'Soil', 'Terrain', 'Climate', 'Radiation', 'Urban']

fig, ax = plt.subplots(figsize=(10, 6))
bottoms = np.zeros(len(K_SCALES))
for cat in ALL_CATS:
    pcts = []
    for k in K_SCALES:
        vc = scale_dfs[k]['dominant_cat'].value_counts(normalize=True)
        pcts.append(vc.get(cat, 0) * 100)
    pcts = np.array(pcts)
    ax.bar([str(k) for k in K_SCALES], pcts, bottom=bottoms,
           color=CATEGORY_COLORS.get(cat, '#CCC'), label=cat, edgecolor='none')
    bottoms += pcts
ax.set_xlabel('Neighborhood Size (k)')
ax.set_ylabel('Percentage (%)')
ax.set_title('Dominant Local Category vs. Scale')
ax.legend(fontsize=8, ncol=2, loc='upper right')
ax.set_ylim(0, 105)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_multiscale_categories.png', dpi=300, facecolor='white')
plt.close(fig)
print("  ✓ Category vs scale")

# --- 6d: Alignment CONUS maps at each scale ---
fig = plt.figure(figsize=(6 * len(K_SCALES), 5))
for i, k in enumerate(K_SCALES):
    if HAS_CARTOPY:
        ax = fig.add_subplot(1, len(K_SCALES), i + 1, projection=ccrs.PlateCarree())
        ax.set_extent([CONUS_EXTENT[0], CONUS_EXTENT[1],
                       CONUS_EXTENT[2], CONUS_EXTENT[3]])
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.2, edgecolor='gray')
    else:
        ax = fig.add_subplot(1, len(K_SCALES), i + 1)
        ax.set_xlim(CONUS_EXTENT[0], CONUS_EXTENT[1])
        ax.set_ylim(CONUS_EXTENT[2], CONUS_EXTENT[3])
        ax.set_aspect('equal')

    df_k = scale_dfs[k]
    sc = ax.scatter(df_k['longitude'], df_k['latitude'],
                    c=df_k['align_global_pc1'], s=1.5, cmap='RdYlGn',
                    vmin=0, vmax=0.8, alpha=0.8, rasterized=True,
                    transform=ccrs.PlateCarree() if HAS_CARTOPY else None)
    ax.set_title(f'k={k}\n(align={df_k["align_global_pc1"].mean():.3f})',
                 fontweight='bold', fontsize=11)
    ax.tick_params(labelsize=6)

plt.colorbar(sc, ax=fig.axes, shrink=0.8, label='PC1 Alignment', pad=0.02)
plt.suptitle('Local–Global PC1 Alignment Across Scales', fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_multiscale_alignment_maps.png', dpi=300,
            facecolor='white', bbox_inches='tight')
plt.close(fig)
print("  ✓ Multi-scale alignment maps")


# ═══════════════════════════════════════════════════════════════════════════
# 7. PUBLICATION FIGURE — Publication figure 5
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PUBLICATION FIGURE (Publication figure 5)")
print("-" * 60)

fig = plt.figure(figsize=(18, 16))
gs = gridspec.GridSpec(3, 2, hspace=0.32, wspace=0.25, height_ratios=[1, 1, 0.9])

# ── (a) Alignment vs scale ──
ax_a = fig.add_subplot(gs[0, 0])
ax_a.plot(summary_df['k'], summary_df['mean_align_pc1'], 'o-', color='coral',
          lw=2.5, ms=10, label='PC1 (moisture–vegetation)', zorder=3)
ax_a.plot(summary_df['k'], summary_df['mean_align_pc2'], 's--', color='steelblue',
          lw=2, ms=9, label='PC2 (temperature)', zorder=3)
ax_a.axhline(y=RANDOM_BASELINE, color='gray', ls='--', lw=1.2,
             label=f'Random baseline ({RANDOM_BASELINE:.3f})')
ax_a.fill_between(summary_df['k'],
                  summary_df['mean_align_pc1'] - summary_df['std_align_pc1'],
                  summary_df['mean_align_pc1'] + summary_df['std_align_pc1'],
                  alpha=0.12, color='coral')
ax_a.set_xlabel('Neighborhood Size (k)')
ax_a.set_ylabel('|cos(local PC, global PC)|')
ax_a.set_title('(a) Local–Global Alignment vs. Scale', fontweight='bold', fontsize=13)
ax_a.set_xscale('log')
ax_a.legend(fontsize=9)
ax_a.grid(True, alpha=0.3)

# ── (b) Tangent angle + dimensionality vs scale ──
ax_b = fig.add_subplot(gs[0, 1])
ax_b_twin = ax_b.twinx()

l1 = ax_b.plot(summary_df['k'], summary_df['mean_tangent_angle'], 'o-',
               color='firebrick', lw=2.5, ms=10, label='Tangent angle (°)')
ax_b.fill_between(summary_df['k'],
                  summary_df['mean_tangent_angle'] - summary_df['std_tangent_angle'],
                  summary_df['mean_tangent_angle'] + summary_df['std_tangent_angle'],
                  alpha=0.12, color='firebrick')
ax_b.set_ylabel('Tangent Angle (°)', color='firebrick')
ax_b.tick_params(axis='y', labelcolor='firebrick')

l2 = ax_b_twin.plot(summary_df['k'], summary_df['mean_local_pr'], 's--',
                     color='steelblue', lw=2, ms=9, label='Local PR')
if global_pr > 0:
    ax_b_twin.axhline(y=global_pr, color='green', ls=':', lw=1.5, label=f'Global PR ({global_pr:.1f})')
ax_b_twin.set_ylabel('Participation Ratio', color='steelblue')
ax_b_twin.tick_params(axis='y', labelcolor='steelblue')

lines = l1 + l2
labels_leg = [l.get_label() for l in lines]
ax_b.legend(lines, labels_leg, fontsize=8, loc='upper right')
ax_b.set_xlabel('Neighborhood Size (k)')
ax_b.set_title('(b) Curvature and Dimensionality vs. Scale', fontweight='bold', fontsize=13)
ax_b.set_xscale('log')
ax_b.grid(True, alpha=0.3)

# ── (c) & (d) Alignment CONUS maps: k=20 vs k=2000 ──
for panel_idx, (k_val, panel_label) in enumerate([(K_SCALES[0], '(c)'), (K_SCALES[-1], '(d)')]):
    if HAS_CARTOPY:
        ax = fig.add_subplot(gs[1, panel_idx], projection=ccrs.PlateCarree())
        ax.set_extent([CONUS_EXTENT[0], CONUS_EXTENT[1],
                       CONUS_EXTENT[2], CONUS_EXTENT[3]])
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color='black')
        ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='gray')
        transform = ccrs.PlateCarree()
    else:
        ax = fig.add_subplot(gs[1, panel_idx])
        ax.set_xlim(CONUS_EXTENT[0], CONUS_EXTENT[1])
        ax.set_ylim(CONUS_EXTENT[2], CONUS_EXTENT[3])
        ax.set_aspect('equal')
        transform = None

    df_k = scale_dfs[k_val]
    scatter_kw = dict(c=df_k['align_global_pc1'], s=2, cmap='RdYlGn',
                      vmin=0, vmax=0.8, alpha=0.8, rasterized=True)
    if transform:
        scatter_kw['transform'] = transform
    sc = ax.scatter(df_k['longitude'], df_k['latitude'], **scatter_kw)
    plt.colorbar(sc, ax=ax, shrink=0.7, label='|cos θ|', pad=0.02)
    m_align = df_k['align_global_pc1'].mean()
    ax.set_title(f'{panel_label} PC1 Alignment at k={k_val} (mean={m_align:.3f})',
                 fontweight='bold', fontsize=12)

# ── (e) Dominant category stacked bar ──
ax_e = fig.add_subplot(gs[2, 0])
bottoms_e = np.zeros(len(K_SCALES))
for cat in ALL_CATS:
    pcts = []
    for k in K_SCALES:
        vc = scale_dfs[k]['dominant_cat'].value_counts(normalize=True)
        pcts.append(vc.get(cat, 0) * 100)
    pcts = np.array(pcts)
    ax_e.bar([str(k) for k in K_SCALES], pcts, bottom=bottoms_e,
             color=CATEGORY_COLORS.get(cat, '#CCC'), label=cat, edgecolor='none')
    bottoms_e += pcts
ax_e.set_xlabel('Neighborhood Size (k)')
ax_e.set_ylabel('Percentage (%)')
ax_e.set_title('(e) Dominant Category vs. Scale', fontweight='bold', fontsize=13)
ax_e.legend(fontsize=7, ncol=4, loc='upper center')
ax_e.set_ylim(0, 105)

# ── (f) PC1 variance explained vs scale ──
ax_f = fig.add_subplot(gs[2, 1])
pc1_vars = [scale_dfs[k]['local_pc1_var'].mean() * 100 for k in K_SCALES]
pc1_stds = [scale_dfs[k]['local_pc1_var'].std() * 100 for k in K_SCALES]
ax_f.bar([str(k) for k in K_SCALES], pc1_vars, yerr=pc1_stds, capsize=5,
         color='steelblue', alpha=0.8, edgecolor='none')
ax_f.set_xlabel('Neighborhood Size (k)')
ax_f.set_ylabel('Variance Explained by Local PC1 (%)')
ax_f.set_title('(f) Local PC1 Dominance vs. Scale', fontweight='bold', fontsize=13)
ax_f.grid(True, alpha=0.3, axis='y')

fig.savefig(f'{FIG_DIR}/fig5_multiscale_geometry.png', dpi=300,
            facecolor='white', bbox_inches='tight')
fig.savefig(f'{FIG_DIR}/fig5_multiscale_geometry.pdf', dpi=300,
            facecolor='white', bbox_inches='tight')
plt.close(fig)
print("  ✓ Figure 5 saved")


# ═══════════════════════════════════════════════════════════════════════════
# 8. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SUMMARY: MULTI-SCALE GEOMETRY")
print("-" * 60)

print(f"\n  PROBES: {len(probe_indices):,}")
print(f"  SCALES: k = {K_SCALES}")

print(f"\n  ALIGNMENT WITH GLOBAL PC1 (moisture–vegetation):")
for _, row in summary_df.iterrows():
    print(f"    k={int(row['k']):4d}: {row['mean_align_pc1']:.3f} ± {row['std_align_pc1']:.3f}")
print(f"    Random baseline: {RANDOM_BASELINE:.3f}")

print(f"\n  TANGENT STABILITY:")
for _, row in summary_df.iterrows():
    print(f"    k={int(row['k']):4d}: {row['mean_tangent_angle']:.1f}° "
          f"(flat: {row['pct_flat']:.1f}%, curved: {row['pct_curved']:.1f}%)")

print(f"\n  LOCAL DIMENSIONALITY:")
for _, row in summary_df.iterrows():
    print(f"    k={int(row['k']):4d}: PR={row['mean_local_pr']:.1f}, 90%→{row['mean_local_n90']:.1f}")
if global_pr > 0:
    print(f"    Global: PR={global_pr:.1f}")

print(f"\n  DOMINANT CATEGORY SHIFT:")
for _, row in summary_df.iterrows():
    print(f"    k={int(row['k']):4d}: {row['top_category']} ({row['top_category_pct']:.0f}%)")

# Identify arithmetic reliability scale
for _, row in summary_df.iterrows():
    if row['mean_align_pc1'] > 0.3:
        print(f"\n  ARITHMETIC RELIABILITY THRESHOLD:")
        print(f"    Alignment > 0.3 first achieved at k={int(row['k'])}")
        break
else:
    print(f"\n  ARITHMETIC RELIABILITY THRESHOLD:")
    print(f"    Alignment never exceeds 0.3 at tested scales")
    print(f"    → Compositional arithmetic requires location-aware directions")

print(f"\n  OUTPUTS:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    fp = os.path.join(OUTPUT_DIR, f)
    if os.path.isfile(fp) and ('multiscale' in f or 'fig5' in f):
        print(f"    {f} ({os.path.getsize(fp)/1024:.1f} KB)")

print("\n" + "=" * 70)
print("Complete.")
print("  → If alignment recovers: use that k as minimum for arithmetic")
print("  → If alignment stays low: arithmetic must use local directions")
print("  → Category shift shows when continental gradients dominate")
print("-" * 60)

del E, nn, all_nbr_idx, scale_pc1_dirs
gc.collect()
print("Done.")
