"""
================================================================================
PHASE 2A — LINEAR PROBES FOR CONCEPT DIRECTIONS IN ALPHAEARTH EMBEDDINGS
================================================================================

Paper 2: Geometric Characterization of Satellite Foundation Model Embeddings

Motivation (Sam Barrett's suggestion):
    Phase 2 showed that compositional vector arithmetic fails using PCA-derived
    directions, and attributed this to manifold curvature. However, PCA directions
    are unsupervised and post-hoc: a single PC may not capture the true "concept
    direction" for a property like precipitation. A supervised linear probe
    (Ridge/Lasso regression from embeddings to environmental variables) finds the
    optimal linear direction for each property. If arithmetic still fails with
    probe-derived concept vectors, the failure is genuinely geometric (curvature)
    rather than methodological (poor direction estimation).

Experiments:
    1. Train linear probes at three spatial scales:
       - Global:   All CONUS data (one direction per property)
       - Regional: Within each of 5 CONUS subregions (one direction per region)
       - Local:    k=100 nearest neighbors at each source location
    2. Compare R² across scales (does local > regional > global?)
    3. Measure direction stability: cosine similarity between scales
    4. Run targeted shift experiments using probe-derived directions
    5. Compare probe-based shifts against PCA-based shifts and baselines

Author: Mashrekur Rahman | March 2026
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy import stats
import pyarrow.parquet as pq
import json, os, gc, time, warnings
warnings.filterwarnings('ignore')

# ── Config ─────────────────────────────────────────────────────────────────
DATA_DIR = '../../data/unified_conus'
PHASE1_DIR = 'manifold_results'
OUTPUT_DIR = 'manifold_results/linear_probes'
FIG_DIR = f'{OUTPUT_DIR}/figures'

YEARS = list(range(2017, 2024))
SUBSAMPLE_TOTAL = 500_000
N_SOURCE_LOCATIONS = 500
K_LOCAL = 100            # neighbors for local probes (same as phase2_arithmetic)
K_RETRIEVAL = 10
SHIFT_SIGMAS = [0.5, 1.0, 1.5, 2.0]
SEED = 42
RIDGE_ALPHA = 1.0
LASSO_ALPHA = 0.001      # light regularization for Lasso comparison

AE_COLS = [f'A{i:02d}' for i in range(64)]
N_DIMS = 64

# Same target properties as phase2_arithmetic.py
TARGET_PROPERTIES = {
    'precip_annual_mm': {'label': 'Precipitation', 'unit': 'mm/yr'},
    'temp_mean_c':      {'label': 'Temperature',   'unit': '°C'},
    'evi_mean':         {'label': 'EVI',            'unit': ''},
    'elevation':        {'label': 'Elevation',      'unit': 'm'},
}

# Regions from phase2b_retrieval_coherence.py
REGIONS = {
    'Pacific NW':    {'lon': (-125, -116), 'lat': (42, 49)},
    'Great Plains':  {'lon': (-104, -95),  'lat': (35, 48)},
    'Southeast':     {'lon': (-90, -75),   'lat': (25, 36)},
    'Mountain West': {'lon': (-115, -104), 'lat': (35, 45)},
    'Northeast':     {'lon': (-80, -67),   'lat': (39, 47)},
}

# Additional env columns for evaluating collateral changes
EXTRA_ENV = ['ndvi_mean', 'lst_day_c', 'tree_cover_2000',
             'soil_moisture', 'et_annual_mm', 'slope',
             'soil_ph', 'impervious_pct']

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 12,
    'axes.linewidth': 0.8, 'axes.labelsize': 14,
    'axes.titlesize': 16, 'axes.titleweight': 'bold',
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 70)
print("PHASE 2A: LINEAR PROBES FOR CONCEPT DIRECTIONS")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA (identical to phase2_arithmetic.py for reproducibility)
# ═══════════════════════════════════════════════════════════════════════════

def load_data():
    per_year = SUBSAMPLE_TOTAL // len(YEARS)
    rng = np.random.default_rng(SEED)

    fp0 = f'{DATA_DIR}/conus_{YEARS[0]}_unified.parquet'
    all_cols = pq.read_schema(fp0).names

    env_cols = [c for c in TARGET_PROPERTIES.keys() if c in all_cols]
    extra_env = [c for c in EXTRA_ENV if c in all_cols]
    load_cols = list(set(['longitude', 'latitude'] + AE_COLS + env_cols + extra_env))

    print(f"\nLoading: {per_year:,}/yr x {len(YEARS)} yrs = {per_year*len(YEARS):,}")

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
        print(f"  {year}: {len(df_y):,} -> {len(idx):,}")
        del df_y; gc.collect()

    df = pd.concat(frames, ignore_index=True)
    E = df[AE_COLS].values.astype(np.float64)
    coords = df[['longitude', 'latitude']].values

    env_stats = {}
    for col in env_cols + extra_env:
        if col in df.columns:
            vals = df[col].dropna()
            env_stats[col] = {'mean': float(vals.mean()), 'std': float(vals.std()),
                              'min': float(vals.min()), 'max': float(vals.max())}

    print(f"  Combined: {len(df):,}")
    print(f"  Env columns: {env_cols}")
    return E, coords, df, env_cols, extra_env, env_stats


E, coords, df_full, env_cols, extra_env, env_stats = load_data()
all_env = env_cols + [c for c in extra_env if c not in env_cols]


# ═══════════════════════════════════════════════════════════════════════════
# 2. BUILD K-NN INDEX
# ═══════════════════════════════════════════════════════════════════════════

print("\nBuilding k-NN index...")
t0 = time.time()
nn_index = NearestNeighbors(n_neighbors=K_LOCAL + K_RETRIEVAL + 1,
                             algorithm='auto', metric='euclidean')
nn_index.fit(E)
print(f"  Fitted in {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════
# 3. LOAD GLOBAL EIGENVECTORS (for PCA baseline comparison)
# ═══════════════════════════════════════════════════════════════════════════

print("\nLoading Phase 1 outputs...")
global_evecs = None
evec_path = f'{PHASE1_DIR}/eigenvectors.csv'
if os.path.exists(evec_path):
    global_evecs = pd.read_csv(evec_path, index_col=0).values
    print(f"  Global eigenvectors: {global_evecs.shape}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. SELECT SOURCE LOCATIONS (same seed as phase2_arithmetic.py)
# ═══════════════════════════════════════════════════════════════════════════

def select_sources(n=N_SOURCE_LOCATIONS):
    rng = np.random.default_rng(SEED + 1)
    if 'elevation' in df_full.columns:
        elev = df_full['elevation'].values
        bins = [-100, 100, 500, 1000, 2000, 5000]
        labels = ['<100m', '100-500m', '500-1000m', '1000-2000m', '>2000m']
        groups = pd.cut(elev, bins=bins, labels=labels)
        per_group = n // len(labels)
        sources = []
        for label in labels:
            gidx = np.where(groups == label)[0]
            valid = [i for i in gidx if all(
                np.isfinite(df_full[c].iloc[i]) for c in env_cols if c in df_full.columns)]
            sources.extend(rng.choice(valid, size=min(per_group, len(valid)), replace=False))
        sources = np.array(sources)
    else:
        valid = df_full.dropna(subset=env_cols).index.values
        sources = rng.choice(valid, size=min(n, len(valid)), replace=False)
    print(f"\n  Selected {len(sources)} source locations")
    return sources


source_indices = select_sources()


# ═══════════════════════════════════════════════════════════════════════════
# 5. ASSIGN REGIONS TO ALL SAMPLES
# ═══════════════════════════════════════════════════════════════════════════

print("\nAssigning regions...")
region_labels = np.full(len(df_full), '', dtype=object)
for rname, bounds in REGIONS.items():
    mask = (
        (coords[:, 0] >= bounds['lon'][0]) & (coords[:, 0] <= bounds['lon'][1]) &
        (coords[:, 1] >= bounds['lat'][0]) & (coords[:, 1] <= bounds['lat'][1])
    )
    region_labels[mask] = rname
    print(f"  {rname}: {mask.sum():,} samples")

# Source location region assignments
source_regions = region_labels[source_indices]
n_in_region = np.sum(source_regions != '')
print(f"  Source locations in defined regions: {n_in_region}/{len(source_indices)}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. TRAIN LINEAR PROBES AT THREE SCALES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("EXPERIMENT 1: LINEAR PROBES ACROSS SCALES")
print("=" * 70)

def train_probe(X, y, alpha=RIDGE_ALPHA, method='ridge'):
    """Train a linear probe and return concept direction, R², model."""
    valid = np.isfinite(y)
    if valid.sum() < 20:
        return None, None, None, None
    X_v, y_v = X[valid], y[valid]

    # Standardize embeddings (zero mean per feature)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_v)

    if method == 'ridge':
        model = Ridge(alpha=alpha, fit_intercept=True)
    else:
        model = Lasso(alpha=alpha, fit_intercept=True, max_iter=5000)

    model.fit(X_s, y_v)
    y_pred = model.predict(X_s)
    r2 = r2_score(y_v, y_pred)

    # Concept direction: model.coef_ in standardized space,
    # convert back to original space for direction (divide by std)
    coef_original = model.coef_ / scaler.scale_
    direction = coef_original / np.linalg.norm(coef_original)

    return direction, r2, model, scaler


# --- 6A: GLOBAL PROBES ---
print("\n  [Scale 1] Global probes (all CONUS)...")
global_probes = {}

for prop in TARGET_PROPERTIES:
    if prop not in df_full.columns:
        continue
    y = df_full[prop].values
    direction, r2, model, scaler = train_probe(E, y, method='ridge')
    if direction is not None:
        # Also train Lasso for comparison
        dir_lasso, r2_lasso, _, _ = train_probe(E, y, alpha=LASSO_ALPHA, method='lasso')

        # Compute sigma along this direction
        projections = E @ direction
        sigma = projections.std()

        global_probes[prop] = {
            'direction': direction,
            'r2_ridge': r2,
            'r2_lasso': r2_lasso if r2_lasso is not None else np.nan,
            'sigma': sigma,
            'n_samples': int(np.isfinite(y).sum()),
            'cosine_ridge_lasso': float(np.dot(direction, dir_lasso)) if dir_lasso is not None else np.nan,
        }
        print(f"    {TARGET_PROPERTIES[prop]['label']:15s}: "
              f"Ridge R2={r2:.4f}, Lasso R2={r2_lasso:.4f}, "
              f"cos(Ridge,Lasso)={global_probes[prop]['cosine_ridge_lasso']:.4f}")


# --- 6B: REGIONAL PROBES ---
print("\n  [Scale 2] Regional probes...")
regional_probes = {}  # {prop: {region: {...}}}

for prop in TARGET_PROPERTIES:
    if prop not in df_full.columns:
        continue
    regional_probes[prop] = {}

    for rname, bounds in REGIONS.items():
        mask = region_labels == rname
        if mask.sum() < 50:
            continue

        E_reg = E[mask]
        y_reg = df_full[prop].values[mask]
        direction, r2, model, scaler = train_probe(E_reg, y_reg, method='ridge')

        if direction is not None:
            sigma = (E_reg @ direction).std()
            # Cosine similarity with global direction
            cos_global = float(np.dot(direction, global_probes[prop]['direction']))

            regional_probes[prop][rname] = {
                'direction': direction,
                'r2': r2,
                'sigma': sigma,
                'n_samples': int(np.isfinite(y_reg).sum()),
                'cosine_with_global': cos_global,
            }

    # Print regional summary
    if regional_probes[prop]:
        r2s = [v['r2'] for v in regional_probes[prop].values()]
        cos_globals = [v['cosine_with_global'] for v in regional_probes[prop].values()]
        print(f"    {TARGET_PROPERTIES[prop]['label']:15s}: "
              f"R2 range [{min(r2s):.4f}, {max(r2s):.4f}], "
              f"cos(reg,global) range [{min(cos_globals):.4f}, {max(cos_globals):.4f}]")


# --- 6C: LOCAL PROBES (at each source location) ---
print(f"\n  [Scale 3] Local probes at {len(source_indices)} source locations...")
local_probes = {}  # {prop: {src_idx: {...}}}
t0 = time.time()

for prop in TARGET_PROPERTIES:
    if prop not in df_full.columns:
        continue
    local_probes[prop] = {}
    n_success = 0
    r2_list = []
    cos_global_list = []
    cos_regional_list = []

    for i, src_idx in enumerate(source_indices):
        if i % 100 == 0 and i > 0:
            print(f"      {prop}: {i}/{len(source_indices)} ({time.time()-t0:.1f}s)")

        # Get k-NN
        point = E[src_idx].reshape(1, -1)
        dists, nbr_idx = nn_index.kneighbors(point, n_neighbors=K_LOCAL + 1)
        nbr_idx = nbr_idx[0, 1:]  # exclude self

        E_local = E[nbr_idx]
        y_local = df_full[prop].values[nbr_idx]
        direction, r2, model, scaler = train_probe(E_local, y_local, method='ridge')

        if direction is not None:
            sigma = (E_local @ direction).std()
            cos_global = float(np.dot(direction, global_probes[prop]['direction']))
            cos_global_list.append(cos_global)

            # Cosine with regional direction if source is in a region
            src_region = region_labels[src_idx]
            cos_regional = np.nan
            if src_region != '' and src_region in regional_probes.get(prop, {}):
                reg_dir = regional_probes[prop][src_region]['direction']
                cos_regional = float(np.dot(direction, reg_dir))
                cos_regional_list.append(cos_regional)

            local_probes[prop][int(src_idx)] = {
                'direction': direction,
                'r2': r2,
                'sigma': sigma,
                'cosine_with_global': cos_global,
                'cosine_with_regional': cos_regional,
                'region': src_region,
            }
            r2_list.append(r2)
            n_success += 1

    print(f"    {TARGET_PROPERTIES[prop]['label']:15s}: "
          f"{n_success}/{len(source_indices)} successful, "
          f"R2 median={np.median(r2_list):.4f} [{np.percentile(r2_list,25):.4f}, {np.percentile(r2_list,75):.4f}], "
          f"cos(local,global) median={np.median(cos_global_list):.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# 7. COMPILE SCALE COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SCALE COMPARISON SUMMARY")
print("=" * 70)

scale_rows = []
for prop in TARGET_PROPERTIES:
    if prop not in global_probes:
        continue

    # Global
    scale_rows.append({
        'property': prop,
        'label': TARGET_PROPERTIES[prop]['label'],
        'scale': 'global',
        'r2_mean': global_probes[prop]['r2_ridge'],
        'r2_std': 0.0,
        'n_probes': 1,
        'n_samples_mean': global_probes[prop]['n_samples'],
    })

    # Regional
    if prop in regional_probes and regional_probes[prop]:
        reg_r2s = [v['r2'] for v in regional_probes[prop].values()]
        reg_ns = [v['n_samples'] for v in regional_probes[prop].values()]
        scale_rows.append({
            'property': prop,
            'label': TARGET_PROPERTIES[prop]['label'],
            'scale': 'regional',
            'r2_mean': np.mean(reg_r2s),
            'r2_std': np.std(reg_r2s),
            'n_probes': len(reg_r2s),
            'n_samples_mean': np.mean(reg_ns),
        })

    # Local
    if prop in local_probes and local_probes[prop]:
        loc_r2s = [v['r2'] for v in local_probes[prop].values()]
        scale_rows.append({
            'property': prop,
            'label': TARGET_PROPERTIES[prop]['label'],
            'scale': 'local',
            'r2_mean': np.median(loc_r2s),  # median for robustness
            'r2_std': np.std(loc_r2s),
            'n_probes': len(loc_r2s),
            'n_samples_mean': K_LOCAL,
        })

scale_df = pd.DataFrame(scale_rows)
scale_df.to_csv(f'{OUTPUT_DIR}/probe_scale_comparison.csv', index=False)

for prop in TARGET_PROPERTIES:
    sub = scale_df[scale_df['property'] == prop]
    if len(sub) == 0:
        continue
    print(f"\n  {TARGET_PROPERTIES[prop]['label']}:")
    for _, row in sub.iterrows():
        print(f"    {row['scale']:10s}: R2={row['r2_mean']:.4f} +/- {row['r2_std']:.4f} "
              f"(n_probes={int(row['n_probes'])}, ~{int(row['n_samples_mean'])} samples/probe)")


# ═══════════════════════════════════════════════════════════════════════════
# 8. DIRECTION STABILITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("DIRECTION STABILITY ANALYSIS")
print("=" * 70)

stability_rows = []

for prop in TARGET_PROPERTIES:
    if prop not in local_probes:
        continue

    # Local-to-global cosine similarities
    cos_global = [v['cosine_with_global'] for v in local_probes[prop].values()
                  if 'cosine_with_global' in v]
    cos_regional = [v['cosine_with_regional'] for v in local_probes[prop].values()
                    if np.isfinite(v.get('cosine_with_regional', np.nan))]

    # Regional-to-global cosine similarities
    reg_cos_global = [v['cosine_with_global'] for v in regional_probes.get(prop, {}).values()]

    # Inter-regional cosine similarities (all pairs)
    reg_dirs = {rn: v['direction'] for rn, v in regional_probes.get(prop, {}).items()}
    inter_regional_cos = []
    rnames = list(reg_dirs.keys())
    for i in range(len(rnames)):
        for j in range(i+1, len(rnames)):
            cos = float(np.dot(reg_dirs[rnames[i]], reg_dirs[rnames[j]]))
            inter_regional_cos.append(cos)

    stability_rows.append({
        'property': prop,
        'label': TARGET_PROPERTIES[prop]['label'],
        'cos_local_global_mean': np.mean(np.abs(cos_global)),
        'cos_local_global_std': np.std(np.abs(cos_global)),
        'cos_local_regional_mean': np.mean(np.abs(cos_regional)) if cos_regional else np.nan,
        'cos_local_regional_std': np.std(np.abs(cos_regional)) if cos_regional else np.nan,
        'cos_regional_global_mean': np.mean(np.abs(reg_cos_global)) if reg_cos_global else np.nan,
        'cos_inter_regional_mean': np.mean(np.abs(inter_regional_cos)) if inter_regional_cos else np.nan,
        'cos_inter_regional_min': np.min(np.abs(inter_regional_cos)) if inter_regional_cos else np.nan,
    })

    print(f"\n  {TARGET_PROPERTIES[prop]['label']}:")
    print(f"    |cos(local, global)|:   mean={np.mean(np.abs(cos_global)):.4f} +/- {np.std(np.abs(cos_global)):.4f}")
    if cos_regional:
        print(f"    |cos(local, regional)|: mean={np.mean(np.abs(cos_regional)):.4f} +/- {np.std(np.abs(cos_regional)):.4f}")
    if reg_cos_global:
        print(f"    |cos(regional, global)|: {[f'{abs(c):.3f}' for c in reg_cos_global]}")
    if inter_regional_cos:
        print(f"    |cos(region_i, region_j)|: mean={np.mean(np.abs(inter_regional_cos)):.4f}, "
              f"min={np.min(np.abs(inter_regional_cos)):.4f}")

stability_df = pd.DataFrame(stability_rows)
stability_df.to_csv(f'{OUTPUT_DIR}/direction_stability.csv', index=False)


# ═══════════════════════════════════════════════════════════════════════════
# 9. COMPARE PCA VS PROBE DIRECTIONS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PCA vs PROBE DIRECTION COMPARISON")
print("=" * 70)

pca_vs_probe = []

for prop in TARGET_PROPERTIES:
    if prop not in global_probes:
        continue

    probe_dir = global_probes[prop]['direction']

    # Global PCA: find best-correlated global PC
    if global_evecs is not None:
        rng_pca = np.random.default_rng(SEED + 2)
        sample_idx = rng_pca.choice(len(E), size=min(50000, len(E)), replace=False)
        y_sample = df_full[prop].values[sample_idx]
        valid = np.isfinite(y_sample)

        best_corr = 0
        best_pc = -1
        for pc_i in range(min(10, global_evecs.shape[1])):
            proj = E[sample_idx] @ global_evecs[:, pc_i]
            r = np.corrcoef(proj[valid], y_sample[valid])[0, 1]
            if abs(r) > abs(best_corr):
                best_corr = r
                best_pc = pc_i

        pca_direction = global_evecs[:, best_pc]
        if best_corr < 0:
            pca_direction = -pca_direction

        cos_pca_probe = float(np.dot(pca_direction, probe_dir))
        print(f"  {TARGET_PROPERTIES[prop]['label']:15s}: "
              f"PCA PC{best_pc} (r={best_corr:.3f}) vs Probe: cos={cos_pca_probe:.4f}")

        pca_vs_probe.append({
            'property': prop,
            'pca_pc_index': best_pc,
            'pca_correlation': best_corr,
            'probe_r2': global_probes[prop]['r2_ridge'],
            'cosine_pca_probe': cos_pca_probe,
        })

pca_vs_probe_df = pd.DataFrame(pca_vs_probe)
pca_vs_probe_df.to_csv(f'{OUTPUT_DIR}/pca_vs_probe_directions.csv', index=False)


# ═══════════════════════════════════════════════════════════════════════════
# 10. TARGETED SHIFT EXPERIMENT WITH PROBE-DERIVED DIRECTIONS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("EXPERIMENT 2: TARGETED SHIFTS WITH PROBE DIRECTIONS")
print("=" * 70)


def shift_and_retrieve(embedding, direction, sigma, n_sigma, k=K_RETRIEVAL):
    """Shift an embedding along a direction, L2-normalize, find nearest neighbors."""
    shifted = embedding + direction * sigma * n_sigma
    norm = np.linalg.norm(shifted)
    if norm > 0:
        shifted = shifted / norm * np.linalg.norm(embedding)
    dists, idx = nn_index.kneighbors(shifted.reshape(1, -1), n_neighbors=k + 1)
    return shifted, idx[0], dists[0]


def compute_metrics(source_idx, retrieved_indices, target_col):
    """Compute target change and non-target preservation for a shift."""
    source_env = {c: df_full[c].iloc[source_idx] for c in all_env if c in df_full.columns}
    results = []
    for ret_idx in retrieved_indices:
        if ret_idx >= len(df_full):
            continue
        ret_env = {c: df_full[c].iloc[ret_idx] for c in all_env if c in df_full.columns}

        # Target accuracy
        src_val = source_env.get(target_col, np.nan)
        ret_val = ret_env.get(target_col, np.nan)
        target_sigma = env_stats.get(target_col, {}).get('std', 1)

        if np.isfinite(src_val) and np.isfinite(ret_val) and target_sigma > 0:
            target_change = (ret_val - src_val) / target_sigma
        else:
            target_change = np.nan

        # Non-target preservation
        nontarget_devs = []
        for col in all_env:
            if col == target_col or col not in df_full.columns:
                continue
            s_val = source_env.get(col, np.nan)
            r_val = ret_env.get(col, np.nan)
            col_sigma = env_stats.get(col, {}).get('std', 1)
            if np.isfinite(s_val) and np.isfinite(r_val) and col_sigma > 0:
                nontarget_devs.append(abs(r_val - s_val) / col_sigma)

        mean_nontarget = float(np.mean(nontarget_devs)) if nontarget_devs else np.nan

        # On-manifold distance
        on_manifold_dist = float(np.linalg.norm(E[source_idx] - E[ret_idx]))

        results.append({
            'target_change_sigma': target_change,
            'mean_nontarget_dev': mean_nontarget,
            'on_manifold_dist': on_manifold_dist,
        })
    return results


# Also compute PCA-based local direction for direct comparison
def find_local_pca_direction(idx, target_col, k=K_LOCAL):
    """Find the local PCA direction (single best PC). Matches phase2_arithmetic.py exactly."""
    point = E[idx].reshape(1, -1)
    dists, nbr_idx = nn_index.kneighbors(point, n_neighbors=k + 1)
    nbr_idx = nbr_idx[0, 1:]
    E_local = E[nbr_idx]
    pca = PCA(n_components=min(k - 1, N_DIMS))
    pca.fit(E_local)

    target_vals = df_full[target_col].values[nbr_idx]
    valid = np.isfinite(target_vals)
    if valid.sum() < 10:
        return None, None

    pc_scores = pca.transform(E_local)
    n_comp = min(10, pc_scores.shape[1])
    correlations = np.zeros(n_comp)
    for c in range(n_comp):
        mask = valid
        if mask.sum() > 5:
            r = np.corrcoef(pc_scores[mask, c], target_vals[mask])[0, 1]
            correlations[c] = r if np.isfinite(r) else 0

    best_pc = np.argmax(np.abs(correlations))
    best_corr = correlations[best_pc]
    direction = pca.components_[best_pc]
    if best_corr < 0:
        direction = -direction

    projections = E_local @ direction
    sigma = projections.std()
    return direction, sigma


# --- Run shifts ---
# Methods: probe_global, probe_regional, probe_local, pca_local, random
all_shift_results = []
t0 = time.time()

for i, src_idx in enumerate(source_indices):
    if i % 100 == 0 and i > 0:
        elapsed = time.time() - t0
        print(f"  {i}/{len(source_indices)} ({elapsed:.1f}s)")

    src_region = region_labels[src_idx]

    for prop in TARGET_PROPERTIES:
        if prop not in df_full.columns:
            continue
        src_val = df_full[prop].iloc[src_idx]
        if not np.isfinite(src_val):
            continue

        for n_sigma in SHIFT_SIGMAS:
            base_row = {
                'property': prop,
                'n_sigma': n_sigma,
                'source_idx': int(src_idx),
                'source_val': float(src_val),
                'source_region': src_region,
            }

            # ── Method 1: Probe Global ──
            if prop in global_probes:
                gp = global_probes[prop]
                shifted, ret_idx, ret_dists = shift_and_retrieve(
                    E[src_idx], gp['direction'], gp['sigma'], n_sigma)
                metrics = compute_metrics(src_idx, ret_idx[:K_RETRIEVAL], prop)
                for m in metrics:
                    m.update(base_row)
                    m['method'] = 'probe_global'
                all_shift_results.extend(metrics)

            # ── Method 2: Probe Regional ──
            if (src_region != '' and prop in regional_probes
                    and src_region in regional_probes.get(prop, {})):
                rp = regional_probes[prop][src_region]
                shifted, ret_idx, ret_dists = shift_and_retrieve(
                    E[src_idx], rp['direction'], rp['sigma'], n_sigma)
                metrics = compute_metrics(src_idx, ret_idx[:K_RETRIEVAL], prop)
                for m in metrics:
                    m.update(base_row)
                    m['method'] = 'probe_regional'
                all_shift_results.extend(metrics)

            # ── Method 3: Probe Local ──
            if prop in local_probes and int(src_idx) in local_probes[prop]:
                lp = local_probes[prop][int(src_idx)]
                shifted, ret_idx, ret_dists = shift_and_retrieve(
                    E[src_idx], lp['direction'], lp['sigma'], n_sigma)
                metrics = compute_metrics(src_idx, ret_idx[:K_RETRIEVAL], prop)
                for m in metrics:
                    m.update(base_row)
                    m['method'] = 'probe_local'
                all_shift_results.extend(metrics)

            # ── Method 4: PCA Local (baseline from phase2_arithmetic) ──
            pca_dir, pca_sig = find_local_pca_direction(src_idx, prop)
            if pca_dir is not None and pca_sig > 0:
                shifted, ret_idx, ret_dists = shift_and_retrieve(
                    E[src_idx], pca_dir, pca_sig, n_sigma)
                metrics = compute_metrics(src_idx, ret_idx[:K_RETRIEVAL], prop)
                for m in metrics:
                    m.update(base_row)
                    m['method'] = 'pca_local'
                all_shift_results.extend(metrics)

            # ── Method 5: Random Direction ──
            rng_rand = np.random.default_rng(SEED + i * 100 + hash(prop) % 1000)
            rand_dir = rng_rand.standard_normal(N_DIMS)
            rand_dir = rand_dir / np.linalg.norm(rand_dir)
            rand_sig = np.std(E @ rand_dir)
            shifted, ret_idx, ret_dists = shift_and_retrieve(
                E[src_idx], rand_dir, rand_sig, n_sigma)
            metrics = compute_metrics(src_idx, ret_idx[:K_RETRIEVAL], prop)
            for m in metrics:
                m.update(base_row)
                m['method'] = 'random'
            all_shift_results.extend(metrics)

elapsed = time.time() - t0
print(f"\n  Completed in {elapsed:.1f}s")
print(f"  Total result rows: {len(all_shift_results):,}")

shift_df = pd.DataFrame(all_shift_results)
shift_df.to_csv(f'{OUTPUT_DIR}/probe_shift_results.csv', index=False)
print("  Saved probe_shift_results.csv")


# ═══════════════════════════════════════════════════════════════════════════
# 11. AGGREGATE SHIFT RESULTS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SHIFT EXPERIMENT RESULTS")
print("=" * 70)


def aggregate_shifts(df):
    agg = df.groupby(['method', 'property', 'n_sigma']).agg(
        target_change_mean=('target_change_sigma', 'mean'),
        target_change_std=('target_change_sigma', 'std'),
        nontarget_mean=('mean_nontarget_dev', 'mean'),
        nontarget_std=('mean_nontarget_dev', 'std'),
        on_manifold_mean=('on_manifold_dist', 'mean'),
        count=('target_change_sigma', 'count'),
    ).reset_index()

    # Precision: target > 0.3sigma AND nontarget < 1sigma
    def precision_fn(group):
        valid = group.dropna(subset=['target_change_sigma', 'mean_nontarget_dev'])
        if len(valid) == 0:
            return 0.0
        hits = ((valid['target_change_sigma'] > 0.3) &
                (valid['mean_nontarget_dev'] < 1.0)).sum()
        return hits / len(valid)

    prec = df.groupby(['method', 'property', 'n_sigma']).apply(precision_fn)
    prec = prec.reset_index()
    prec.columns = ['method', 'property', 'n_sigma', 'precision']
    agg = agg.merge(prec, on=['method', 'property', 'n_sigma'])
    return agg


agg_df = aggregate_shifts(shift_df)
agg_df.to_csv(f'{OUTPUT_DIR}/probe_shift_aggregate.csv', index=False)

# Print summary at 1sigma
METHOD_ORDER = ['probe_global', 'probe_regional', 'probe_local', 'pca_local', 'random']
METHOD_LABELS_SHORT = {
    'probe_global': 'Probe (global)',
    'probe_regional': 'Probe (regional)',
    'probe_local': 'Probe (local)',
    'pca_local': 'PCA (local)',
    'random': 'Random',
}

for prop in TARGET_PROPERTIES:
    sub = agg_df[(agg_df['property'] == prop) & (agg_df['n_sigma'] == 1.0)]
    if len(sub) == 0:
        continue
    print(f"\n  {TARGET_PROPERTIES[prop]['label']} (at 1sigma shift):")
    for method in METHOD_ORDER:
        row = sub[sub['method'] == method]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        print(f"    {METHOD_LABELS_SHORT.get(method, method):20s}: "
              f"Dtarget={row['target_change_mean']:+.3f}sigma, "
              f"Dother={row['nontarget_mean']:.3f}sigma, "
              f"prec={row['precision']:.3f}, "
              f"n={int(row['count'])}")


# ═══════════════════════════════════════════════════════════════════════════
# 12. KEY STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STATISTICAL TESTS: PROBE vs PCA")
print("=" * 70)

stat_rows = []

for prop in TARGET_PROPERTIES:
    sub_1sig = shift_df[(shift_df['property'] == prop) & (shift_df['n_sigma'] == 1.0)]

    probe_local = sub_1sig[sub_1sig['method'] == 'probe_local']['target_change_sigma'].dropna()
    pca_local = sub_1sig[sub_1sig['method'] == 'pca_local']['target_change_sigma'].dropna()
    probe_global = sub_1sig[sub_1sig['method'] == 'probe_global']['target_change_sigma'].dropna()
    random = sub_1sig[sub_1sig['method'] == 'random']['target_change_sigma'].dropna()

    if len(probe_local) > 10 and len(pca_local) > 10:
        # Probe local vs PCA local
        t_stat, p_val = stats.mannwhitneyu(probe_local, pca_local, alternative='two-sided')
        print(f"\n  {TARGET_PROPERTIES[prop]['label']}:")
        print(f"    Probe local (n={len(probe_local)}): mean={probe_local.mean():.4f}")
        print(f"    PCA local   (n={len(pca_local)}):   mean={pca_local.mean():.4f}")
        print(f"    Mann-Whitney U: stat={t_stat:.1f}, p={p_val:.4e}")
        print(f"    Effect size (mean diff): {probe_local.mean() - pca_local.mean():+.4f}sigma")

        # Probe local vs random
        if len(random) > 10:
            t2, p2 = stats.mannwhitneyu(probe_local, random, alternative='two-sided')
            print(f"    Probe local vs Random: p={p2:.4e}, diff={probe_local.mean() - random.mean():+.4f}sigma")

        # Probe global vs probe local
        if len(probe_global) > 10:
            t3, p3 = stats.mannwhitneyu(probe_local, probe_global, alternative='two-sided')
            print(f"    Probe local vs Probe global: p={p3:.4e}, diff={probe_local.mean() - probe_global.mean():+.4f}sigma")

        # Also compare nontarget deviations
        probe_nt = sub_1sig[sub_1sig['method'] == 'probe_local']['mean_nontarget_dev'].dropna()
        pca_nt = sub_1sig[sub_1sig['method'] == 'pca_local']['mean_nontarget_dev'].dropna()
        if len(probe_nt) > 10 and len(pca_nt) > 10:
            print(f"    Collateral: probe={probe_nt.mean():.4f}sigma, pca={pca_nt.mean():.4f}sigma")

        stat_rows.append({
            'property': prop,
            'probe_local_target_mean': probe_local.mean(),
            'pca_local_target_mean': pca_local.mean(),
            'probe_local_nontarget_mean': probe_nt.mean() if len(probe_nt) > 0 else np.nan,
            'pca_local_nontarget_mean': pca_nt.mean() if len(pca_nt) > 0 else np.nan,
            'p_value_target': p_val,
            'diff_target': probe_local.mean() - pca_local.mean(),
        })

stat_df = pd.DataFrame(stat_rows)
stat_df.to_csv(f'{OUTPUT_DIR}/probe_vs_pca_stats.csv', index=False)


# ═══════════════════════════════════════════════════════════════════════════
# 13. FIGURES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("GENERATING FIGURES")
print("=" * 70)

METHOD_COLORS = {
    'probe_global':   '#1565C0',   # dark blue
    'probe_regional': '#42A5F5',   # medium blue
    'probe_local':    '#E53935',   # red
    'pca_local':      '#4CAF50',   # green
    'random':         '#9E9E9E',   # grey
}

# --- Figure 1: R^2 across scales ---
print("  Generating Fig 1: R2 across scales...")
fig, axes = plt.subplots(1, len(TARGET_PROPERTIES), figsize=(4.5 * len(TARGET_PROPERTIES), 5),
                         sharey=True)
if len(TARGET_PROPERTIES) == 1:
    axes = [axes]

for i, prop in enumerate(TARGET_PROPERTIES):
    if prop not in local_probes:
        continue

    # Boxplot data
    global_r2 = global_probes[prop]['r2_ridge']
    regional_r2s = [v['r2'] for v in regional_probes.get(prop, {}).values()]
    local_r2s = [v['r2'] for v in local_probes[prop].values()]

    # Box for local, scatter for global/regional
    bp = axes[i].boxplot([local_r2s], positions=[3], widths=0.5,
                         patch_artist=True, showfliers=False,
                         medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set(facecolor='#E53935', alpha=0.4)

    axes[i].scatter([1], [global_r2], s=120, c='#1565C0', marker='D',
                    zorder=5, edgecolor='black', linewidth=0.5, label='Global')
    if regional_r2s:
        axes[i].scatter([2]*len(regional_r2s), regional_r2s, s=80, c='#42A5F5',
                        marker='s', zorder=5, edgecolor='black', linewidth=0.5, label='Regional')

    axes[i].set_xticks([1, 2, 3])
    axes[i].set_xticklabels(['Global\n(1 probe)', 'Regional\n(5 probes)',
                              f'Local\n({len(local_r2s)} probes)'], fontsize=10)
    axes[i].set_title(TARGET_PROPERTIES[prop]['label'])
    axes[i].grid(True, alpha=0.3)
    if i == 0:
        axes[i].set_ylabel('$R^2$ (Ridge Regression)')

plt.suptitle('Linear Probe Quality Across Spatial Scales', fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_probe_r2_by_scale.png', dpi=300, facecolor='white')
plt.close(fig)
print("    Saved fig_probe_r2_by_scale.png")


# --- Figure 2: Direction stability (cosine similarity distributions) ---
print("  Generating Fig 2: Direction stability...")
fig, axes = plt.subplots(1, len(TARGET_PROPERTIES), figsize=(4.5 * len(TARGET_PROPERTIES), 5),
                         sharey=True)
if len(TARGET_PROPERTIES) == 1:
    axes = [axes]

for i, prop in enumerate(TARGET_PROPERTIES):
    if prop not in local_probes:
        continue

    cos_global = [v['cosine_with_global'] for v in local_probes[prop].values()]
    cos_regional = [v['cosine_with_regional'] for v in local_probes[prop].values()
                    if np.isfinite(v.get('cosine_with_regional', np.nan))]

    axes[i].hist(np.abs(cos_global), bins=30, alpha=0.6, color='#1565C0',
                 label=f'|cos(local, global)|\n(med={np.median(np.abs(cos_global)):.2f})',
                 density=True)
    if cos_regional:
        axes[i].hist(np.abs(cos_regional), bins=30, alpha=0.6, color='#42A5F5',
                     label=f'|cos(local, regional)|\n(med={np.median(np.abs(cos_regional)):.2f})',
                     density=True)

    # Expected value for random unit vectors in 64D
    # E[|cos|] = sqrt(2/pi) * 1/sqrt(d) ~ 0.10 for d=64
    axes[i].axvline(0.10, color='grey', linestyle='--', alpha=0.7, label='Random baseline')

    axes[i].set_xlim(0, 1)
    axes[i].set_xlabel('|Cosine Similarity|')
    axes[i].set_title(TARGET_PROPERTIES[prop]['label'])
    axes[i].legend(fontsize=8, loc='upper right')
    axes[i].grid(True, alpha=0.3)
    if i == 0:
        axes[i].set_ylabel('Density')

plt.suptitle('Concept Direction Alignment: Local vs. Global/Regional Probes',
             fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_direction_stability.png', dpi=300, facecolor='white')
plt.close(fig)
print("    Saved fig_direction_stability.png")


# --- Figure 3: Shift experiment comparison (probe vs PCA vs random) ---
print("  Generating Fig 3: Shift comparison...")
fig, axes = plt.subplots(2, len(TARGET_PROPERTIES),
                         figsize=(4.5 * len(TARGET_PROPERTIES), 9),
                         sharey='row')
if len(TARGET_PROPERTIES) == 1:
    axes = axes.reshape(-1, 1)

for i, prop in enumerate(TARGET_PROPERTIES):
    sub = agg_df[(agg_df['property'] == prop) & (agg_df['n_sigma'] == 1.0)]
    if len(sub) == 0:
        continue

    methods = []
    target_vals = []
    nontarget_vals = []
    colors = []

    for method in METHOD_ORDER:
        row = sub[sub['method'] == method]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        methods.append(METHOD_LABELS_SHORT.get(method, method))
        target_vals.append(row['target_change_mean'])
        nontarget_vals.append(row['nontarget_mean'])
        colors.append(METHOD_COLORS.get(method, '#888888'))

    x = np.arange(len(methods))

    # Top row: target change
    axes[0, i].bar(x, target_vals, color=colors, edgecolor='black', linewidth=0.5)
    axes[0, i].set_title(TARGET_PROPERTIES[prop]['label'])
    axes[0, i].set_ylabel('Target Change (sigma)' if i == 0 else '')
    axes[0, i].set_xticks(x)
    axes[0, i].set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    axes[0, i].grid(True, alpha=0.3, axis='y')
    axes[0, i].axhline(0, color='black', linewidth=0.5)

    # Bottom row: nontarget deviation
    axes[1, i].bar(x, nontarget_vals, color=colors, edgecolor='black', linewidth=0.5)
    axes[1, i].set_ylabel('Non-target Deviation (sigma)' if i == 0 else '')
    axes[1, i].set_xticks(x)
    axes[1, i].set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    axes[1, i].grid(True, alpha=0.3, axis='y')

fig.suptitle('Targeted Shift: Probe vs. PCA vs. Random (1sigma shift)',
             fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_probe_shift_comparison.png', dpi=300, facecolor='white')
plt.close(fig)
print("    Saved fig_probe_shift_comparison.png")


# --- Figure 4: Precision by shift magnitude (all methods) ---
print("  Generating Fig 4: Precision curves...")
fig, axes = plt.subplots(1, len(TARGET_PROPERTIES),
                         figsize=(4.5 * len(TARGET_PROPERTIES), 5),
                         sharey=True)
if len(TARGET_PROPERTIES) == 1:
    axes = [axes]

for i, prop in enumerate(TARGET_PROPERTIES):
    sub = agg_df[agg_df['property'] == prop]
    if len(sub) == 0:
        continue

    for method in METHOD_ORDER:
        msub = sub[sub['method'] == method].sort_values('n_sigma')
        if len(msub) == 0:
            continue
        axes[i].plot(msub['n_sigma'], msub['precision'],
                     marker='o', color=METHOD_COLORS.get(method, '#888'),
                     label=METHOD_LABELS_SHORT.get(method, method), linewidth=2)

    axes[i].set_xlabel('Shift Magnitude (sigma)')
    axes[i].set_title(TARGET_PROPERTIES[prop]['label'])
    axes[i].grid(True, alpha=0.3)
    if i == 0:
        axes[i].set_ylabel('Precision')
    if i == len(TARGET_PROPERTIES) - 1:
        axes[i].legend(fontsize=7, loc='best')

plt.suptitle('Shift Precision vs. Magnitude by Method',
             fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_precision_curves.png', dpi=300, facecolor='white')
plt.close(fig)
print("    Saved fig_precision_curves.png")


# --- Figure 5: Combined panel for paper ---
print("  Generating Fig 5: Paper-ready combined panel...")
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

# Panel A: R2 by scale (first property as representative)
ax_a = fig.add_subplot(gs[0, 0])
prop0 = list(TARGET_PROPERTIES.keys())[0]
if prop0 in local_probes:
    global_r2 = global_probes[prop0]['r2_ridge']
    regional_r2s = [v['r2'] for v in regional_probes.get(prop0, {}).values()]
    local_r2s = [v['r2'] for v in local_probes[prop0].values()]

    bp = ax_a.boxplot([regional_r2s, local_r2s], positions=[2, 3], widths=0.5,
                      patch_artist=True, showfliers=False,
                      medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set(facecolor='#42A5F5', alpha=0.4)
    bp['boxes'][1].set(facecolor='#E53935', alpha=0.4)
    ax_a.scatter([1], [global_r2], s=120, c='#1565C0', marker='D',
                 zorder=5, edgecolor='black', linewidth=0.5)
    ax_a.set_xticks([1, 2, 3])
    ax_a.set_xticklabels(['Global', 'Regional', 'Local'], fontsize=10)
    ax_a.set_ylabel('$R^2$')
    ax_a.set_title(f'(a) Probe Quality: {TARGET_PROPERTIES[prop0]["label"]}')
    ax_a.grid(True, alpha=0.3)

# Panel B: Direction stability (first property)
ax_b = fig.add_subplot(gs[0, 1])
if prop0 in local_probes:
    cos_g = [abs(v['cosine_with_global']) for v in local_probes[prop0].values()]
    ax_b.hist(cos_g, bins=30, alpha=0.7, color='#1565C0', density=True, edgecolor='white')
    ax_b.axvline(np.median(cos_g), color='#E53935', linewidth=2, linestyle='-',
                 label=f'Median={np.median(cos_g):.2f}')
    ax_b.axvline(0.10, color='grey', linewidth=1.5, linestyle='--', label='Random baseline')
    ax_b.set_xlabel('|cos(local, global)|')
    ax_b.set_ylabel('Density')
    ax_b.set_title(f'(b) Direction Rotation: {TARGET_PROPERTIES[prop0]["label"]}')
    ax_b.legend(fontsize=9)
    ax_b.grid(True, alpha=0.3)

# Panel C: PCA vs Probe cosine similarity (all properties)
ax_c = fig.add_subplot(gs[0, 2])
if pca_vs_probe_df is not None and len(pca_vs_probe_df) > 0:
    props_plot = pca_vs_probe_df['property'].values
    cos_vals = pca_vs_probe_df['cosine_pca_probe'].values
    y_pos = np.arange(len(props_plot))
    colors_c = ['#E53935' if abs(c) < 0.5 else '#4CAF50' if abs(c) > 0.8 else '#FF9800'
                for c in cos_vals]
    ax_c.barh(y_pos, np.abs(cos_vals), color=colors_c, edgecolor='black', linewidth=0.5)
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels([TARGET_PROPERTIES.get(p, {}).get('label', p) for p in props_plot],
                          fontsize=10)
    ax_c.set_xlabel('|cos(PCA direction, Probe direction)|')
    ax_c.set_title('(c) PCA vs. Probe Direction Agreement')
    ax_c.set_xlim(0, 1)
    ax_c.grid(True, alpha=0.3, axis='x')

# Panel D-E: Shift comparison (target change and nontarget, all properties aggregated)
ax_d = fig.add_subplot(gs[1, 0:2])

# Aggregate across all properties at 1sigma
sub_1sig = agg_df[agg_df['n_sigma'] == 1.0]
method_agg = sub_1sig.groupby('method').agg(
    target_mean=('target_change_mean', 'mean'),
    nontarget_mean=('nontarget_mean', 'mean'),
    precision_mean=('precision', 'mean'),
).reindex(METHOD_ORDER)

x = np.arange(len(METHOD_ORDER))
width = 0.35
bars1 = ax_d.bar(x - width/2, method_agg['target_mean'], width,
                  label='Target Change (sigma)', edgecolor='black', linewidth=0.5,
                  color=[METHOD_COLORS[m] for m in METHOD_ORDER], alpha=0.8)
bars2 = ax_d.bar(x + width/2, method_agg['nontarget_mean'], width,
                  label='Non-target Dev. (sigma)', edgecolor='black', linewidth=0.5,
                  color=[METHOD_COLORS[m] for m in METHOD_ORDER], alpha=0.4,
                  hatch='//')
ax_d.set_xticks(x)
ax_d.set_xticklabels([METHOD_LABELS_SHORT[m] for m in METHOD_ORDER],
                      rotation=30, ha='right', fontsize=9)
ax_d.set_ylabel('sigma')
ax_d.set_title('(d) Shift Quality: All Properties at 1sigma')
ax_d.legend(fontsize=9)
ax_d.grid(True, alpha=0.3, axis='y')
ax_d.axhline(0, color='black', linewidth=0.5)

# Panel F: Precision comparison
ax_f = fig.add_subplot(gs[1, 2])
for method in METHOD_ORDER:
    msub = agg_df[agg_df['method'] == method].groupby('n_sigma')['precision'].mean()
    if len(msub) == 0:
        continue
    ax_f.plot(msub.index, msub.values, marker='o',
              color=METHOD_COLORS[method],
              label=METHOD_LABELS_SHORT[method], linewidth=2)
ax_f.set_xlabel('Shift Magnitude (sigma)')
ax_f.set_ylabel('Precision')
ax_f.set_title('(e) Precision vs. Shift Magnitude')
ax_f.legend(fontsize=7, loc='best')
ax_f.grid(True, alpha=0.3)

plt.suptitle('Linear Probes for Concept Directions in AlphaEarth Embeddings',
             fontweight='bold', fontsize=16, y=1.01)
fig.savefig(f'{FIG_DIR}/fig_linear_probes_combined.png', dpi=300, facecolor='white')
plt.close(fig)
print("    Saved fig_linear_probes_combined.png")


# ═══════════════════════════════════════════════════════════════════════════
# 14. SAVE COMPREHENSIVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

results_dict = {
    'experiment': 'Phase 2A: Linear Probes for Concept Directions',
    'config': {
        'subsample_total': SUBSAMPLE_TOTAL,
        'n_source_locations': N_SOURCE_LOCATIONS,
        'k_local': K_LOCAL,
        'ridge_alpha': RIDGE_ALPHA,
        'lasso_alpha': LASSO_ALPHA,
        'shift_sigmas': SHIFT_SIGMAS,
        'seed': SEED,
        'n_regions': len(REGIONS),
        'regions': list(REGIONS.keys()),
    },
    'global_probes': {},
    'regional_summary': {},
    'local_summary': {},
    'pca_vs_probe': {},
    'shift_summary': {},
}

for prop in TARGET_PROPERTIES:
    if prop not in global_probes:
        continue

    # Global probe info (without the direction vector itself, for readability)
    gp = global_probes[prop]
    results_dict['global_probes'][prop] = {
        'r2_ridge': gp['r2_ridge'],
        'r2_lasso': gp['r2_lasso'],
        'cosine_ridge_lasso': gp['cosine_ridge_lasso'],
        'n_samples': gp['n_samples'],
        'sigma': gp['sigma'],
        'top_5_dims': np.argsort(np.abs(gp['direction']))[-5:][::-1].tolist(),
        'top_5_weights': gp['direction'][np.argsort(np.abs(gp['direction']))[-5:][::-1]].tolist(),
    }

    # Regional summary
    if prop in regional_probes:
        results_dict['regional_summary'][prop] = {}
        for rname, rp in regional_probes[prop].items():
            results_dict['regional_summary'][prop][rname] = {
                'r2': rp['r2'],
                'n_samples': rp['n_samples'],
                'cosine_with_global': rp['cosine_with_global'],
            }

    # Local summary
    if prop in local_probes:
        loc_r2s = [v['r2'] for v in local_probes[prop].values()]
        cos_g = [v['cosine_with_global'] for v in local_probes[prop].values()]
        results_dict['local_summary'][prop] = {
            'n_probes': len(loc_r2s),
            'r2_median': float(np.median(loc_r2s)),
            'r2_q25': float(np.percentile(loc_r2s, 25)),
            'r2_q75': float(np.percentile(loc_r2s, 75)),
            'r2_mean': float(np.mean(loc_r2s)),
            'cos_global_median': float(np.median(np.abs(cos_g))),
            'cos_global_mean': float(np.mean(np.abs(cos_g))),
        }

    # Shift summary at 1sigma
    sub_1sig = agg_df[(agg_df['property'] == prop) & (agg_df['n_sigma'] == 1.0)]
    results_dict['shift_summary'][prop] = {}
    for _, row in sub_1sig.iterrows():
        results_dict['shift_summary'][prop][row['method']] = {
            'target_change_mean': float(row['target_change_mean']),
            'nontarget_mean': float(row['nontarget_mean']),
            'precision': float(row['precision']),
            'count': int(row['count']),
        }

# PCA vs probe
for _, row in pca_vs_probe_df.iterrows():
    results_dict['pca_vs_probe'][row['property']] = {
        'pca_pc_index': int(row['pca_pc_index']),
        'pca_correlation': float(row['pca_correlation']),
        'probe_r2': float(row['probe_r2']),
        'cosine_pca_probe': float(row['cosine_pca_probe']),
    }

with open(f'{OUTPUT_DIR}/linear_probe_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2, default=str)
print("  Saved linear_probe_results.json")

# Save global probe direction vectors for use by Phase 3 agent
probe_directions = {}
for prop in global_probes:
    probe_directions[prop] = {
        'global_direction': global_probes[prop]['direction'].tolist(),
        'global_sigma': float(global_probes[prop]['sigma']),
        'global_r2': float(global_probes[prop]['r2_ridge']),
    }
    if prop in regional_probes:
        probe_directions[prop]['regional'] = {}
        for rname, rp in regional_probes[prop].items():
            probe_directions[prop]['regional'][rname] = {
                'direction': rp['direction'].tolist(),
                'sigma': float(rp['sigma']),
                'r2': float(rp['r2']),
            }

with open(f'{OUTPUT_DIR}/probe_concept_vectors.json', 'w') as f:
    json.dump(probe_directions, f, indent=2)
print("  Saved probe_concept_vectors.json")


# ═══════════════════════════════════════════════════════════════════════════
# 15. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 2A SUMMARY: LINEAR PROBES")
print("=" * 70)

print(f"\n  DATA: {len(E):,} vectors, {len(source_indices)} source locations, "
      f"{len(REGIONS)} regions")

print(f"\n  PROBE QUALITY (R2):")
for prop in TARGET_PROPERTIES:
    if prop not in global_probes:
        continue
    g_r2 = global_probes[prop]['r2_ridge']
    reg_r2s = [v['r2'] for v in regional_probes.get(prop, {}).values()]
    loc_r2s = [v['r2'] for v in local_probes.get(prop, {}).values()]
    print(f"    {TARGET_PROPERTIES[prop]['label']:15s}: "
          f"global={g_r2:.4f}, regional={np.mean(reg_r2s):.4f}+/-{np.std(reg_r2s):.4f}, "
          f"local(med)={np.median(loc_r2s):.4f}")

print(f"\n  DIRECTION STABILITY:")
for prop in TARGET_PROPERTIES:
    if prop not in local_probes:
        continue
    cos_g = [abs(v['cosine_with_global']) for v in local_probes[prop].values()]
    print(f"    {TARGET_PROPERTIES[prop]['label']:15s}: "
          f"|cos(local,global)| median={np.median(cos_g):.4f}, mean={np.mean(cos_g):.4f}")

print(f"\n  SHIFT RESULTS (at 1sigma, averaged across properties):")
sub_1 = agg_df[agg_df['n_sigma'] == 1.0]
for method in METHOD_ORDER:
    msub = sub_1[sub_1['method'] == method]
    if len(msub) == 0:
        continue
    print(f"    {METHOD_LABELS_SHORT.get(method, method):20s}: "
          f"Dtarget={msub['target_change_mean'].mean():+.4f}sigma, "
          f"Dother={msub['nontarget_mean'].mean():.4f}sigma, "
          f"prec={msub['precision'].mean():.4f}")

print(f"\n  OUTPUTS: {OUTPUT_DIR}/")
for f_name in sorted(os.listdir(OUTPUT_DIR)):
    fp = os.path.join(OUTPUT_DIR, f_name)
    if os.path.isfile(fp):
        print(f"    {f_name} ({os.path.getsize(fp)/1024:.1f} KB)")

print("\n" + "=" * 70)
print("Phase 2A complete.")
print("  Key question answered: Do supervised concept directions rescue")
print("  compositional arithmetic, or does manifold curvature dominate?")
print("=" * 70)

del E, nn_index
gc.collect()
print("Done.")
