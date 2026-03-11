"""
Compositional Arithmetic over AlphaEarth Embeddings

Tests compositional operations in the embedding space:
  A. Targeted Shift — shift along local vs. global directions for a target property
  B. Property Transfer — transplant one property from location B to location A
  C. Analogy — "A is to B as C is to ?" via vector arithmetic + FAISS retrieval

Compares four strategies: global direction, local PCA direction, random, geographic baseline.
Evaluates with on-manifold distance, target accuracy, non-target preservation, precision@k.

Input:  Yearly parquet files + manifold_results/ from characterization steps
Output: manifold_results/ (arithmetic results, analogy results, figures)

Author: Mashrekur Rahman | 2026
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
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
OUTPUT_DIR = 'manifold_results/arithmetic'
FIG_DIR = f'{OUTPUT_DIR}/figures'

YEARS = list(range(2017, 2024))
SUBSAMPLE_TOTAL = 500_000        # larger pool for realistic retrieval
N_SOURCE_LOCATIONS = 500         # test at 500 source locations
K_LOCAL_PCA = 100                # neighbors for local PCA
K_RETRIEVAL = 10                 # retrieve 10 analogs per shift
SHIFT_SIGMAS = [0.5, 1.0, 1.5, 2.0]  # shift magnitudes in local σ
SEED = 42

AE_COLS = [f'A{i:02d}' for i in range(64)]
N_DIMS = 64
CONUS_EXTENT = [-125.0, -66.5, 24.5, 49.5]

# Target properties for arithmetic experiments
TARGET_PROPERTIES = {
    'precip_annual_mm': {'label': 'Precipitation', 'unit': 'mm/yr'},
    'temp_mean_c':      {'label': 'Temperature',   'unit': '°C'},
    'evi_mean':         {'label': 'EVI',            'unit': ''},
    'elevation':        {'label': 'Elevation',      'unit': 'm'},
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
print("COMPOSITIONAL ARITHMETIC")
print("-" * 60)


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════

def load_data():
    per_year = SUBSAMPLE_TOTAL // len(YEARS)
    rng = np.random.default_rng(SEED)
    
    fp0 = f'{DATA_DIR}/conus_{YEARS[0]}_unified.parquet'
    all_cols = pq.read_schema(fp0).names
    
    env_cols = [c for c in TARGET_PROPERTIES.keys() if c in all_cols]
    extra_env = [c for c in ['ndvi_mean', 'lst_day_c', 'tree_cover_2000',
                             'soil_moisture', 'et_annual_mm', 'slope',
                             'soil_ph', 'impervious_pct'] if c in all_cols]
    load_cols = list(set(['longitude', 'latitude'] + AE_COLS + env_cols + extra_env))
    
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
    
    env_stats = {}
    for col in env_cols + extra_env:
        if col in df.columns:
            vals = df[col].dropna()
            env_stats[col] = {'mean': float(vals.mean()), 'std': float(vals.std()),
                              'min': float(vals.min()), 'max': float(vals.max())}
    
    print(f"  Combined: {len(df):,}")
    print(f"  Env columns: {env_cols}")
    print(f"  Extra env: {extra_env}")
    
    return E, coords, df, env_cols, extra_env, env_stats

E, coords, df_full, env_cols, extra_env, env_stats = load_data()
all_env = env_cols + [c for c in extra_env if c not in env_cols]


# ═══════════════════════════════════════════════════════════════════════════
# 2. BUILD K-NN INDEX
# ═══════════════════════════════════════════════════════════════════════════

print("\nBuilding k-NN index...")
t0 = time.time()
nn_index = NearestNeighbors(n_neighbors=K_LOCAL_PCA + K_RETRIEVAL + 1,
                             algorithm='auto', metric='euclidean')
nn_index.fit(E)
print(f"  Fitted in {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════
# 3. LOAD GLOBAL EIGENVECTORS (for global baseline)
# ═══════════════════════════════════════════════════════════════════════════

print("\nLoading manifold characterization outputs...")
global_evecs = None
evec_path = f'{RESULTS_DIR}/eigenvectors.csv'
if os.path.exists(evec_path):
    global_evecs = pd.read_csv(evec_path, index_col=0).values
    print(f"  Global eigenvectors: {global_evecs.shape}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. SELECT SOURCE LOCATIONS (stratified by complexity)
# ═══════════════════════════════════════════════════════════════════════════

def select_sources(n=N_SOURCE_LOCATIONS):
    """Select source locations stratified by elevation for diversity."""
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
            # Filter: require all target properties to be non-null
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
# 5. CORE ARITHMETIC FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_local_pca(idx, k=K_LOCAL_PCA):
    """Compute local PCA at a given index."""
    point = E[idx].reshape(1, -1)
    dists, nbr_idx = nn_index.kneighbors(point, n_neighbors=k + 1)
    nbr_idx = nbr_idx[0, 1:]
    E_local = E[nbr_idx]
    pca = PCA(n_components=min(k - 1, N_DIMS))
    pca.fit(E_local)
    return pca, nbr_idx


def find_local_property_direction(idx, target_col, k=K_LOCAL_PCA):
    """
    Find the local direction in embedding space most correlated with
    the target environmental property at a given location.
    
    Method: Compute local PCA, then correlate each local PC with the
    target variable across the k neighbors. The direction with highest
    |correlation| is the local property direction.
    """
    pca, nbr_idx = get_local_pca(idx, k)
    
    # Get target variable values for neighbors
    if target_col not in df_full.columns:
        return None, None, None, None
    
    target_vals = df_full[target_col].values[nbr_idx]
    valid = np.isfinite(target_vals)
    if valid.sum() < 10:
        return None, None, None, None
    
    # Project neighbors into local PC space
    E_local = E[nbr_idx]
    pc_scores = pca.transform(E_local)  # (k, n_components)
    
    # Correlate each PC with target variable
    n_comp = min(10, pc_scores.shape[1])  # only check top 10 PCs
    correlations = np.zeros(n_comp)
    for c in range(n_comp):
        mask = valid
        if mask.sum() > 5:
            r = np.corrcoef(pc_scores[mask, c], target_vals[mask])[0, 1]
            correlations[c] = r if np.isfinite(r) else 0
    
    # Best correlated PC
    best_pc = np.argmax(np.abs(correlations))
    best_corr = correlations[best_pc]
    
    # Convert back to 64-D direction
    # Local PC direction in original space = pca.components_[best_pc]
    local_direction = pca.components_[best_pc]  # shape (64,)
    
    # Sign: ensure positive correlation means shift in positive direction
    if best_corr < 0:
        local_direction = -local_direction
        best_corr = -best_corr
    
    # Local σ along this direction
    projections = E_local @ local_direction
    local_sigma = projections.std()
    
    return local_direction, local_sigma, best_corr, best_pc


def find_global_property_direction(target_col):
    """
    Find the global direction most correlated with the target property.
    Uses global PCA from the covariance analysis.
    """
    if global_evecs is None:
        return None, None, None
    
    # Sample to compute correlation
    rng = np.random.default_rng(SEED + 2)
    sample_idx = rng.choice(len(E), size=min(50000, len(E)), replace=False)
    
    target_vals = df_full[target_col].values[sample_idx]
    valid = np.isfinite(target_vals)
    if valid.sum() < 100:
        return None, None, None
    
    E_sample = E[sample_idx]
    
    # Project onto global PCs
    E_centered = E_sample - E_sample.mean(axis=0)
    pc_scores = E_centered @ global_evecs  # (n, 64)
    
    # Correlate each global PC with target
    n_check = min(20, pc_scores.shape[1])
    correlations = np.zeros(n_check)
    for c in range(n_check):
        r = np.corrcoef(pc_scores[valid, c], target_vals[valid])[0, 1]
        correlations[c] = r if np.isfinite(r) else 0
    
    best_pc = np.argmax(np.abs(correlations))
    best_corr = correlations[best_pc]
    
    direction = global_evecs[:, best_pc]
    if best_corr < 0:
        direction = -direction
        best_corr = -best_corr
    
    # Global σ
    projections = E_sample @ direction
    global_sigma = projections.std()
    
    return direction, global_sigma, best_corr


def shift_and_retrieve(embedding, direction, sigma, n_sigma, k=K_RETRIEVAL):
    """
    Shift an embedding along a direction, L2-normalize, find nearest neighbors.
    
    Returns: shifted vector, retrieved neighbor indices, distances
    """
    shifted = embedding + direction * sigma * n_sigma
    
    # L2 normalize (AlphaEarth embeddings are approximately unit-normalized)
    norm = np.linalg.norm(shifted)
    if norm > 0:
        shifted = shifted / norm * np.linalg.norm(embedding)
    
    # Retrieve nearest neighbors
    dists, idx = nn_index.kneighbors(shifted.reshape(1, -1), n_neighbors=k + 1)
    return shifted, idx[0], dists[0]


# ═══════════════════════════════════════════════════════════════════════════
# 6. METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(source_idx, retrieved_indices, target_col, shift_direction,
                    n_sigma, env_stats):
    """
    Compute all metrics for a single shift experiment.
    """
    source_env = {c: df_full[c].iloc[source_idx] for c in all_env if c in df_full.columns}
    
    results = []
    for ret_idx in retrieved_indices:
        if ret_idx >= len(df_full):
            continue
        
        ret_env = {c: df_full[c].iloc[ret_idx] for c in all_env if c in df_full.columns}
        
        # On-manifold distance: L2 between shifted and retrieved
        on_manifold_dist = float(np.linalg.norm(E[source_idx] - E[ret_idx]))
        
        # Target accuracy: normalized change in target property
        src_val = source_env.get(target_col, np.nan)
        ret_val = ret_env.get(target_col, np.nan)
        target_sigma = env_stats.get(target_col, {}).get('std', 1)
        
        if np.isfinite(src_val) and np.isfinite(ret_val) and target_sigma > 0:
            target_change = (ret_val - src_val) / target_sigma
            target_change_raw = ret_val - src_val
        else:
            target_change = np.nan
            target_change_raw = np.nan
        
        # Non-target preservation: mean |Δ|/σ across other properties
        nontarget_devs = []
        for col in all_env:
            if col == target_col or col not in df_full.columns:
                continue
            s_val = source_env.get(col, np.nan)
            r_val = ret_env.get(col, np.nan)
            col_sigma = env_stats.get(col, {}).get('std', 1)
            if np.isfinite(s_val) and np.isfinite(r_val) and col_sigma > 0:
                nontarget_devs.append(abs(r_val - s_val) / col_sigma)
        
        mean_nontarget_dev = float(np.mean(nontarget_devs)) if nontarget_devs else np.nan
        
        results.append({
            'on_manifold_dist': on_manifold_dist,
            'target_change_sigma': target_change,
            'target_change_raw': target_change_raw,
            'mean_nontarget_dev': mean_nontarget_dev,
        })
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 7. EXPERIMENT A: TARGETED SHIFT
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("EXPERIMENT A: TARGETED SHIFT")
print("-" * 60)

# Pre-compute global directions for each target property
print("\nComputing global property directions...")
global_directions = {}
for prop in TARGET_PROPERTIES:
    if prop not in df_full.columns:
        continue
    d, s, r = find_global_property_direction(prop)
    if d is not None:
        global_directions[prop] = {'direction': d, 'sigma': s, 'corr': r}
        print(f"  {prop}: global corr = {r:.3f}")

# Run experiments
all_shift_results = []
n_done = 0
t0 = time.time()

for i, src_idx in enumerate(source_indices):
    if i % 100 == 0 and i > 0:
        elapsed = time.time() - t0
        print(f"  {i}/{len(source_indices)} ({elapsed:.1f}s)")
    
    for prop in TARGET_PROPERTIES:
        if prop not in df_full.columns:
            continue
        
        src_val = df_full[prop].iloc[src_idx]
        if not np.isfinite(src_val):
            continue
        
        for n_sigma in SHIFT_SIGMAS:
            # ── Method 1: Local shift ──
            local_dir, local_sig, local_corr, local_pc = \
                find_local_property_direction(src_idx, prop)
            
            if local_dir is not None and local_sig > 0:
                shifted, ret_idx, ret_dists = shift_and_retrieve(
                    E[src_idx], local_dir, local_sig, n_sigma)
                metrics = compute_metrics(src_idx, ret_idx[:K_RETRIEVAL], prop,
                                          local_dir, n_sigma, env_stats)
                for m in metrics:
                    m.update({
                        'method': 'local',
                        'property': prop,
                        'n_sigma': n_sigma,
                        'source_idx': int(src_idx),
                        'source_val': float(src_val),
                        'local_corr': float(local_corr),
                        'local_pc': int(local_pc),
                    })
                all_shift_results.extend(metrics)
            
            # ── Method 2: Global shift ──
            if prop in global_directions:
                gd = global_directions[prop]
                shifted_g, ret_idx_g, ret_dists_g = shift_and_retrieve(
                    E[src_idx], gd['direction'], gd['sigma'], n_sigma)
                metrics_g = compute_metrics(src_idx, ret_idx_g[:K_RETRIEVAL], prop,
                                            gd['direction'], n_sigma, env_stats)
                for m in metrics_g:
                    m.update({
                        'method': 'global',
                        'property': prop,
                        'n_sigma': n_sigma,
                        'source_idx': int(src_idx),
                        'source_val': float(src_val),
                        'local_corr': float(gd['corr']),
                        'local_pc': -1,
                    })
                all_shift_results.extend(metrics_g)
            
            # ── Method 3: Random direction baseline ──
            rng_rand = np.random.default_rng(SEED + i * 100 + hash(prop) % 1000)
            rand_dir = rng_rand.standard_normal(N_DIMS)
            rand_dir = rand_dir / np.linalg.norm(rand_dir)
            rand_sig = np.std(E @ rand_dir)
            
            shifted_r, ret_idx_r, ret_dists_r = shift_and_retrieve(
                E[src_idx], rand_dir, rand_sig, n_sigma)
            metrics_r = compute_metrics(src_idx, ret_idx_r[:K_RETRIEVAL], prop,
                                        rand_dir, n_sigma, env_stats)
            for m in metrics_r:
                m.update({
                    'method': 'random',
                    'property': prop,
                    'n_sigma': n_sigma,
                    'source_idx': int(src_idx),
                    'source_val': float(src_val),
                    'local_corr': 0.0,
                    'local_pc': -1,
                })
            all_shift_results.extend(metrics_r)
            
            # ── Method 4: Geographic nearest neighbor ──
            geo_dists = np.sqrt((coords[:, 0] - coords[src_idx, 0])**2 +
                                (coords[:, 1] - coords[src_idx, 1])**2)
            geo_dists[src_idx] = np.inf
            geo_nbrs = np.argsort(geo_dists)[:K_RETRIEVAL]
            metrics_geo = compute_metrics(src_idx, geo_nbrs, prop,
                                          np.zeros(N_DIMS), n_sigma, env_stats)
            for m in metrics_geo:
                m.update({
                    'method': 'geographic',
                    'property': prop,
                    'n_sigma': n_sigma,
                    'source_idx': int(src_idx),
                    'source_val': float(src_val),
                    'local_corr': 0.0,
                    'local_pc': -1,
                })
            all_shift_results.extend(metrics_geo)

elapsed = time.time() - t0
print(f"\n  Completed in {elapsed:.1f}s")
print(f"  Total result rows: {len(all_shift_results):,}")

shift_df = pd.DataFrame(all_shift_results)
shift_df.to_csv(f'{OUTPUT_DIR}/experiment_a_shifts.csv', index=False)
print("  ✓ Saved experiment A results")


# ═══════════════════════════════════════════════════════════════════════════
# 8. EXPERIMENT A: AGGREGATE METRICS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("EXPERIMENT A: AGGREGATE RESULTS")
print("-" * 60)

def aggregate_shift_results(shift_df):
    """Aggregate by method × property × n_sigma."""
    
    agg = shift_df.groupby(['method', 'property', 'n_sigma']).agg(
        target_change_mean=('target_change_sigma', 'mean'),
        target_change_std=('target_change_sigma', 'std'),
        nontarget_mean=('mean_nontarget_dev', 'mean'),
        nontarget_std=('mean_nontarget_dev', 'std'),
        on_manifold_mean=('on_manifold_dist', 'mean'),
        count=('target_change_sigma', 'count'),
    ).reset_index()
    
    # Precision: fraction with target_change > 0.5σ AND nontarget < 1σ
    def precision_fn(group):
        valid = group.dropna(subset=['target_change_sigma', 'mean_nontarget_dev'])
        if len(valid) == 0:
            return 0.0
        hits = ((valid['target_change_sigma'] > 0.3) & 
                (valid['mean_nontarget_dev'] < 1.0)).sum()
        return hits / len(valid)
    
    prec = shift_df.groupby(['method', 'property', 'n_sigma']).apply(precision_fn)
    prec = prec.reset_index()
    prec.columns = ['method', 'property', 'n_sigma', 'precision']
    
    agg = agg.merge(prec, on=['method', 'property', 'n_sigma'])
    
    return agg

agg_df = aggregate_shift_results(shift_df)
agg_df.to_csv(f'{OUTPUT_DIR}/experiment_a_aggregate.csv', index=False)

# Print summary
for prop in TARGET_PROPERTIES:
    if prop not in shift_df['property'].unique():
        continue
    print(f"\n  {TARGET_PROPERTIES[prop]['label']}:")
    sub = agg_df[(agg_df['property'] == prop) & (agg_df['n_sigma'] == 1.0)]
    for _, row in sub.iterrows():
        print(f"    {row['method']:12s}: Δtarget={row['target_change_mean']:+.3f}σ, "
              f"Δother={row['nontarget_mean']:.3f}σ, prec={row['precision']:.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# 9. EXPERIMENT B: PROPERTY TRANSFER
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("EXPERIMENT B: PROPERTY TRANSFER")
print("-" * 60)

transfer_results = []
rng_t = np.random.default_rng(SEED + 10)
n_transfer = min(200, len(source_indices))
t0 = time.time()

for i in range(n_transfer):
    if i % 50 == 0 and i > 0:
        print(f"  {i}/{n_transfer} ({time.time()-t0:.1f}s)")
    
    # Pick two random sources as A (source) and B (donor)
    idx_a = source_indices[rng_t.integers(len(source_indices))]
    idx_b = source_indices[rng_t.integers(len(source_indices))]
    if idx_a == idx_b:
        continue
    
    for prop in TARGET_PROPERTIES:
        if prop not in df_full.columns:
            continue
        
        val_a = df_full[prop].iloc[idx_a]
        val_b = df_full[prop].iloc[idx_b]
        if not (np.isfinite(val_a) and np.isfinite(val_b)):
            continue
        
        # Local direction at A for this property
        local_dir, local_sig, local_corr, _ = find_local_property_direction(idx_a, prop)
        if local_dir is None or local_sig < 1e-8:
            continue
        
        # Project A and B onto local direction
        proj_a = np.dot(E[idx_a], local_dir)
        proj_b = np.dot(E[idx_b], local_dir)
        
        # Transfer: replace A's component along this direction with B's
        transferred = E[idx_a] + (proj_b - proj_a) * local_dir
        norm = np.linalg.norm(transferred)
        if norm > 0:
            transferred = transferred / norm * np.linalg.norm(E[idx_a])
        
        # Retrieve
        dists, ret_idx = nn_index.kneighbors(transferred.reshape(1, -1), n_neighbors=K_RETRIEVAL + 1)
        ret_idx = ret_idx[0, :K_RETRIEVAL]
        
        for ridx in ret_idx:
            if ridx >= len(df_full):
                continue
            ret_val = df_full[prop].iloc[ridx]
            target_sigma = env_stats.get(prop, {}).get('std', 1)
            
            # How close is the retrieved value to B's value?
            if np.isfinite(ret_val) and target_sigma > 0:
                target_error = abs(ret_val - val_b) / target_sigma
            else:
                target_error = np.nan
            
            # Non-target: how close to A?
            nt_devs = []
            for col in all_env:
                if col == prop or col not in df_full.columns:
                    continue
                s = df_full[col].iloc[idx_a]
                r = df_full[col].iloc[ridx]
                cs = env_stats.get(col, {}).get('std', 1)
                if np.isfinite(s) and np.isfinite(r) and cs > 0:
                    nt_devs.append(abs(r - s) / cs)
            
            transfer_results.append({
                'property': prop,
                'source_val': float(val_a),
                'donor_val': float(val_b),
                'retrieved_val': float(ret_val) if np.isfinite(ret_val) else np.nan,
                'target_error_sigma': float(target_error),
                'mean_nontarget_dev': float(np.mean(nt_devs)) if nt_devs else np.nan,
                'local_corr': float(local_corr),
            })

transfer_df = pd.DataFrame(transfer_results)
transfer_df.to_csv(f'{OUTPUT_DIR}/experiment_b_transfer.csv', index=False)
print(f"  ✓ Saved {len(transfer_df):,} transfer results ({time.time()-t0:.1f}s)")

# Summary
for prop in TARGET_PROPERTIES:
    sub = transfer_df[transfer_df['property'] == prop]
    if len(sub) == 0:
        continue
    print(f"  {TARGET_PROPERTIES[prop]['label']}: "
          f"target error = {sub['target_error_sigma'].mean():.3f}σ, "
          f"nontarget = {sub['mean_nontarget_dev'].mean():.3f}σ")


# ═══════════════════════════════════════════════════════════════════════════
# 10. EXPERIMENT C: ANALOGY (A:B :: C:?)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("EXPERIMENT C: ANALOGY")
print("-" * 60)

analogy_results = []
rng_a = np.random.default_rng(SEED + 20)
n_analogy = min(200, len(source_indices))
t0 = time.time()

for i in range(n_analogy):
    if i % 50 == 0 and i > 0:
        print(f"  {i}/{n_analogy} ({time.time()-t0:.1f}s)")
    
    # Pick A, B, C randomly
    abc_idx = rng_a.choice(source_indices, size=3, replace=False)
    idx_a, idx_b, idx_c = abc_idx
    
    for prop in TARGET_PROPERTIES:
        if prop not in df_full.columns:
            continue
        
        val_a = df_full[prop].iloc[idx_a]
        val_b = df_full[prop].iloc[idx_b]
        val_c = df_full[prop].iloc[idx_c]
        if not all(np.isfinite([val_a, val_b, val_c])):
            continue
        
        # Expected D value for the target property
        expected_d = val_c + (val_b - val_a)
        
        # ── Naive analogy: D = C + (B - A) ──
        d_naive = E[idx_c] + (E[idx_b] - E[idx_a])
        norm = np.linalg.norm(d_naive)
        if norm > 0:
            d_naive = d_naive / norm * np.linalg.norm(E[idx_c])
        
        dists_n, ret_n = nn_index.kneighbors(d_naive.reshape(1, -1), n_neighbors=K_RETRIEVAL + 1)
        
        # ── Local analogy: project difference onto local direction at C ──
        local_dir, local_sig, local_corr, _ = find_local_property_direction(idx_c, prop)
        if local_dir is not None and local_sig > 0:
            # Project (B-A) onto C's local property direction
            diff_proj = np.dot(E[idx_b] - E[idx_a], local_dir)
            d_local = E[idx_c] + diff_proj * local_dir
            norm_l = np.linalg.norm(d_local)
            if norm_l > 0:
                d_local = d_local / norm_l * np.linalg.norm(E[idx_c])
            
            dists_l, ret_l = nn_index.kneighbors(d_local.reshape(1, -1), n_neighbors=K_RETRIEVAL + 1)
        else:
            ret_l = ret_n  # fallback
            local_corr = 0
        
        target_sigma = env_stats.get(prop, {}).get('std', 1)
        
        for method, ret_idx in [('naive', ret_n[0, :K_RETRIEVAL]),
                                 ('local', ret_l[0, :K_RETRIEVAL])]:
            for ridx in ret_idx:
                if ridx >= len(df_full):
                    continue
                ret_val = df_full[prop].iloc[ridx]
                
                if np.isfinite(ret_val) and target_sigma > 0:
                    error = abs(ret_val - expected_d) / target_sigma
                else:
                    error = np.nan
                
                nt_devs = []
                for col in all_env:
                    if col == prop or col not in df_full.columns:
                        continue
                    s = df_full[col].iloc[idx_c]
                    r = df_full[col].iloc[ridx]
                    cs = env_stats.get(col, {}).get('std', 1)
                    if np.isfinite(s) and np.isfinite(r) and cs > 0:
                        nt_devs.append(abs(r - s) / cs)
                
                analogy_results.append({
                    'method': method,
                    'property': prop,
                    'expected_val': float(expected_d),
                    'retrieved_val': float(ret_val) if np.isfinite(ret_val) else np.nan,
                    'target_error_sigma': float(error),
                    'mean_nontarget_dev': float(np.mean(nt_devs)) if nt_devs else np.nan,
                })

analogy_df = pd.DataFrame(analogy_results)
analogy_df.to_csv(f'{OUTPUT_DIR}/experiment_c_analogy.csv', index=False)
print(f"  ✓ Saved {len(analogy_df):,} analogy results ({time.time()-t0:.1f}s)")

for prop in TARGET_PROPERTIES:
    for method in ['naive', 'local']:
        sub = analogy_df[(analogy_df['property'] == prop) & (analogy_df['method'] == method)]
        if len(sub) == 0:
            continue
        print(f"  {TARGET_PROPERTIES[prop]['label']} ({method}): "
              f"error = {sub['target_error_sigma'].mean():.3f}σ, "
              f"nontarget = {sub['mean_nontarget_dev'].mean():.3f}σ")


# ═══════════════════════════════════════════════════════════════════════════
# 11. BUILD ENHANCED DICTIONARY
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("BUILDING ENHANCED DICTIONARY")
print("-" * 60)

enhanced_dict = {
    'global_directions': {},
    'arithmetic_summary': {},
    'property_metadata': {},
}

# Global directions
for prop, gd in global_directions.items():
    enhanced_dict['global_directions'][prop] = {
        'correlation': float(gd['corr']),
        'sigma': float(gd['sigma']),
        'direction': gd['direction'].tolist(),
    }

# Per-property arithmetic summary
for prop in TARGET_PROPERTIES:
    if prop not in shift_df['property'].unique():
        continue
    
    sub_1sig = agg_df[(agg_df['property'] == prop) & (agg_df['n_sigma'] == 1.0)]
    
    prop_summary = {}
    for _, row in sub_1sig.iterrows():
        prop_summary[row['method']] = {
            'target_change_mean': float(row['target_change_mean']),
            'nontarget_mean': float(row['nontarget_mean']),
            'precision': float(row['precision']),
        }
    
    enhanced_dict['arithmetic_summary'][prop] = prop_summary
    enhanced_dict['property_metadata'][prop] = {
        'label': TARGET_PROPERTIES[prop]['label'],
        'unit': TARGET_PROPERTIES[prop]['unit'],
        'global_sigma': float(env_stats.get(prop, {}).get('std', 0)),
        'global_mean': float(env_stats.get(prop, {}).get('mean', 0)),
    }

with open(f'{OUTPUT_DIR}/enhanced_dictionary.json', 'w') as f:
    json.dump(enhanced_dict, f, indent=2, default=str)
print("  ✓ Saved enhanced dictionary")


# ═══════════════════════════════════════════════════════════════════════════
# 12. FIGURES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("GENERATING FIGURES")
print("-" * 60)

METHOD_COLORS = {
    'local': '#4CAF50', 'global': '#FF5722',
    'random': '#9E9E9E', 'geographic': '#2196F3',
}
METHOD_LABELS = {
    'local': 'Local (ours)', 'global': 'Global (naive)',
    'random': 'Random', 'geographic': 'Geographic NN',
}

# --- Fig A1: Target change by method and property (at 1σ shift) ---
fig, axes = plt.subplots(1, len(env_cols), figsize=(5 * len(env_cols), 5))
if len(env_cols) == 1:
    axes = [axes]

for i, prop in enumerate(env_cols):
    sub = shift_df[(shift_df['property'] == prop) & (shift_df['n_sigma'] == 1.0)]
    if len(sub) == 0:
        continue
    
    methods = ['local', 'global', 'random', 'geographic']
    bp_data = [sub[sub['method'] == m]['target_change_sigma'].dropna().values for m in methods]
    
    bp = axes[i].boxplot(bp_data, labels=[METHOD_LABELS.get(m, m) for m in methods],
                          patch_artist=True, widths=0.6, showfliers=False,
                          medianprops=dict(color='black', lw=2))
    for j, patch in enumerate(bp['boxes']):
        patch.set_facecolor(METHOD_COLORS.get(methods[j], '#CCC'))
        patch.set_alpha(0.7)
    
    axes[i].axhline(y=0, color='gray', ls='--', lw=0.8)
    axes[i].set_ylabel('Target Change (σ)' if i == 0 else '')
    axes[i].set_title(TARGET_PROPERTIES[prop]['label'], fontweight='bold')
    axes[i].tick_params(axis='x', rotation=30, labelsize=9)
    axes[i].grid(True, alpha=0.3, axis='y')

plt.suptitle('Experiment A: Targeted Shift (1σ) — Target Property Change',
             fontweight='bold', fontsize=14)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_expA_target_change.png', dpi=300, facecolor='white')
plt.close(fig)
print("  ✓ Exp A: target change")

# --- Fig A2: Non-target preservation ---
fig, axes = plt.subplots(1, len(env_cols), figsize=(5 * len(env_cols), 5))
if len(env_cols) == 1:
    axes = [axes]

for i, prop in enumerate(env_cols):
    sub = shift_df[(shift_df['property'] == prop) & (shift_df['n_sigma'] == 1.0)]
    if len(sub) == 0:
        continue
    
    methods = ['local', 'global', 'random', 'geographic']
    bp_data = [sub[sub['method'] == m]['mean_nontarget_dev'].dropna().values for m in methods]
    
    bp = axes[i].boxplot(bp_data, labels=[METHOD_LABELS.get(m, m) for m in methods],
                          patch_artist=True, widths=0.6, showfliers=False,
                          medianprops=dict(color='black', lw=2))
    for j, patch in enumerate(bp['boxes']):
        patch.set_facecolor(METHOD_COLORS.get(methods[j], '#CCC'))
        patch.set_alpha(0.7)
    
    axes[i].set_ylabel('Non-target Deviation (σ)' if i == 0 else '')
    axes[i].set_title(TARGET_PROPERTIES[prop]['label'], fontweight='bold')
    axes[i].tick_params(axis='x', rotation=30, labelsize=9)
    axes[i].grid(True, alpha=0.3, axis='y')

plt.suptitle('Experiment A: Targeted Shift (1σ) — Non-Target Preservation',
             fontweight='bold', fontsize=14)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_expA_nontarget.png', dpi=300, facecolor='white')
plt.close(fig)
print("  ✓ Exp A: non-target preservation")

# --- Fig A3: Precision vs shift magnitude ---
fig, ax = plt.subplots(figsize=(10, 6))
for method in ['local', 'global', 'random']:
    sub = agg_df[agg_df['method'] == method]
    # Average precision across properties
    prec_by_sigma = sub.groupby('n_sigma')['precision'].mean()
    ax.plot(prec_by_sigma.index, prec_by_sigma.values, 'o-',
            color=METHOD_COLORS.get(method, '#CCC'), lw=2, ms=8,
            label=METHOD_LABELS.get(method, method))
ax.set_xlabel('Shift Magnitude (σ)')
ax.set_ylabel('Precision (target > 0.3σ AND non-target < 1σ)')
ax.set_title('Shift Precision vs. Magnitude')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_expA_precision_vs_sigma.png', dpi=300, facecolor='white')
plt.close(fig)
print("  ✓ Exp A: precision vs sigma")

# --- Fig B: Transfer scatter ---
fig, axes = plt.subplots(1, len(env_cols), figsize=(5 * len(env_cols), 5))
if len(env_cols) == 1:
    axes = [axes]

for i, prop in enumerate(env_cols):
    sub = transfer_df[transfer_df['property'] == prop].dropna(subset=['donor_val', 'retrieved_val'])
    if len(sub) == 0:
        continue
    
    axes[i].scatter(sub['donor_val'], sub['retrieved_val'], s=3, alpha=0.3, color='#4CAF50')
    lims = [min(sub['donor_val'].min(), sub['retrieved_val'].min()),
            max(sub['donor_val'].max(), sub['retrieved_val'].max())]
    axes[i].plot(lims, lims, 'k--', lw=1, alpha=0.5)
    axes[i].set_xlabel(f'Donor Value ({TARGET_PROPERTIES[prop]["unit"]})')
    axes[i].set_ylabel(f'Retrieved Value ({TARGET_PROPERTIES[prop]["unit"]})')
    axes[i].set_title(TARGET_PROPERTIES[prop]['label'], fontweight='bold')
    
    # R² annotation
    from scipy.stats import pearsonr
    r, _ = pearsonr(sub['donor_val'], sub['retrieved_val'])
    axes[i].annotate(f'r = {r:.3f}', xy=(0.05, 0.90), xycoords='axes fraction',
                     fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle('Experiment B: Property Transfer — Donor vs. Retrieved',
             fontweight='bold', fontsize=14)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_expB_transfer.png', dpi=300, facecolor='white')
plt.close(fig)
print("  ✓ Exp B: transfer scatter")

# --- Fig C: Analogy comparison ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i, method in enumerate(['naive', 'local']):
    sub = analogy_df[analogy_df['method'] == method].dropna(subset=['target_error_sigma'])
    if len(sub) == 0:
        continue
    
    # Group by property
    for prop in env_cols:
        prop_sub = sub[sub['property'] == prop]
        if len(prop_sub) == 0:
            continue
        axes[i].hist(prop_sub['target_error_sigma'].clip(0, 5), bins=40, alpha=0.5,
                     label=TARGET_PROPERTIES[prop]['label'], density=True)
    
    axes[i].set_xlabel('Target Error (σ)')
    axes[i].set_ylabel('Density')
    axes[i].set_title(f'{"Naive" if method == "naive" else "Local"} Analogy', fontweight='bold')
    axes[i].legend(fontsize=8)
    axes[i].set_xlim(0, 5)
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Experiment C: Analogy — Target Error Distribution',
             fontweight='bold', fontsize=14)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_expC_analogy.png', dpi=300, facecolor='white')
plt.close(fig)
print("  ✓ Exp C: analogy")


# ═══════════════════════════════════════════════════════════════════════════
# 13. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SUMMARY: COMPOSITIONAL ARITHMETIC")
print("-" * 60)

print(f"\n  DATA: {len(E):,} vectors, {len(source_indices)} source locations")

print(f"\n  EXPERIMENT A: TARGETED SHIFT (at 1σ)")
for prop in env_cols:
    sub = agg_df[(agg_df['property'] == prop) & (agg_df['n_sigma'] == 1.0)]
    if len(sub) == 0:
        continue
    print(f"\n    {TARGET_PROPERTIES[prop]['label']}:")
    for _, row in sub.iterrows():
        print(f"      {row['method']:12s}: Δtarget={row['target_change_mean']:+.3f}σ, "
              f"Δother={row['nontarget_mean']:.3f}σ, prec={row['precision']:.3f}")

print(f"\n  EXPERIMENT B: PROPERTY TRANSFER")
for prop in env_cols:
    sub = transfer_df[transfer_df['property'] == prop]
    if len(sub) == 0:
        continue
    print(f"    {TARGET_PROPERTIES[prop]['label']}: "
          f"target err = {sub['target_error_sigma'].mean():.3f}σ, "
          f"nontarget = {sub['mean_nontarget_dev'].mean():.3f}σ")

print(f"\n  EXPERIMENT C: ANALOGY")
for method in ['naive', 'local']:
    sub = analogy_df[analogy_df['method'] == method]
    if len(sub) == 0:
        continue
    print(f"    {method:6s}: target err = {sub['target_error_sigma'].mean():.3f}σ, "
          f"nontarget = {sub['mean_nontarget_dev'].mean():.3f}σ")

print(f"\n  OUTPUTS: {OUTPUT_DIR}/")
for f_name in sorted(os.listdir(OUTPUT_DIR)):
    fp = os.path.join(OUTPUT_DIR, f_name)
    if os.path.isfile(fp):
        print(f"    {f_name} ({os.path.getsize(fp)/1024:.1f} KB)")

print("\n" + "=" * 70)
print("Complete.")

print("  → Results determine which operations are viable for deployment")
print("-" * 60)

del E, nn_index
gc.collect()
print("Done.")
