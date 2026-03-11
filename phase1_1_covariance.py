"""
Global Covariance Structure of AlphaEarth Embeddings

Characterizes how the 64 embedding dimensions co-vary with each other:
  1. 64x64 covariance and correlation matrices (combined + per-year)
  2. Eigendecomposition: principal directions, effective dimensionality
  3. Hierarchical clustering of co-varying dimension groups
  4. Per-year eigenstructure stability (subspace angles)

Input:  Yearly parquet files from data/unified_conus/
Output: manifold_results/ (CSV matrices, cluster assignments, figures)

Author: Mashrekur Rahman | 2026
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.linalg import subspace_angles
import json
import os
import gc
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR = '../../data/unified_conus'
DICT_DIR = '../results'
OUTPUT_DIR = 'manifold_results'
FIG_DIR = f'{OUTPUT_DIR}/figures'

YEARS = list(range(2017, 2024))   # 2017–2023
SUBSAMPLE_TOTAL = 1_000_000       # 1M total → ~143K per year
SEED = 42

AE_COLS = [f'A{i:02d}' for i in range(64)]
N_DIMS = 64

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.linewidth': 0.8,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print("-" * 60)
print("GLOBAL COVARIANCE STRUCTURE")
print("  Configuration: All 7 years (2017–2023), balanced sampling")
print("-" * 60)

# ── Dimension Dictionary Loader ────────────────────────────────────
def load_dimension_dictionary():
    """
    Load dimension dictionary with robust column detection.
    The column names may vary depending on which script generated it.
    """
    dd_path = f'{DICT_DIR}/dimension_dictionary.csv'
    if not os.path.exists(dd_path):
        print(f"  ⚠ dimension dictionary not found at {dd_path}")
        return None
    
    dd = pd.read_csv(dd_path)
    print(f"  ✓ Loaded dimension dictionary: {len(dd)} rows, columns: {dd.columns.tolist()}")
    
    # Detect column names — different scripts may use different names
    col_map = {}
    
    # Dimension column
    for c in ['dimension', 'dim', 'Dimension']:
        if c in dd.columns:
            col_map['dimension'] = c; break
    
    # Primary variable column
    for c in ['sp_primary', 'primary_variable', 'primary_var', 'spearman_primary', 'variable', 'top_variable']:
        if c in dd.columns:
            col_map['primary_variable'] = c; break
    
    # Spearman rho column
    for c in ['sp_rho', 'spearman_rho', 'rho', 'abs_rho', 'spearman_r', 'correlation']:
        if c in dd.columns:
            col_map['spearman_rho'] = c; break
    
    # Category column
    for c in ['sp_category', 'category', 'env_category', 'var_category', 'group']:
        if c in dd.columns:
            col_map['category'] = c; break
    
    print(f"  Column mapping: {col_map}")
    
    # Build standardized lookup dictionaries
    lookups = {}
    dim_col = col_map.get('dimension')
    
    if dim_col:
        if 'primary_variable' in col_map:
            lookups['dim_to_var'] = dict(zip(dd[dim_col], dd[col_map['primary_variable']]))
        if 'spearman_rho' in col_map:
            lookups['dim_to_rho'] = dict(zip(dd[dim_col], dd[col_map['spearman_rho']]))
        if 'category' in col_map:
            lookups['dim_to_cat'] = dict(zip(dd[dim_col], dd[col_map['category']]))
    
    return dd, col_map, lookups


def load_embeddings_all_years(years=YEARS, total_n=SUBSAMPLE_TOTAL, seed=SEED):
    """
    Load embedding vectors with balanced sampling across all years.
    ~143K per year × 7 years ≈ 1M total.
    """
    per_year = total_n // len(years)
    rng = np.random.default_rng(seed)
    
    all_embeddings = []
    all_coords = []
    all_years = []
    year_embeddings = {}  # Keep per-year arrays for stability analysis
    
    print(f"\nLoading embeddings: {per_year:,} per year × {len(years)} years = {per_year*len(years):,} total")
    
    for year in years:
        fp = f'{DATA_DIR}/conus_{year}_unified.parquet'
        if not os.path.exists(fp):
            print(f"  ⚠ Missing: {fp}")
            continue
        
        df = pd.read_parquet(fp, columns=['longitude', 'latitude'] + AE_COLS)
        n_total = len(df)
        
        # Subsample
        idx = rng.choice(n_total, size=min(per_year, n_total), replace=False)
        df_sub = df.iloc[idx]
        
        E_year = df_sub[AE_COLS].values.astype(np.float64)
        coords_year = df_sub[['longitude', 'latitude']].values
        
        all_embeddings.append(E_year)
        all_coords.append(coords_year)
        all_years.extend([year] * len(E_year))
        year_embeddings[year] = E_year
        
        print(f"  {year}: {n_total:,} total → {len(E_year):,} sampled")
        
        del df, df_sub
        gc.collect()
    
    # Combine
    E = np.vstack(all_embeddings)
    coords = np.vstack(all_coords)
    years_arr = np.array(all_years)
    
    print(f"\n  Combined: {E.shape[0]:,} vectors × {E.shape[1]} dims")
    print(f"  Value range: [{E.min():.4f}, {E.max():.4f}]")
    norms = np.linalg.norm(E, axis=1)
    print(f"  L2 norm range: [{norms.min():.6f}, {norms.max():.6f}]")
    print(f"  Mean L2 norm: {norms.mean():.6f}")
    
    return E, coords, years_arr, year_embeddings

E, coords, years_arr, year_embeddings = load_embeddings_all_years()


def compute_covariance_structure(E, dim_names=AE_COLS, label="Combined"):
    """Compute 64×64 covariance and correlation matrices."""
    print(f"\nComputing 64×64 inter-dimension structure ({label}, n={E.shape[0]:,})...")
    
    cov_matrix = np.cov(E, rowvar=False)
    corr_matrix = np.corrcoef(E, rowvar=False)
    
    cov_df = pd.DataFrame(cov_matrix, index=dim_names, columns=dim_names)
    corr_df = pd.DataFrame(corr_matrix, index=dim_names, columns=dim_names)
    
    # Off-diagonal statistics
    mask = ~np.eye(N_DIMS, dtype=bool)
    off_diag = corr_matrix[mask]
    
    print(f"  Variance range: [{np.diag(cov_matrix).min():.6f}, {np.diag(cov_matrix).max():.6f}]")
    print(f"  Off-diagonal |r|: mean={np.abs(off_diag).mean():.4f}, max={np.abs(off_diag).max():.4f}")
    print(f"  Pairs with |r| > 0.5: {(np.abs(off_diag) > 0.5).sum() // 2}")
    print(f"  Pairs with |r| > 0.3: {(np.abs(off_diag) > 0.3).sum() // 2}")
    
    return cov_df, corr_df

# Combined
cov_df, corr_df = compute_covariance_structure(E, label="All years combined")
cov_df.to_csv(f'{OUTPUT_DIR}/covariance_matrix_64x64.csv')
corr_df.to_csv(f'{OUTPUT_DIR}/correlation_matrix_64x64.csv')
print("  ✓ Saved combined covariance and correlation matrices")


def eigendecompose(cov_df, label="Combined"):
    """Eigendecompose the 64×64 covariance matrix."""
    print(f"\nEigendecomposing ({label})...")
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov_df.values)
    
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    total_var = eigenvalues.sum()
    var_explained = eigenvalues / total_var
    cumvar = np.cumsum(var_explained)
    
    print(f"  Total variance: {total_var:.6f}")
    for i in range(min(10, len(eigenvalues))):
        print(f"    PC{i+1}: λ={eigenvalues[i]:.6f} "
              f"({var_explained[i]*100:.2f}%, cum: {cumvar[i]*100:.2f}%)")
    
    for thr in [0.80, 0.90, 0.95, 0.99]:
        n_comp = np.searchsorted(cumvar, thr) + 1
        print(f"  {thr*100:.0f}% variance: {n_comp} components")
    
    pr = (eigenvalues.sum())**2 / (eigenvalues**2).sum()
    print(f"  Participation ratio: {pr:.2f} (of {N_DIMS})")
    
    return eigenvalues, eigenvectors, var_explained, cumvar

eigenvalues, eigenvectors, var_explained, cumvar = eigendecompose(cov_df)

# Save
eigen_df = pd.DataFrame({
    'component': [f'PC{i+1}' for i in range(N_DIMS)],
    'eigenvalue': eigenvalues,
    'variance_explained': var_explained,
    'cumulative_variance': cumvar,
})
eigen_df.to_csv(f'{OUTPUT_DIR}/eigenvalues.csv', index=False)

evec_df = pd.DataFrame(eigenvectors, index=AE_COLS, columns=[f'PC{i+1}' for i in range(N_DIMS)])
evec_df.to_csv(f'{OUTPUT_DIR}/eigenvectors.csv')
print("  ✓ Saved eigenvalues and eigenvectors")


def per_year_analysis(year_embeddings):
    """
    Compute eigendecomposition per year and measure stability.
    
    This tests whether the GEOMETRIC STRUCTURE is stable over time,
    complementing the finding that dimension values are temporally stable.
    
    Metrics:
    - Per-year eigenvalue spectra (do they match?)
    - Subspace angles between top-k eigenvector subspaces across years
    - Per-year participation ratio
    """
    print("\n" + "=" * 70)
    print("PER-YEAR EIGENDECOMPOSITION")
    print("-" * 60)
    
    year_results = {}
    
    for year, E_year in sorted(year_embeddings.items()):
        cov_year = np.cov(E_year, rowvar=False)
        evals, evecs = np.linalg.eigh(cov_year)
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]
        
        total = evals.sum()
        var_exp = evals / total
        cumvar = np.cumsum(var_exp)
        pr = total**2 / (evals**2).sum()
        
        n80 = np.searchsorted(cumvar, 0.80) + 1
        n90 = np.searchsorted(cumvar, 0.90) + 1
        n95 = np.searchsorted(cumvar, 0.95) + 1
        
        year_results[year] = {
            'eigenvalues': evals,
            'eigenvectors': evecs,
            'var_explained': var_exp,
            'cumvar': cumvar,
            'participation_ratio': pr,
            'n80': n80, 'n90': n90, 'n95': n95,
        }
        
        print(f"  {year}: PR={pr:.2f}, 80%→{n80}, 90%→{n90}, 95%→{n95}, "
              f"top PC={var_exp[0]*100:.1f}%")
    
    # Subspace angle stability
    # For each pair of years, compute the principal angle between
    # their top-k eigenvector subspaces
    print("\n  Subspace stability (principal angles between year pairs):")
    years_sorted = sorted(year_results.keys())
    
    stability_records = []
    for k in [5, 10, 20]:
        angles_all = []
        for i in range(len(years_sorted)):
            for j in range(i + 1, len(years_sorted)):
                y1, y2 = years_sorted[i], years_sorted[j]
                V1 = year_results[y1]['eigenvectors'][:, :k]
                V2 = year_results[y2]['eigenvectors'][:, :k]
                
                angles = subspace_angles(V1, V2)
                max_angle_deg = np.degrees(angles.max())
                mean_angle_deg = np.degrees(angles.mean())
                
                angles_all.append(max_angle_deg)
                stability_records.append({
                    'year1': y1, 'year2': y2, 'k': k,
                    'max_angle_deg': max_angle_deg,
                    'mean_angle_deg': mean_angle_deg,
                })
        
        print(f"    k={k:2d}: mean max angle = {np.mean(angles_all):.2f}° "
              f"(range: {np.min(angles_all):.2f}° – {np.max(angles_all):.2f}°)")
    
    # Save per-year eigenvalues
    per_year_evals = pd.DataFrame({
        f'{year}': year_results[year]['eigenvalues'] for year in years_sorted
    })
    per_year_evals.index = [f'PC{i+1}' for i in range(N_DIMS)]
    per_year_evals.to_csv(f'{OUTPUT_DIR}/per_year_eigenvalues.csv')
    
    stability_df = pd.DataFrame(stability_records)
    stability_df.to_csv(f'{OUTPUT_DIR}/per_year_stability.csv', index=False)
    
    print("  ✓ Saved per-year eigenvalues and stability metrics")
    
    return year_results, stability_df

year_results, stability_df = per_year_analysis(year_embeddings)


def plot_scree(eigenvalues, var_explained, cumvar, year_results):
    """Scree plot with per-year comparison."""
    print("\nGenerating scree plot...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(1, N_DIMS + 1)
    
    # (a) Individual variance explained — combined
    axes[0].bar(x, var_explained * 100, color='steelblue', alpha=0.8, edgecolor='none')
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Variance Explained (%)')
    axes[0].set_title('(a) Individual Variance (All Years)')
    axes[0].set_xlim(0.5, N_DIMS + 0.5)
    axes[0].set_xticks([1, 10, 20, 30, 40, 50, 60, 64])
    
    # (b) Cumulative — combined
    axes[1].plot(x, cumvar * 100, 'o-', color='steelblue', markersize=3, linewidth=1.5)
    for thr in [0.80, 0.90, 0.95]:
        axes[1].axhline(y=thr*100, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
        nc = np.searchsorted(cumvar, thr) + 1
        axes[1].annotate(f'{thr*100:.0f}% → {nc} PCs',
                         xy=(nc, thr * 100), xytext=(nc + 5, thr * 100 - 3),
                         arrowprops=dict(arrowstyle='->', color='gray', lw=0.8),
                         fontsize=8, color='dimgray')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Variance (%)')
    axes[1].set_title('(b) Cumulative Variance (All Years)')
    axes[1].set_xlim(0.5, N_DIMS + 0.5)
    axes[1].set_ylim(0, 102)
    axes[1].set_xticks([1, 10, 20, 30, 40, 50, 60, 64])
    
    # (c) Per-year overlay — eigenvalue spectra comparison
    cmap = plt.cm.viridis(np.linspace(0, 1, len(year_results)))
    for i, (year, res) in enumerate(sorted(year_results.items())):
        axes[2].plot(x, res['var_explained'] * 100, '-', color=cmap[i],
                     linewidth=1.2, alpha=0.8, label=str(year))
    axes[2].set_xlabel('Principal Component')
    axes[2].set_ylabel('Variance Explained (%)')
    axes[2].set_title('(c) Per-Year Comparison')
    axes[2].set_xlim(0.5, N_DIMS + 0.5)
    axes[2].set_xticks([1, 10, 20, 30, 40, 50, 60, 64])
    axes[2].legend(fontsize=7, ncol=2, loc='upper right')
    
    plt.tight_layout()
    fig.savefig(f'{FIG_DIR}/fig_scree_plot.png', dpi=300, facecolor='white')
    fig.savefig(f'{FIG_DIR}/fig_scree_plot.pdf', dpi=300, facecolor='white')
    plt.show()
    print("  ✓ Saved scree plot")

plot_scree(eigenvalues, var_explained, cumvar, year_results)


def plot_correlation_heatmap(corr_df):
    """64×64 inter-dimension correlation heatmap with hierarchical clustering."""
    print("\nGenerating inter-dimension correlation heatmap...")
    
    corr_vals = corr_df.values
    dist = 1 - np.abs(corr_vals)
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    dist = np.clip(dist, 0, None)
    
    condensed = squareform(dist)
    linkage_mat = linkage(condensed, method='ward')
    
    dendro = dendrogram(linkage_mat, no_plot=True)
    order = dendro['leaves']
    ordered_labels = [AE_COLS[i] for i in order]
    corr_ordered = corr_df.loc[ordered_labels, ordered_labels]
    
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr_ordered.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ax.set_xticks(range(N_DIMS))
    ax.set_yticks(range(N_DIMS))
    ax.set_xticklabels(ordered_labels, fontsize=6, rotation=90)
    ax.set_yticklabels(ordered_labels, fontsize=6)
    ax.set_title('Inter-Dimension Correlation Matrix (Ward Clustering)')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Pearson r')
    
    plt.tight_layout()
    fig.savefig(f'{FIG_DIR}/fig_correlation_heatmap.png', dpi=300, facecolor='white')
    fig.savefig(f'{FIG_DIR}/fig_correlation_heatmap.pdf', dpi=300, facecolor='white')
    plt.show()
    print("  ✓ Saved correlation heatmap")
    
    return linkage_mat, order

linkage_mat, cluster_order = plot_correlation_heatmap(corr_df)


def identify_dimension_clusters(corr_df, linkage_mat):
    """Identify co-varying dimension clusters using hierarchical clustering."""
    print("\nIdentifying dimension clusters...")
    
    from sklearn.metrics import silhouette_score
    
    dist = 1 - np.abs(corr_df.values)
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    dist = np.clip(dist, 0, None)
    
    results = []
    for n_clusters in range(3, 26):
        labels = fcluster(linkage_mat, n_clusters, criterion='maxclust')
        if len(set(labels)) > 1:
            sil = silhouette_score(dist, labels, metric='precomputed')
        else:
            sil = -1
        sizes = [int(np.sum(labels == c)) for c in sorted(set(labels))]
        results.append({
            'n_clusters': n_clusters, 'silhouette': sil,
            'sizes': sizes, 'labels': labels.copy(),
        })
        print(f"  k={n_clusters:2d}: silhouette={sil:.4f}, sizes={sizes}")
    
    best_idx = max(range(len(results)), key=lambda i: results[i]['silhouette'])
    best_k = results[best_idx]['n_clusters']
    best_labels = results[best_idx]['labels']
    print(f"\n  Best: k={best_k} (silhouette={results[best_idx]['silhouette']:.4f})")
    
    # Build cluster assignment table
    cluster_df = pd.DataFrame({'dimension': AE_COLS, 'cluster_id': best_labels})
    
    # Merge dimension dictionary
    p1 = load_dimension_dictionary()
    if p1 is not None:
        dd, col_map, lookups = p1
        if 'dim_to_var' in lookups:
            cluster_df['primary_variable'] = cluster_df['dimension'].map(lookups['dim_to_var'])
        if 'dim_to_rho' in lookups:
            cluster_df['spearman_rho'] = cluster_df['dimension'].map(lookups['dim_to_rho'])
        if 'dim_to_cat' in lookups:
            cluster_df['category'] = cluster_df['dimension'].map(lookups['dim_to_cat'])
    
    cluster_df = cluster_df.sort_values(['cluster_id', 'dimension']).reset_index(drop=True)
    return cluster_df, results, best_k

cluster_df, cluster_results, best_k = identify_dimension_clusters(corr_df, linkage_mat)
cluster_df.to_csv(f'{OUTPUT_DIR}/dimension_clusters.csv', index=False)
print("  ✓ Saved dimension clusters")


def build_cluster_summary(cluster_df, cov_df, corr_df):
    """Build detailed cluster summary ."""
    print("\nBuilding cluster summary...")
    
    has_category = 'category' in cluster_df.columns
    has_var = 'primary_variable' in cluster_df.columns
    
    summary = {}
    for cid in sorted(cluster_df['cluster_id'].unique()):
        members = cluster_df[cluster_df['cluster_id'] == cid]['dimension'].tolist()
        
        # Within-cluster correlations
        sub_corr = corr_df.loc[members, members].values
        mask = ~np.eye(len(members), dtype=bool)
        within_corr = float(np.abs(sub_corr[mask]).mean()) if mask.sum() > 0 else 1.0
        
        # Within-cluster covariance
        sub_cov = cov_df.loc[members, members].values
        
        # Physical label
        cluster_rows = cluster_df[cluster_df['cluster_id'] == cid]
        if has_category:
            cats = cluster_rows['category'].dropna()
            label = cats.value_counts().index[0] if len(cats) > 0 else 'Mixed'
        else:
            label = 'Unknown'
        
        prim_vars = cluster_rows['primary_variable'].dropna().tolist() if has_var else []
        
        summary[int(cid)] = {
            'cluster_id': int(cid),
            'n_members': len(members),
            'dimensions': members,
            'mean_within_correlation': within_corr,
            'physical_label': label,
            'primary_variables': prim_vars,
            'covariance_matrix': sub_cov.tolist(),
        }
        
        print(f"  Cluster {cid} ({label}): {len(members)} dims, mean |r|={within_corr:.3f}")
        print(f"    Dims: {members}")
        if prim_vars:
            print(f"    Vars: {prim_vars[:6]}{'...' if len(prim_vars) > 6 else ''}")
    
    with open(f'{OUTPUT_DIR}/cluster_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  ✓ Saved cluster summary ({len(summary)} clusters)")
    return summary

cluster_summary = build_cluster_summary(cluster_df, cov_df, corr_df)


def plot_annotated_dendrogram(linkage_mat, cluster_df, best_k):
    """Dendrogram with environmental category colors."""
    print("\nGenerating annotated dendrogram...")
    
    CATEGORY_COLORS = {
        'Terrain': '#8B4513', 'Soil': '#DAA520', 'Vegetation': '#228B22',
        'Temperature': '#DC143C', 'Climate': '#4169E1', 'Hydrology': '#00CED1',
        'Urban': '#696969', 'Radiation': '#FFD700', 'Mixed': '#AAAAAA',
        'Unknown': '#AAAAAA',
    }
    
    fig, ax = plt.subplots(figsize=(16, 8))
    dendro = dendrogram(
        linkage_mat, labels=AE_COLS, leaf_rotation=90, leaf_font_size=8,
        ax=ax, color_threshold=linkage_mat[-(best_k - 1), 2],
    )
    ax.set_title(f'Dimension Cluster Dendrogram (k={best_k})')
    ax.set_ylabel('Distance (1 − |r|, Ward)')
    ax.set_xlabel('AlphaEarth Dimension')
    
    if 'category' in cluster_df.columns:
        dim_to_cat = dict(zip(cluster_df['dimension'], cluster_df['category'].fillna('Unknown')))
        for lbl in ax.get_xticklabels():
            cat = dim_to_cat.get(lbl.get_text(), 'Unknown')
            lbl.set_color(CATEGORY_COLORS.get(cat, '#AAAAAA'))
            lbl.set_fontweight('bold')
        
        from matplotlib.patches import Patch
        cats_present = set(cluster_df['category'].dropna())
        legend_elements = [Patch(facecolor=CATEGORY_COLORS.get(c, '#AAA'), label=c)
                           for c in sorted(cats_present) if c in CATEGORY_COLORS]
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8,
                      title='Env. Category', title_fontsize=9)
    
    plt.tight_layout()
    fig.savefig(f'{FIG_DIR}/fig_cluster_dendrogram.png', dpi=300, facecolor='white')
    fig.savefig(f'{FIG_DIR}/fig_cluster_dendrogram.pdf', dpi=300, facecolor='white')
    plt.show()
    print("  ✓ Saved cluster dendrogram")

plot_annotated_dendrogram(linkage_mat, cluster_df, best_k)


def interpret_eigenvectors(eigenvectors, var_explained, n_top=5):
    """
    For each top eigenvector, identify strongest-contributing dimensions
    and cross-reference with dimension dictionary.
    """
    print("\nInterpreting top eigenvectors...")
    
    p1 = load_dimension_dictionary()
    dim_to_var = {}
    dim_to_rho = {}
    dim_to_cat = {}
    if p1 is not None:
        _, _, lookups = p1
        dim_to_var = lookups.get('dim_to_var', {})
        dim_to_rho = lookups.get('dim_to_rho', {})
        dim_to_cat = lookups.get('dim_to_cat', {})
    
    CATEGORY_COLORS = {
        'Terrain': '#8B4513', 'Soil': '#DAA520', 'Vegetation': '#228B22',
        'Temperature': '#DC143C', 'Climate': '#4169E1', 'Hydrology': '#00CED1',
        'Urban': '#696969', 'Radiation': '#FFD700',
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    interpretations = []
    
    for pc_idx in range(min(6, eigenvectors.shape[1])):
        evec = eigenvectors[:, pc_idx]
        top_idx = np.argsort(np.abs(evec))[::-1][:n_top]
        
        pc_info = {'component': f'PC{pc_idx+1}', 'top_dimensions': []}
        
        print(f"\n  PC{pc_idx+1} (var={var_explained[pc_idx]*100:.2f}%):")
        for didx in top_idx:
            dim = AE_COLS[didx]
            w = evec[didx]
            var = dim_to_var.get(dim, '?')
            rho = dim_to_rho.get(dim, 0)
            if rho != 0:
                rho = float(rho)
            print(f"    {dim} (w={w:+.4f}) → {var} (ρ={rho:+.3f})")
            pc_info['top_dimensions'].append({
                'dimension': dim, 'weight': float(w),
                'primary_variable': var, 'spearman_rho': float(rho),
            })
        interpretations.append(pc_info)
        
        # Bar plot
        ax = axes[pc_idx]
        colors = [CATEGORY_COLORS.get(dim_to_cat.get(d, ''), '#CCCCCC') for d in AE_COLS]
        ax.bar(range(N_DIMS), evec, color=colors, alpha=0.8, edgecolor='none')
        ax.set_title(f'PC{pc_idx+1} ({var_explained[pc_idx]*100:.1f}%)')
        ax.set_xlabel('Dimension Index')
        ax.set_ylabel('Weight')
        ax.set_xlim(-1, N_DIMS)
        ax.axhline(y=0, color='black', linewidth=0.5)
        
        # Annotate top 3
        for didx in top_idx[:3]:
            dim = AE_COLS[didx]
            w = evec[didx]
            var = dim_to_var.get(dim, '?')
            short_var = var[:12] if isinstance(var, str) else '?'
            ax.annotate(f'{dim}\n({short_var})', xy=(didx, w),
                        fontsize=6, ha='center',
                        va='bottom' if w > 0 else 'top')
    
    plt.suptitle('Top 6 Principal Components — Eigenvector Weights\n'
                 '(colored by environmental category)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(f'{FIG_DIR}/fig_eigenvector_interpretation.png', dpi=300,
                facecolor='white', bbox_inches='tight')
    fig.savefig(f'{FIG_DIR}/fig_eigenvector_interpretation.pdf', dpi=300,
                facecolor='white', bbox_inches='tight')
    plt.show()
    print("\n  ✓ Saved eigenvector interpretation")
    
    return interpretations

pc_interpretations = interpret_eigenvectors(eigenvectors, var_explained)


def plot_per_year_stability(year_results, stability_df):
    """Visualize geometric stability across years."""
    print("\nGenerating per-year stability figure...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # (a) Participation ratio per year
    years = sorted(year_results.keys())
    prs = [year_results[y]['participation_ratio'] for y in years]
    axes[0].bar([str(y) for y in years], prs, color='steelblue', alpha=0.8)
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Participation Ratio')
    axes[0].set_title('(a) Effective Dimensionality by Year')
    axes[0].axhline(y=np.mean(prs), color='red', linestyle='--', alpha=0.5,
                     label=f'Mean: {np.mean(prs):.1f}')
    axes[0].legend(fontsize=9)
    
    # (b) Components for 90% variance per year
    n90s = [year_results[y]['n90'] for y in years]
    axes[1].bar([str(y) for y in years], n90s, color='coral', alpha=0.8)
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Components for 90% Variance')
    axes[1].set_title('(b) Dimensionality (90% Threshold) by Year')
    axes[1].axhline(y=np.mean(n90s), color='red', linestyle='--', alpha=0.5,
                     label=f'Mean: {np.mean(n90s):.1f}')
    axes[1].legend(fontsize=9)
    
    # (c) Subspace angles by k
    for k_val in stability_df['k'].unique():
        sub = stability_df[stability_df['k'] == k_val]
        axes[2].bar(f'k={k_val}', sub['max_angle_deg'].mean(), 
                     yerr=sub['max_angle_deg'].std(),
                     alpha=0.8, capsize=5)
    axes[2].set_xlabel('Subspace Size')
    axes[2].set_ylabel('Mean Max Principal Angle (°)')
    axes[2].set_title('(c) Subspace Stability Across Years')
    
    plt.tight_layout()
    fig.savefig(f'{FIG_DIR}/fig_per_year_stability.png', dpi=300, facecolor='white')
    fig.savefig(f'{FIG_DIR}/fig_per_year_stability.pdf', dpi=300, facecolor='white')
    plt.show()
    print("  ✓ Saved per-year stability figure")

plot_per_year_stability(year_results, stability_df)


def plot_silhouette_analysis(cluster_results):
    """Plot silhouette scores vs number of clusters."""
    print("\nPlotting silhouette analysis...")
    
    ks = [r['n_clusters'] for r in cluster_results]
    sils = [r['silhouette'] for r in cluster_results]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, sils, 'o-', color='steelblue', markersize=6, linewidth=1.5)
    best_idx = np.argmax(sils)
    ax.plot(ks[best_idx], sils[best_idx], 'o', color='red', markersize=10,
            zorder=5, label=f'Best: k={ks[best_idx]}')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Cluster Selection: Silhouette Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(f'{FIG_DIR}/fig_silhouette_analysis.png', dpi=300, facecolor='white')
    plt.show()
    print("  ✓ Saved silhouette analysis")

plot_silhouette_analysis(cluster_results)


def print_summary(eigenvalues, var_explained, cumvar, cluster_summary, 
                  pc_interpretations, year_results, stability_df):
    """Summary of covariance analysis findings."""
    
    print("\n" + "=" * 70)
    print("SUMMARY: GLOBAL COVARIANCE STRUCTURE")
    print("-" * 60)
    
    n90 = np.searchsorted(cumvar, 0.90) + 1
    n95 = np.searchsorted(cumvar, 0.95) + 1
    pr = (eigenvalues.sum())**2 / (eigenvalues**2).sum()
    
    print(f"\n  DATA: {sum(len(v) for v in year_embeddings.values()):,} vectors "
          f"across {len(year_embeddings)} years (2017–2023)")
    
    print(f"\n  EFFECTIVE DIMENSIONALITY (combined):")
    print(f"    90% variance: {n90} components")
    print(f"    95% variance: {n95} components")
    print(f"    Participation ratio: {pr:.1f}")
    print(f"    Top PC: {var_explained[0]*100:.1f}% variance")
    
    print(f"\n  TEMPORAL STABILITY:")
    years = sorted(year_results.keys())
    prs = [year_results[y]['participation_ratio'] for y in years]
    print(f"    Participation ratio: {np.mean(prs):.1f} ± {np.std(prs):.2f}")
    for k_val in sorted(stability_df['k'].unique()):
        sub = stability_df[stability_df['k'] == k_val]
        print(f"    Subspace angles (k={k_val}): "
              f"mean max = {sub['max_angle_deg'].mean():.1f}° ± {sub['max_angle_deg'].std():.1f}°")
    
    print(f"\n  DIMENSION CLUSTERS ({len(cluster_summary)}):")
    for cid, info in cluster_summary.items():
        print(f"    Cluster {cid} [{info['physical_label']}]: "
              f"{info['n_members']} dims, |r|={info['mean_within_correlation']:.3f}")
    
    print(f"\n  TOP PCs:")
    for pc in pc_interpretations[:3]:
        top = pc['top_dimensions'][:3]
        dims_str = ', '.join([f"{d['dimension']}({d['primary_variable']})" for d in top])
        print(f"    {pc['component']}: {dims_str}")
    
    print(f"\n  OUTPUTS: {OUTPUT_DIR}/")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fp = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(fp):
            print(f"    {f} ({os.path.getsize(fp)/1024:.1f} KB)")
    
    print("\n" + "=" * 70)
    print("Complete.")
    print("-" * 60)

print_summary(eigenvalues, var_explained, cumvar, cluster_summary,
              pc_interpretations, year_results, stability_df)


def create_publication_figure(eigenvalues, var_explained, cumvar, eigenvectors,
                               year_results, linkage_mat, cluster_df, best_k):
    """
    Publication figure: Geometric Structure of the AlphaEarth Embedding Space.
    
    4-panel composite:
      (a) Cumulative variance explained (scree) — effective dimensionality
      (b) Top 2 PC eigenvector weights colored by environmental category — physical axes
      (c) Per-year eigenvalue overlay — temporal geometric stability
      (d) Dendrogram with environmental category colors — dimension cluster structure
    """
    print("\n" + "=" * 70)
    print("GENERATING PUBLICATION FIGURE (Publication figure 2)")
    print("-" * 60)
    
    # Load dimension dictionary
    p1 = load_dimension_dictionary()
    dim_to_var, dim_to_rho, dim_to_cat = {}, {}, {}
    if p1 is not None:
        _, _, lookups = p1
        dim_to_var = lookups.get('dim_to_var', {})
        dim_to_rho = lookups.get('dim_to_rho', {})
        dim_to_cat = lookups.get('dim_to_cat', {})
    
    CATEGORY_COLORS = {
        'Terrain': '#8B4513', 'Soil': '#DAA520', 'Vegetation': '#228B22',
        'Temperature': '#DC143C', 'Climate': '#4169E1', 'Hydrology': '#00CED1',
        'Urban': '#696969', 'Radiation': '#FFD700',
    }
    
    # ── Figure layout ──
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30,
                           height_ratios=[1, 1.1])
    
    x = np.arange(1, N_DIMS + 1)
    
    # (a) Cumulative variance explained
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(x, cumvar * 100, 'o-', color='steelblue', markersize=3, linewidth=1.5)
    
    for thr in [0.80, 0.90, 0.95]:
        ax_a.axhline(y=thr * 100, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
        nc = int(np.searchsorted(cumvar, thr) + 1)
        ax_a.plot(nc, thr * 100, 's', color='firebrick', markersize=7, zorder=5)
        ax_a.annotate(f'{thr*100:.0f}% → {nc} PCs',
                      xy=(nc, thr * 100), xytext=(nc + 6, thr * 100 - 2.5),
                      arrowprops=dict(arrowstyle='->', color='firebrick', lw=0.8),
                      fontsize=9, color='firebrick')
    
    # Shade effective region
    n90 = int(np.searchsorted(cumvar, 0.90) + 1)
    ax_a.axvspan(0.5, n90 + 0.5, alpha=0.07, color='steelblue')
    
    pr = (eigenvalues.sum())**2 / (eigenvalues**2).sum()
    ax_a.annotate(f'Participation ratio = {pr:.1f}',
                  xy=(0.97, 0.05), xycoords='axes fraction',
                  ha='right', fontsize=9, color='dimgray',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax_a.set_xlabel('Number of Principal Components')
    ax_a.set_ylabel('Cumulative Variance Explained (%)')
    ax_a.set_title('(a) Effective Dimensionality', fontweight='bold')
    ax_a.set_xlim(0.5, N_DIMS + 0.5)
    ax_a.set_ylim(0, 102)
    ax_a.set_xticks([1, 10, 20, 30, 40, 50, 60, 64])
    
    # (b) PC1 and PC2 eigenvector weights
    ax_b = fig.add_subplot(gs[0, 1])
    
    bar_width = 0.38
    colors = [CATEGORY_COLORS.get(dim_to_cat.get(d, ''), '#CCCCCC') for d in AE_COLS]
    
    evec1 = eigenvectors[:, 0]
    evec2 = eigenvectors[:, 1]
    
    bars1 = ax_b.bar(x - bar_width/2, evec1, bar_width, color=colors, alpha=0.85,
                      edgecolor='none', label=f'PC1 ({var_explained[0]*100:.1f}%)')
    bars2 = ax_b.bar(x + bar_width/2, evec2, bar_width, color=colors, alpha=0.45,
                      edgecolor='none', hatch='///',
                      label=f'PC2 ({var_explained[1]*100:.1f}%)')
    
    ax_b.axhline(y=0, color='black', linewidth=0.5)
    
    # Annotate top contributors for PC1
    top1 = np.argsort(np.abs(evec1))[::-1][:3]
    for idx in top1:
        dim = AE_COLS[idx]
        w = evec1[idx]
        var = dim_to_var.get(dim, '?')
        short = var[:10] if isinstance(var, str) else '?'
        ax_b.annotate(f'{dim}\n({short})',
                      xy=(idx + 1 - bar_width/2, w),
                      fontsize=6, ha='center', fontweight='bold',
                      va='bottom' if w > 0 else 'top',
                      color='black')
    
    # Annotate top contributors for PC2
    top2 = np.argsort(np.abs(evec2))[::-1][:2]
    for idx in top2:
        dim = AE_COLS[idx]
        w = evec2[idx]
        var = dim_to_var.get(dim, '?')
        short = var[:10] if isinstance(var, str) else '?'
        ax_b.annotate(f'{dim}\n({short})',
                      xy=(idx + 1 + bar_width/2, w),
                      fontsize=6, ha='center', fontstyle='italic',
                      va='bottom' if w > 0 else 'top',
                      color='dimgray')
    
    ax_b.set_xlabel('AlphaEarth Dimension Index')
    ax_b.set_ylabel('Eigenvector Weight')
    ax_b.set_title('(b) Principal Axes: Moisture–Vegetation (PC1) & Temperature (PC2)',
                    fontweight='bold')
    ax_b.set_xlim(0, N_DIMS + 1)
    ax_b.legend(fontsize=8, loc='lower right')
    
    # Category legend
    from matplotlib.patches import Patch
    cats_in_data = sorted(set(dim_to_cat.values()) & set(CATEGORY_COLORS.keys()))
    cat_legend = [Patch(facecolor=CATEGORY_COLORS[c], label=c) for c in cats_in_data]
    ax_b_leg = ax_b.legend(handles=cat_legend, fontsize=6, ncol=4,
                            loc='upper right', title='Category', title_fontsize=7)
    ax_b.add_artist(ax_b_leg)
    # Re-add PC legend
    ax_b.legend(fontsize=8, loc='lower right')
    
    # (c) Per-year eigenvalue overlay
    ax_c = fig.add_subplot(gs[1, 0])
    
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(year_results)))
    for i, (year, res) in enumerate(sorted(year_results.items())):
        ax_c.plot(x, res['var_explained'] * 100, '-', color=cmap[i],
                  linewidth=1.5, alpha=0.85, label=str(year))
    
    # Inset: participation ratio bar
    ax_inset = ax_c.inset_axes([0.55, 0.45, 0.40, 0.45])
    years_sorted = sorted(year_results.keys())
    prs = [year_results[y]['participation_ratio'] for y in years_sorted]
    ax_inset.bar(range(len(years_sorted)), prs, color=cmap, alpha=0.8, edgecolor='none')
    ax_inset.set_xticks(range(len(years_sorted)))
    ax_inset.set_xticklabels([str(y)[2:] for y in years_sorted], fontsize=7)
    ax_inset.set_ylabel('PR', fontsize=7)
    ax_inset.set_title(f'PR = {np.mean(prs):.1f} ± {np.std(prs):.2f}', fontsize=8)
    ax_inset.set_ylim(12, 13.5)
    ax_inset.tick_params(labelsize=6)
    
    ax_c.set_xlabel('Principal Component')
    ax_c.set_ylabel('Variance Explained (%)')
    ax_c.set_title('(c) Temporal Stability of Geometric Structure (2017–2023)',
                    fontweight='bold')
    ax_c.set_xlim(0.5, N_DIMS + 0.5)
    ax_c.set_xticks([1, 10, 20, 30, 40, 50, 60, 64])
    ax_c.legend(fontsize=7, ncol=4, loc='upper right')
    
    # (d) Dendrogram with environmental category colors
    ax_d = fig.add_subplot(gs[1, 1])
    
    dendro = dendrogram(
        linkage_mat, labels=AE_COLS, leaf_rotation=90, leaf_font_size=6,
        ax=ax_d, color_threshold=0,  # single color for branches
        above_threshold_color='gray',
    )
    
    # Color leaf labels by environmental category
    for lbl in ax_d.get_xticklabels():
        dim = lbl.get_text()
        cat = dim_to_cat.get(dim, 'Unknown')
        lbl.set_color(CATEGORY_COLORS.get(cat, '#AAAAAA'))
        lbl.set_fontweight('bold')
    
    ax_d.set_ylabel('Distance (1 − |r|, Ward)')
    ax_d.set_title('(d) Dimension Cluster Dendrogram', fontweight='bold')
    
    # Category legend on dendrogram
    cats_present = sorted(set(dim_to_cat.values()) & set(CATEGORY_COLORS.keys()))
    legend_elements = [Patch(facecolor=CATEGORY_COLORS[c], label=c) for c in cats_present]
    ax_d.legend(handles=legend_elements, fontsize=6, ncol=2,
                loc='upper right', title='Env. Category', title_fontsize=7)
    
    # ── Save ──
    fig.savefig(f'{FIG_DIR}/fig2_geometric_structure.png', dpi=300,
                facecolor='white', bbox_inches='tight')
    fig.savefig(f'{FIG_DIR}/fig2_geometric_structure.pdf', dpi=300,
                facecolor='white', bbox_inches='tight')
    plt.show()
    print("  ✓ Saved fig2_geometric_structure.png/pdf")

create_publication_figure(eigenvalues, var_explained, cumvar, eigenvectors,
                           year_results, linkage_mat, cluster_df, best_k)


del E, year_embeddings
gc.collect()
print("Done.")
