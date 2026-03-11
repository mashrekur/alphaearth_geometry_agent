"""
Publication Figures

Generates from manifold_results/:
  Fig 4: Local Geometry and Scale Dependence (6 panels)
  Fig 5: Regional Profiles in 3D Embedding Space (6 subplots + coherence map)

Author: Mashrekur Rahman | 2026
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, FancyArrowPatch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import pyarrow.parquet as pq
import json, os, gc, warnings
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
FIG_DIR = 'manifold_results/figures'
AE_COLS = [f'A{i:02d}' for i in range(64)]
N_DIMS = 64
CONUS_EXTENT = [-125.0, -66.5, 24.5, 49.5]
SEED = 42

REGIONS = {
    'Pacific NW':    {'lon': (-125, -116), 'lat': (42, 49), 'color': '#1B5E20'},
    'Great Plains':  {'lon': (-104, -95),  'lat': (35, 48), 'color': '#E65100'},
    'Southeast':     {'lon': (-90, -75),   'lat': (25, 36), 'color': '#4A148C'},
    'Mountain West': {'lon': (-115, -104), 'lat': (35, 45), 'color': '#01579B'},
    'Northeast':     {'lon': (-80, -67),   'lat': (39, 47), 'color': '#B71C1C'},
    'Southwest':     {'lon': (-115, -104), 'lat': (31, 37), 'color': '#FF6F00'},
}

CATEGORY_COLORS = {
    'Terrain': '#8B4513', 'Soil': '#DAA520', 'Vegetation': '#228B22',
    'Temperature': '#DC143C', 'Climate': '#4169E1', 'Hydrology': '#00CED1',
    'Urban': '#696969', 'Radiation': '#FFD700', '?': '#CCCCCC',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 12,
    'axes.linewidth': 0.8, 'axes.labelsize': 13,
    'axes.titlesize': 14, 'axes.titleweight': 'bold',
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})

os.makedirs(FIG_DIR, exist_ok=True)

# ── Load dimension dictionary ──
dd_path = '../results/dimension_dictionary.csv'
dim_to_var, dim_to_cat = {}, {}
if os.path.exists(dd_path):
    dd = pd.read_csv(dd_path)
    for c in ['sp_primary', 'primary_variable']:
        if c in dd.columns: dim_to_var = dict(zip(dd['dimension'], dd[c])); break
    for c in ['sp_category', 'category']:
        if c in dd.columns: dim_to_cat = dict(zip(dd['dimension'], dd[c])); break


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: LOCAL GEOMETRY AND SCALE DEPENDENCE
# ═══════════════════════════════════════════════════════════════════════════

def make_figure_4():
    """
    6-panel figure combining local geometry and scale dependence results.
    
    (a) CONUS: tangent space instability (curvature map)
    (b) CONUS: locally dominant category
    (c) Alignment vs scale
    (d) Alignment distribution histogram with random baseline
    (e) Tangent angle + PR vs scale (dual axis)
    (f) Dominant category stacked bar vs scale
    """
    print("\n" + "=" * 70)
    print("GENERATING FIGURE 4: LOCAL GEOMETRY & SCALE DEPENDENCE")
    print("-" * 60)
    
    # Load local PCA results
    pca_path = f'{RESULTS_DIR}/local_pca_results.csv'
    if not os.path.exists(pca_path):
        print(f"  ⚠ {pca_path} not found"); return
    pca_df = pd.read_csv(pca_path)
    
    # Load multiscale summary
    ms_path = f'{RESULTS_DIR}/multiscale_summary.csv'
    if not os.path.exists(ms_path):
        print(f"  ⚠ {ms_path} not found"); return
    ms_df = pd.read_csv(ms_path)
    
    # Load per-scale data for category breakdown
    K_SCALES = [20, 100, 500, 2000]
    scale_dfs = {}
    for k in K_SCALES:
        p = f'{RESULTS_DIR}/multiscale_k{k}.csv'
        if os.path.exists(p):
            scale_dfs[k] = pd.read_csv(p)
    
    RANDOM_BASELINE = 1.0 / np.sqrt(N_DIMS)
    ALL_CATS = ['Temperature', 'Vegetation', 'Hydrology', 'Soil', 'Terrain', 'Climate', 'Radiation', 'Urban']
    
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.32, wspace=0.28)
    
    # ── (a) CONUS: tangent instability ──
    if HAS_CARTOPY:
        ax_a = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        ax_a.set_extent(CONUS_EXTENT); ax_a.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax_a.add_feature(cfeature.STATES, linewidth=0.2, edgecolor='gray')
        transform = ccrs.PlateCarree()
    else:
        ax_a = fig.add_subplot(gs[0, 0])
        ax_a.set_xlim(CONUS_EXTENT[0], CONUS_EXTENT[1])
        ax_a.set_ylim(CONUS_EXTENT[2], CONUS_EXTENT[3]); ax_a.set_aspect('equal')
        transform = None
    
    if 'tangent_angle_deg' in pca_df.columns:
        vmin_t, vmax_t = pca_df['tangent_angle_deg'].quantile([0.02, 0.98])
        kw = dict(c=pca_df['tangent_angle_deg'], s=2, cmap='hot_r',
                  vmin=vmin_t, vmax=vmax_t, alpha=0.8, rasterized=True)
        if transform: kw['transform'] = transform
        sc_a = ax_a.scatter(pca_df['longitude'], pca_df['latitude'], **kw)
        plt.colorbar(sc_a, ax=ax_a, shrink=0.7, label='Tangent Angle (°)', pad=0.02)
    ax_a.set_title('(a) Manifold Curvature (Tangent Instability)', fontsize=12)
    
    # ── (b) CONUS: dominant category ──
    if HAS_CARTOPY:
        ax_b = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
        ax_b.set_extent(CONUS_EXTENT); ax_b.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax_b.add_feature(cfeature.STATES, linewidth=0.2, edgecolor='gray')
    else:
        ax_b = fig.add_subplot(gs[0, 1])
        ax_b.set_xlim(CONUS_EXTENT[0], CONUS_EXTENT[1])
        ax_b.set_ylim(CONUS_EXTENT[2], CONUS_EXTENT[3]); ax_b.set_aspect('equal')
    
    if 'dominant_dim_cat' in pca_df.columns:
        for cat, color in CATEGORY_COLORS.items():
            mask = pca_df['dominant_dim_cat'] == cat
            if mask.sum() == 0: continue
            sub = pca_df[mask]
            kw = dict(c=color, s=2, alpha=0.7, label=cat, rasterized=True)
            if transform: kw['transform'] = transform
            ax_b.scatter(sub['longitude'], sub['latitude'], **kw)
        ax_b.legend(fontsize=6, ncol=2, loc='lower right', markerscale=4, title_fontsize=7)
    ax_b.set_title('(b) Locally Dominant Environmental Category', fontsize=12)
    
    # ── (c) Alignment vs scale ──
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.plot(ms_df['k'], ms_df['mean_align_pc1'], 'o-', color='#E91E63',
              lw=2.5, ms=10, label='PC1 (moisture)', zorder=3)
    ax_c.plot(ms_df['k'], ms_df['mean_align_pc2'], 's--', color='#2196F3',
              lw=2, ms=9, label='PC2 (temperature)', zorder=3)
    ax_c.axhline(y=RANDOM_BASELINE, color='gray', ls='--', lw=1.2,
                 label=f'Random ({RANDOM_BASELINE:.3f})')
    ax_c.fill_between(ms_df['k'],
                      ms_df['mean_align_pc1'] - ms_df['std_align_pc1'],
                      ms_df['mean_align_pc1'] + ms_df['std_align_pc1'],
                      alpha=0.12, color='#E91E63')
    ax_c.set_xlabel('Neighborhood Size (k)'); ax_c.set_ylabel('|cos(local, global)|')
    ax_c.set_title('(c) Alignment Never Recovers', fontsize=12)
    ax_c.set_xscale('log'); ax_c.legend(fontsize=8); ax_c.grid(True, alpha=0.3)
    
    # ── (d) Alignment distribution ──
    ax_d = fig.add_subplot(gs[1, 0])
    if 'align_global_pc1' in pca_df.columns:
        ax_d.hist(pca_df['align_global_pc1'], bins=50, color='#E91E63', alpha=0.6,
                  density=True, edgecolor='none', label='PC1')
        ax_d.hist(pca_df['align_global_pc2'], bins=50, color='#2196F3', alpha=0.4,
                  density=True, edgecolor='none', label='PC2')
        ax_d.axvline(x=RANDOM_BASELINE, color='gray', ls='--', lw=1.5, label='Random')
        mean_a = pca_df['align_global_pc1'].mean()
        ax_d.axvline(x=mean_a, color='#E91E63', ls='-', lw=2, alpha=0.7,
                     label=f'PC1 mean = {mean_a:.3f}')
    ax_d.set_xlabel('|cos(local PC, global PC)|'); ax_d.set_ylabel('Density')
    ax_d.set_title('(d) Local–Global Alignment Distribution', fontsize=12)
    ax_d.legend(fontsize=8)
    
    # ── (e) Tangent + PR vs scale ──
    ax_e = fig.add_subplot(gs[1, 1])
    ax_e2 = ax_e.twinx()
    l1 = ax_e.plot(ms_df['k'], ms_df['mean_tangent_angle'], 'o-', color='#D32F2F',
                   lw=2.5, ms=10, label='Tangent angle')
    ax_e.fill_between(ms_df['k'],
                      ms_df['mean_tangent_angle'] - ms_df['std_tangent_angle'],
                      ms_df['mean_tangent_angle'] + ms_df['std_tangent_angle'],
                      alpha=0.1, color='#D32F2F')
    ax_e.set_ylabel('Tangent Angle (°)', color='#D32F2F')
    l2 = ax_e2.plot(ms_df['k'], ms_df['mean_local_pr'], 's--', color='#1565C0',
                    lw=2, ms=9, label='Local PR')
    ax_e2.set_ylabel('Participation Ratio', color='#1565C0')
    lines = l1 + l2
    ax_e.legend(lines, [l.get_label() for l in lines], fontsize=8, loc='center right')
    ax_e.set_xlabel('Neighborhood Size (k)')
    ax_e.set_title('(e) Curvature and Complexity vs. Scale', fontsize=12)
    ax_e.set_xscale('log'); ax_e.grid(True, alpha=0.3)
    
    # ── (f) Category vs scale ──
    ax_f = fig.add_subplot(gs[1, 2])
    if scale_dfs:
        bottoms = np.zeros(len(K_SCALES))
        for cat in ALL_CATS:
            pcts = []
            for k in K_SCALES:
                if k in scale_dfs:
                    vc = scale_dfs[k]['dominant_cat'].value_counts(normalize=True)
                    pcts.append(vc.get(cat, 0) * 100)
                else:
                    pcts.append(0)
            pcts = np.array(pcts)
            ax_f.bar([str(k) for k in K_SCALES], pcts, bottom=bottoms,
                     color=CATEGORY_COLORS.get(cat, '#CCC'), label=cat, edgecolor='none')
            bottoms += pcts
        ax_f.legend(fontsize=6, ncol=2, loc='upper right')
    ax_f.set_xlabel('Neighborhood Size (k)'); ax_f.set_ylabel('(%)')
    ax_f.set_title('(f) Dominant Category vs. Scale', fontsize=12)
    ax_f.set_ylim(0, 105)
    
    fig.savefig(f'{FIG_DIR}/fig4_local_geometry_scale.png', dpi=300, facecolor='white', bbox_inches='tight')
    fig.savefig(f'{FIG_DIR}/fig4_local_geometry_scale.pdf', dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Figure 4 saved")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5: REGIONAL 3D PROFILES + RETRIEVAL COHERENCE
# ═══════════════════════════════════════════════════════════════════════════

def make_figure_5():
    """
    7-panel figure:
      Top: CONUS coherence map with region boxes
      Bottom: 6 regional 3D manifold views (one per region)
    """
    print("\n" + "=" * 70)
    print("GENERATING FIGURE 5: REGIONAL PROFILES")
    print("-" * 60)
    
    # Load coherence results
    coh_path = f'{RESULTS_DIR}/retrieval_coherence.csv'
    if not os.path.exists(coh_path):
        print(f"  ⚠ {coh_path} not found"); return
    coh_df = pd.read_csv(coh_path)
    
    # Load data for 3D projections
    print("  Loading embedding data for 3D projections...")
    YEARS = list(range(2017, 2024))
    per_year = 50_000 // len(YEARS)
    rng = np.random.default_rng(SEED)
    
    frames = []
    for year in YEARS:
        fp = f'{DATA_DIR}/conus_{year}_unified.parquet'
        if not os.path.exists(fp): continue
        cols_avail = pq.read_schema(fp).names
        use_cols = [c for c in ['longitude', 'latitude', 'elevation'] + AE_COLS if c in cols_avail]
        df_y = pd.read_parquet(fp, columns=use_cols)
        idx = rng.choice(len(df_y), size=min(per_year, len(df_y)), replace=False)
        frames.append(df_y.iloc[idx].reset_index(drop=True))
        del df_y
    
    df_3d = pd.concat(frames, ignore_index=True)
    E_3d = df_3d[AE_COLS].values.astype(np.float64)
    coords_3d = df_3d[['longitude', 'latitude']].values
    
    # Global PCA
    pca = PCA(n_components=3)
    E_pca = pca.fit_transform(E_3d)
    var_pct = pca.explained_variance_ratio_ * 100
    print(f"  PCA: {var_pct[:3]}")
    
    # Assign region labels
    df_3d['region'] = 'Other'
    for rname, bounds in REGIONS.items():
        mask = ((df_3d['longitude'] >= bounds['lon'][0]) & (df_3d['longitude'] <= bounds['lon'][1]) &
                (df_3d['latitude'] >= bounds['lat'][0]) & (df_3d['latitude'] <= bounds['lat'][1]))
        df_3d.loc[mask, 'region'] = rname
    
    # ── Figure layout: 1 wide map on top, 6 3D plots below (2 rows × 3 cols) ──
    fig = plt.figure(figsize=(20, 18))
    gs = gridspec.GridSpec(3, 3, hspace=0.30, wspace=0.22, height_ratios=[1, 1, 1])
    
    # ── (top) CONUS coherence map spanning 3 columns ──
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0, :])
    if HAS_CARTOPY:
        ax_map = fig.add_subplot(gs_top[0], projection=ccrs.PlateCarree())
        ax_map.set_extent(CONUS_EXTENT)
        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.7, color='black')
        ax_map.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='gray')
        ax_map.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
        transform = ccrs.PlateCarree()
    else:
        ax_map = fig.add_subplot(gs_top[0])
        ax_map.set_xlim(CONUS_EXTENT[0], CONUS_EXTENT[1])
        ax_map.set_ylim(CONUS_EXTENT[2], CONUS_EXTENT[3])
        ax_map.set_aspect('equal')
        transform = None
    
    vmin_c, vmax_c = coh_df['retrieval_coherence'].quantile([0.02, 0.98])
    kw = dict(c=coh_df['retrieval_coherence'], s=3, cmap='RdYlGn_r',
              vmin=vmin_c, vmax=vmax_c, alpha=0.8, rasterized=True)
    if transform: kw['transform'] = transform
    sc = ax_map.scatter(coh_df['longitude'], coh_df['latitude'], **kw)
    plt.colorbar(sc, ax=ax_map, shrink=0.6, label='Retrieval Spread (σ-norm.)', pad=0.02,
                 orientation='horizontal', aspect=40)
    
    # Draw region boxes with labels
    for rname, bounds in REGIONS.items():
        rect_kw = dict(linewidth=2, edgecolor=bounds['color'], facecolor='none', linestyle='-')
        if transform: rect_kw['transform'] = transform
        rect = plt.Rectangle((bounds['lon'][0], bounds['lat'][0]),
                              bounds['lon'][1] - bounds['lon'][0],
                              bounds['lat'][1] - bounds['lat'][0], **rect_kw)
        ax_map.add_patch(rect)
        cx = (bounds['lon'][0] + bounds['lon'][1]) / 2
        cy = bounds['lat'][1] + 0.8
        txt_kw = dict(fontsize=9, ha='center', fontweight='bold', color=bounds['color'])
        if transform: txt_kw['transform'] = transform
        ax_map.text(cx, cy, rname, **txt_kw,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor=bounds['color']))
    
    ax_map.set_title('(a) FAISS Retrieval Coherence and Regional Analysis Zones', fontweight='bold', fontsize=14)
    
    # ── 6 regional 3D panels ──
    region_list = list(REGIONS.keys())
    
    # Load regional dimension importance
    dict_path = f'{RESULTS_DIR}/enhanced_geo_dictionary.json'
    region_info = {}
    if os.path.exists(dict_path):
        with open(dict_path) as f:
            geo_dict = json.load(f)
        region_info = geo_dict.get('regional_profiles', {})
    
    for idx, rname in enumerate(region_list):
        row = idx // 3 + 1  # rows 1 and 2
        col = idx % 3
        
        ax = fig.add_subplot(gs[row, col], projection='3d')
        bounds = REGIONS[rname]
        rcolor = bounds['color']
        
        # Plot all points faintly
        ax.scatter(E_pca[:, 0], E_pca[:, 1], E_pca[:, 2],
                   s=0.1, alpha=0.03, color='gray', rasterized=True)
        
        # Highlight this region
        mask = df_3d['region'] == rname
        if mask.sum() > 0:
            ax.scatter(E_pca[mask, 0], E_pca[mask, 1], E_pca[mask, 2],
                       s=2, alpha=0.6, color=rcolor, rasterized=True)
        
        ax.set_xlabel(f'PC1', fontsize=7)
        ax.set_ylabel(f'PC2', fontsize=7)
        ax.set_zlabel(f'PC3', fontsize=7)
        ax.view_init(elev=20, azim=135)
        ax.tick_params(labelsize=5)
        
        # Annotation with region stats
        rdata = region_info.get(rname, {})
        coh_val = rdata.get('mean_coherence', '?')
        id_val = rdata.get('mean_local_id', '?')
        top_dims = rdata.get('top10', [])
        if top_dims:
            top_str = ', '.join([f"{d['dimension']}({d['variable'][:6]})" for d in top_dims[:3]])
        else:
            top_str = '?'
        
        coh_str = f'{coh_val:.3f}' if isinstance(coh_val, float) else str(coh_val)
        id_str = f'{id_val:.1f}' if isinstance(id_val, float) else str(id_val)
        
        panel_label = chr(ord('b') + idx)
        ax.set_title(f'({panel_label}) {rname}\ncoh={coh_str}, ID={id_str}\n{top_str}',
                     fontsize=9, fontweight='bold', color=rcolor, pad=5)
    
    fig.savefig(f'{FIG_DIR}/fig5_regional_profiles.png', dpi=300, facecolor='white', bbox_inches='tight')
    fig.savefig(f'{FIG_DIR}/fig5_regional_profiles.pdf', dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Figure 5 saved")
    
    del E_3d, E_pca, df_3d
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY FIGURE: INTUITIVE VISUAL ABSTRACT
# ═══════════════════════════════════════════════════════════════════════════

def make_summary_figure():
    """
    Visual abstract summarizing the geometric characterization:
      1. Global: "64-D space is ~13-dimensional"
      2. Local: "Manifold is curved, directions rotate"  
      3. Scale: "Alignment never recovers"
      4. Operational: "Geometry predicts retrieval quality"
    """
    print("\n" + "=" * 70)
    print("GENERATING SUMMARY FIGURE (Visual Abstract)")
    print("-" * 60)
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    # ── Panel 1: Global structure ──
    ax = axes[0]
    # Stylized scree curve
    x = np.arange(1, 65)
    y = 18 * np.exp(-0.15 * (x - 1)) + 0.5
    cumvar = np.cumsum(y) / np.sum(y) * 100
    
    ax.fill_between(x, 0, y, color='#2196F3', alpha=0.3, edgecolor='none')
    ax.plot(x, y, color='#1565C0', lw=2)
    ax.axvline(x=13.3, color='#E91E63', lw=2, ls='--')
    ax.annotate('PR = 13.3', xy=(13.3, 12), fontsize=11, color='#E91E63',
                fontweight='bold', ha='left')
    
    ax.annotate('PC1: Moisture–Vegetation\n(17.6%)', xy=(3, 16),
                fontsize=9, color='#1565C0', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#E3F2FD', edgecolor='#1565C0'))
    ax.annotate('PC2: Temperature\n(11.5%)', xy=(3, 11),
                fontsize=9, color='#1565C0',
                bbox=dict(boxstyle='round', facecolor='#E3F2FD', edgecolor='#1565C0'))
    
    ax.set_xlim(0, 65); ax.set_ylim(0, 22)
    ax.set_xlabel('Principal Component'); ax.set_ylabel('Variance (%)')
    ax.set_title('① GLOBAL\n64-D → ~13 effective dims\nPhysically interpretable axes',
                 fontweight='bold', fontsize=12, color='#0D47A1')
    ax.tick_params(labelsize=8)
    
    # ── Panel 2: Local curvature ──
    ax = axes[1]
    
    # Draw a curved manifold with tangent vectors
    t = np.linspace(0, 2 * np.pi, 200)
    x_manifold = t + 0.3 * np.sin(3 * t)
    y_manifold = np.sin(t) + 0.2 * np.cos(2 * t)
    
    ax.plot(x_manifold, y_manifold, color='#2196F3', lw=3, alpha=0.5)
    ax.fill_between(x_manifold, y_manifold - 0.1, y_manifold + 0.1,
                    color='#2196F3', alpha=0.1)
    
    # Tangent arrows at different points (showing rotation)
    arrow_points = [30, 80, 130, 170]
    arrow_colors = ['#4CAF50', '#FF9800', '#E91E63', '#9C27B0']
    for pi, ac in zip(arrow_points, arrow_colors):
        dx = x_manifold[pi + 1] - x_manifold[pi]
        dy = y_manifold[pi + 1] - y_manifold[pi]
        norm = np.sqrt(dx**2 + dy**2)
        dx, dy = dx / norm * 0.5, dy / norm * 0.5
        ax.annotate('', xy=(x_manifold[pi] + dx, y_manifold[pi] + dy),
                    xytext=(x_manifold[pi], y_manifold[pi]),
                    arrowprops=dict(arrowstyle='->', color=ac, lw=2.5))
    
    # Key numbers
    ax.text(3.2, -0.8, 'MLE ID ≈ 10\nTangent angle: 69°\n84% locations curved',
            fontsize=10, fontweight='bold', color='#BF360C',
            bbox=dict(boxstyle='round', facecolor='#FBE9E7', edgecolor='#BF360C'))
    
    ax.set_xlim(-0.5, 7); ax.set_ylim(-1.5, 1.8)
    ax.set_title('② LOCAL\nManifold is curved\nDirections rotate everywhere',
                 fontweight='bold', fontsize=12, color='#BF360C')
    ax.set_aspect('equal')
    ax.tick_params(labelbottom=False, labelleft=False)
    
    # ── Panel 3: Scale dependence ──
    ax = axes[2]
    
    k_vals = [20, 100, 500, 2000]
    align_vals = [0.150, 0.169, 0.189, 0.210]
    random_bl = 0.125
    
    ax.plot(k_vals, align_vals, 'o-', color='#E91E63', lw=3, ms=12, zorder=3)
    ax.axhline(y=random_bl, color='gray', ls='--', lw=2, label='Random baseline')
    ax.fill_between(k_vals, random_bl, align_vals, alpha=0.15, color='#E91E63')
    
    ax.annotate('Never exceeds\n0.3 threshold', xy=(1000, 0.20),
                fontsize=11, fontweight='bold', color='#880E4F',
                bbox=dict(boxstyle='round', facecolor='#FCE4EC', edgecolor='#880E4F'))
    
    ax.set_xscale('log')
    ax.set_xlabel('Neighborhood Size (k)'); ax.set_ylabel('Alignment')
    ax.set_title('③ SCALE\nAlignment never recovers\nGlobal directions are local noise',
                 fontweight='bold', fontsize=12, color='#880E4F')
    ax.set_ylim(0.05, 0.35)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    
    # ── Panel 4: Operational ──
    ax = axes[3]
    
    # Stylized calibration curve
    pred = np.array([0.097, 0.141, 0.157, 0.168, 0.177, 0.186, 0.195, 0.207, 0.221, 0.254])
    actual = np.array([0.084, 0.158, 0.165, 0.171, 0.177, 0.186, 0.196, 0.200, 0.218, 0.248])
    
    ax.plot(pred, actual, 'o-', color='#4CAF50', lw=2.5, ms=8, label='Calibration')
    ax.plot([0.05, 0.30], [0.05, 0.30], 'k--', lw=1, alpha=0.5)
    ax.fill_between(pred, pred, actual, alpha=0.15, color='#4CAF50')
    
    ax.annotate('R² = 0.32\nGeometry predicts\nretrieval quality', xy=(0.08, 0.22),
                fontsize=10, fontweight='bold', color='#1B5E20',
                bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='#1B5E20'))
    
    ax.set_xlabel('Predicted Spread'); ax.set_ylabel('Observed Spread')
    ax.set_title('④ OPERATIONAL\nConfidence model works\nRegional dictionary deployed',
                 fontweight='bold', fontsize=12, color='#1B5E20')
    ax.grid(True, alpha=0.3)
    
    # ── Connecting arrows between panels ──
    for i in range(3):
        fig.text((i + 1) * 0.25 + 0.005, 0.5, '→', fontsize=30, color='gray',
                 ha='center', va='center', fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(f'{FIG_DIR}/fig_visual_abstract.png', dpi=300, facecolor='white', bbox_inches='tight')
    fig.savefig(f'{FIG_DIR}/fig_visual_abstract.pdf', dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Visual abstract saved")


# ═══════════════════════════════════════════════════════════════════════════
# RUN ALL
# ═══════════════════════════════════════════════════════════════════════════

make_figure_4()
make_figure_5()
make_summary_figure()

print("\n" + "=" * 70)
print("ALL PUBLICATION FIGURES GENERATED")
print("-" * 60)
print(f"\n  Figures:")
print(f"    fig2_geometric_structure.png  — covariance, PCA, clusters")
print(f"    fig3_intrinsic_dimensionality.png — MLE intrinsic dim")
print(f"    fig4_local_geometry_scale.png — local PCA, multiscale")
print(f"    fig5_regional_profiles.png   — regional geometry, coherence")
print(f"    fig_visual_abstract.png      — summary")
print("-" * 60)
