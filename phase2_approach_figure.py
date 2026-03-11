"""
Conceptual figure: global vs. location-aware arithmetic on a curved manifold.

Two-panel illustration showing why naive global shifts land off-manifold while
local PCA-derived shifts follow the manifold curvature.

Author: Mashrekur Rahman | 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import os

FIG_DIR = 'manifold_results/figures'
os.makedirs(FIG_DIR, exist_ok=True)


def draw_curved_manifold(ax, x_range, y_offset=0, color='#2196F3', label=None):
    """Draw a curved 1D manifold (a wiggly band) in 2D."""
    t = np.linspace(x_range[0], x_range[1], 500)
    # Curved manifold path
    y_center = 0.3 * np.sin(1.5 * t) + 0.15 * np.cos(3 * t) + y_offset
    width = 0.08
    
    ax.fill_between(t, y_center - width, y_center + width,
                    color=color, alpha=0.15, edgecolor='none')
    ax.plot(t, y_center, color=color, lw=2, alpha=0.6, label=label)
    return t, y_center


def make_figure():
    fig, (ax_L, ax_R) = plt.subplots(1, 2, figsize=(18, 8))
    
    # LEFT PANEL: Naive Global Approach
    ax = ax_L
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-1.5, 2.0)
    ax.set_aspect('equal')
    
    # Draw manifold
    t = np.linspace(-3.2, 3.2, 500)
    y_manifold = 0.35 * np.sin(1.2 * t) + 0.2 * np.cos(2.5 * t)
    width = 0.1
    ax.fill_between(t, y_manifold - width, y_manifold + width,
                    color='#2196F3', alpha=0.12, edgecolor='none')
    ax.plot(t, y_manifold, color='#2196F3', lw=2.5, alpha=0.5)
    ax.text(2.5, y_manifold[-30] - 0.35, 'Manifold', color='#1565C0',
            fontsize=11, fontstyle='italic')
    
    # Source point (on manifold)
    src_t_idx = 150
    src_x = t[src_t_idx]
    src_y = y_manifold[src_t_idx]
    ax.plot(src_x, src_y, 'o', color='#E91E63', ms=12, zorder=10)
    ax.annotate('Source\n(Denver)', xy=(src_x, src_y),
                xytext=(src_x - 0.9, src_y + 0.7),
                fontsize=10, fontweight='bold', color='#E91E63',
                arrowprops=dict(arrowstyle='->', color='#E91E63', lw=1.5))
    
    # Global direction arrow (doesn't follow manifold)
    global_dx = 1.8
    global_dy = 0.9  # points UP and right — off manifold
    ax.annotate('', xy=(src_x + global_dx, src_y + global_dy),
                xytext=(src_x, src_y),
                arrowprops=dict(arrowstyle='->', color='#FF5722', lw=3, ls='-'))
    ax.text(src_x + global_dx/2 - 0.3, src_y + global_dy/2 + 0.2,
            'Global\n"moisture"\ndirection', fontsize=9, color='#FF5722',
            fontweight='bold', ha='center')
    
    # Shifted point (OFF manifold)
    shifted_x = src_x + global_dx
    shifted_y = src_y + global_dy
    ax.plot(shifted_x, shifted_y, 'X', color='#FF5722', ms=14, zorder=10)
    ax.annotate('Shifted\n(off-manifold!)', xy=(shifted_x, shifted_y),
                xytext=(shifted_x + 0.5, shifted_y + 0.4),
                fontsize=9, color='#FF5722', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#FF5722', lw=1))
    
    # FAISS retrieval (nearest point on manifold — but wrong place)
    # Find closest manifold point to shifted location
    dists_to_manifold = np.sqrt((t - shifted_x)**2 + (y_manifold - shifted_y)**2)
    nearest_idx = np.argmin(dists_to_manifold)
    faiss_x = t[nearest_idx]
    faiss_y = y_manifold[nearest_idx]
    
    ax.plot(faiss_x, faiss_y, 's', color='#9C27B0', ms=11, zorder=10)
    ax.plot([shifted_x, faiss_x], [shifted_y, faiss_y], '--', color='#9C27B0', lw=1.5, alpha=0.6)
    ax.annotate('FAISS retrieval\n(wrong analog!)', xy=(faiss_x, faiss_y),
                xytext=(faiss_x + 0.3, faiss_y - 0.6),
                fontsize=9, color='#9C27B0', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#9C27B0', lw=1))
    
    # Red X for failure
    ax.text(0, 1.6, '✗ Off-manifold shift', fontsize=14, color='#D32F2F',
            fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFCDD2', edgecolor='#D32F2F'))
    
    ax.set_title('(a) Naive Global Approach', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Embedding dimension 1', fontsize=11)
    ax.set_ylabel('Embedding dimension 2', fontsize=11)
    ax.tick_params(labelbottom=False, labelleft=False)
    
    # RIGHT PANEL: Location-Aware Approach
    ax = ax_R
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-1.5, 2.0)
    ax.set_aspect('equal')
    
    # Same manifold
    ax.fill_between(t, y_manifold - width, y_manifold + width,
                    color='#2196F3', alpha=0.12, edgecolor='none')
    ax.plot(t, y_manifold, color='#2196F3', lw=2.5, alpha=0.5)
    ax.text(2.5, y_manifold[-30] - 0.35, 'Manifold', color='#1565C0',
            fontsize=11, fontstyle='italic')
    
    # Source point (same location)
    ax.plot(src_x, src_y, 'o', color='#E91E63', ms=12, zorder=10)
    ax.annotate('Source\n(Denver)', xy=(src_x, src_y),
                xytext=(src_x - 0.9, src_y + 0.7),
                fontsize=10, fontweight='bold', color='#E91E63',
                arrowprops=dict(arrowstyle='->', color='#E91E63', lw=1.5))
    
    # Local tangent direction (follows manifold curve)
    # Compute local tangent at source point
    dt = t[src_t_idx + 1] - t[src_t_idx]
    dy_dt = (y_manifold[src_t_idx + 1] - y_manifold[src_t_idx]) / dt
    tangent = np.array([1, dy_dt])
    tangent = tangent / np.linalg.norm(tangent)
    
    # Show local PCA axes at source
    normal = np.array([-tangent[1], tangent[0]])
    
    # Draw local coordinate frame (faint)
    scale = 0.6
    ax.annotate('', xy=(src_x + tangent[0]*scale, src_y + tangent[1]*scale),
                xytext=(src_x, src_y),
                arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5, ls='--', alpha=0.5))
    ax.text(src_x + tangent[0]*scale + 0.15, src_y + tangent[1]*scale + 0.1,
            'Local PC1', fontsize=8, color='#4CAF50', alpha=0.7)
    
    # Shift along manifold (local direction)
    shift_steps = 80
    target_idx = src_t_idx + shift_steps
    target_x = t[target_idx]
    target_y = y_manifold[target_idx]
    
    # Draw curved arrow along manifold
    path_t = t[src_t_idx:target_idx+1]
    path_y = y_manifold[src_t_idx:target_idx+1]
    ax.plot(path_t, path_y, color='#4CAF50', lw=3, alpha=0.7, zorder=5)
    ax.annotate('', xy=(target_x, target_y),
                xytext=(path_t[-5], path_y[-5]),
                arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=3))
    ax.text((src_x + target_x)/2, max(path_y) + 0.25,
            'Local\n"moisture"\ndirection', fontsize=9, color='#4CAF50',
            fontweight='bold', ha='center')
    
    # Target point (ON manifold)
    ax.plot(target_x, target_y, 's', color='#4CAF50', ms=12, zorder=10)
    ax.annotate('Retrieved analog\n(Seattle-like climate)', xy=(target_x, target_y),
                xytext=(target_x + 0.4, target_y - 0.65),
                fontsize=9, color='#4CAF50', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1))
    
    # Green check for success
    ax.text(0, 1.6, '✓ On-manifold shift', fontsize=14, color='#2E7D32',
            fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#C8E6C9', edgecolor='#2E7D32'))
    
    ax.set_title('(b) Location-Aware Approach', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Embedding dimension 1', fontsize=11)
    ax.set_ylabel('Embedding dimension 2', fontsize=11)
    ax.tick_params(labelbottom=False, labelleft=False)
    
    # ── Shared annotation below ──
    fig.text(0.5, -0.02,
             'Local principal directions differ from global directions at 84% of locations.\n'
             'Alignment with global PC1 = 0.17 (random baseline = 0.125). Tangent angles = 69° mean.\n'
             'Compositional arithmetic must follow the local manifold geometry.',
             ha='center', fontsize=11, color='#424242',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5', edgecolor='#BDBDBD'))
    
    plt.tight_layout()
    fig.savefig(f'{FIG_DIR}/fig_arithmetic_approach.png', dpi=300, facecolor='white', bbox_inches='tight')
    fig.savefig(f'{FIG_DIR}/fig_arithmetic_approach.pdf', dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print("✓ Saved fig_arithmetic_approach.png/pdf")

make_figure()
