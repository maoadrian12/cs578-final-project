import matplotlib.pyplot as plt
import numpy as np

def create_wavelet_decomposition():
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Draw a wavelet decomposition tree
    ax.text(0.5, 0.9, 'Original Image', ha='center', va='center', fontsize=12,
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
    
    # Level 1 decomposition
    ax.text(0.25, 0.7, 'Approx (LL)', ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    ax.text(0.5, 0.7, 'Horizontal (LH)', ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"))
    ax.text(0.75, 0.7, 'Vertical (HL)', ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="orange"))
    
    # Connect lines
    ax.plot([0.5, 0.25], [0.85, 0.75], 'k-', linewidth=2)
    ax.plot([0.5, 0.5], [0.85, 0.75], 'k-', linewidth=2)
    ax.plot([0.5, 0.75], [0.85, 0.75], 'k-', linewidth=2)
    
    # Level 2 decomposition (from LL)
    ax.text(0.125, 0.5, 'LL2', ha='center', va='center', fontsize=9,
           bbox=dict(boxstyle="round,pad=0.2", facecolor="palegreen"))
    ax.text(0.25, 0.5, 'LH2', ha='center', va='center', fontsize=9,
           bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow"))
    ax.text(0.375, 0.5, 'HL2', ha='center', va='center', fontsize=9,
           bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat"))
    
    # Connect level 2
    ax.plot([0.25, 0.125], [0.65, 0.55], 'k-', linewidth=1)
    ax.plot([0.25, 0.25], [0.65, 0.55], 'k-', linewidth=1)
    ax.plot([0.25, 0.375], [0.65, 0.55], 'k-', linewidth=1)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0.4, 1)
    ax.set_title('Wavelet Decomposition Tree', fontweight='bold', fontsize=14)
    ax.axis('off')
    
    # Add legend
    ax.text(0.1, 0.4, 'LL: Low-Low (Approximation)', fontsize=8, color='green')
    ax.text(0.1, 0.37, 'LH: Low-High (Horizontal Details)', fontsize=8, color='orange')
    ax.text(0.1, 0.34, 'HL: High-Low (Vertical Details)', fontsize=8, color='red')
    
    plt.tight_layout()
    plt.savefig('wavelet_decomposition.png', dpi=150, bbox_inches='tight')
    plt.show()

create_wavelet_decomposition()