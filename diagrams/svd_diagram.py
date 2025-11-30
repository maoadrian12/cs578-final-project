import matplotlib.pyplot as plt
import numpy as np

def create_clean_square_svd():
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Main equation
    ax.text(0.5, 0.9, 'A = U · Σ · Vᵀ', ha='center', va='center', fontsize=16,
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
    
    # Matrix boxes
    matrices = [
        (0.25, 0.75, 'U\nPatterns\nm × k', 'lightgreen'),
        (0.5, 0.75, 'Σ\nWeights\nk × k', 'gold'),
        (0.75, 0.75, 'Vᵀ\nPatterns\nk × n', 'lightcoral')
    ]
    
    for x, y, text, color in matrices:
        ax.text(x, y, text, ha='center', va='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.4", facecolor=color))
    
    # Multiplication dots
    ax.text(0.375, 0.75, '·', ha='center', va='center', fontsize=20)
    ax.text(0.625, 0.75, '·', ha='center', va='center', fontsize=20)
    
    # Compression visualization
    ax.text(0.5, 0.6, 'Keep top k singular values:', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Singular values bars - FIXED: Left = Keep, Right = Discard
    bar_bottom = 0.45
    for i in range(10):
        x_pos = 0.2 + i * 0.06
        height = 0.3 - (i * 0.02)  # Left bars are taller (more important)
        color = 'red' if i < 3 else 'lightgray'  # First 3 bars are red (keep)
        ax.bar(x_pos, height * 0.2, width=0.04, bottom=bar_bottom, color=color, alpha=0.7)
    
    # FIXED: Swapped the labels!
    ax.text(0.2, bar_bottom - 0.05, 'Keep', ha='center', va='center', fontsize=10, color='red', fontweight='bold')
    ax.text(0.8, bar_bottom - 0.05, 'Discard', ha='center', va='center', fontsize=10, color='gray')
    
    # Compression result
    ax.text(0.5, 0.3, 'Compressed: A ≈ Uₖ · Σₖ · Vₖᵀ', ha='center', va='center', fontsize=12,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    ax.text(0.5, 0.22, 'Size: k×(m+n+1) values', ha='center', va='center', fontsize=11)
    
    # Key advantage
    ax.text(0.5, 0.12, '✓ Preserves important patterns\n✓ Progressive quality', 
           ha='center', va='center', fontsize=11, color='darkgreen')
    
    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(top=0.95, bottom=0.05)
    plt.savefig('svd_diagram_square.png', dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.show()

create_clean_square_svd()