import matplotlib.pyplot as plt
import numpy as np

def create_clean_square_wavelet():
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Title
    #ax.text(0.5, 0.95, 'Wavelet Transform', ha='center', va='center', 
    #        fontsize=18, fontweight='bold')
    
    # Main concept box
    ax.text(0.5, 0.82, 'Multi-Resolution Analysis', ha='center', va='center', fontsize=14,
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", edgecolor="black"))
    
    # Multi-level decomposition visualization
    levels = [
        (0.5, 0.7, 'Level 1\nCoarse', 0.4, 'lightblue'),
        (0.5, 0.55, 'Level 2\nMedium', 0.3, 'lightgreen'), 
        (0.5, 0.4, 'Level 3\nFine', 0.2, 'lightcoral')
    ]
    
    for x, y, text, width, color in levels:
        # Draw rectangle for each level
        rect = plt.Rectangle((x-width/2, y-0.05), width, 0.1, 
                           fill=True, facecolor=color, alpha=0.7)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Connect levels with arrows
    ax.annotate('', xy=(0.5, 0.65), xytext=(0.5, 0.6),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(0.5, 0.5), xytext=(0.5, 0.45),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Decomposition types
    ax.text(0.3, 0.25, 'Approximation\n(Low Freq)', ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    ax.text(0.7, 0.25, 'Details\n(High Freq)', ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    
    # Compression method
    ax.text(0.5, 0.15, 'Compression Method:', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(0.5, 0.08, 'Threshold small coefficients\nPreserve edges and details', 
           ha='center', va='center', fontsize=12,
           bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow"))
    
    plt.tight_layout()
    plt.savefig('wavelet_diagram_square.png', dpi=150, bbox_inches='tight')
    plt.show()

create_clean_square_wavelet()