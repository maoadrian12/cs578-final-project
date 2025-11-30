import matplotlib.pyplot as plt
import numpy as np

def create_clean_square_fourier():
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Title
    #ax.text(0.5, 0.95, 'Fourier Transform', ha='center', va='center', 
    #        fontsize=18, fontweight='bold')
    
    # Main concept box
    ax.text(0.5, 0.8, 'Spatial → Frequency', ha='center', va='center', fontsize=14,
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", edgecolor="black"))
    
    # Left: Spatial domain representation
    spatial_box = plt.Rectangle((0.1, 0.4), 0.35, 0.3, fill=True, facecolor='lightblue', alpha=0.5)
    ax.add_patch(spatial_box)
    ax.text(0.275, 0.55, 'Spatial\nDomain', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Arrow
    ax.annotate('', xy=(0.55, 0.55), xytext=(0.45, 0.55),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    ax.text(0.5, 0.6, 'FFT', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Right: Frequency domain representation  
    freq_box = plt.Rectangle((0.55, 0.4), 0.35, 0.3, fill=True, facecolor='lightgreen', alpha=0.5)
    ax.add_patch(freq_box)
    ax.text(0.725, 0.55, 'Frequency\nDomain', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw a simple frequency spectrum inside the right box
    x_freq = np.linspace(0.6, 0.9, 50)
    y_freq = 0.45 + 0.2 * np.exp(-(x_freq-0.75)**2/0.005)  # Gaussian shape
    ax.plot(x_freq, y_freq, 'r-', linewidth=3)
    ax.fill_between(x_freq, 0.45, y_freq, color='red', alpha=0.3)
    
    # Compression method
    ax.text(0.5, 0.25, 'Compression Method:', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(0.5, 0.18, 'Keep low frequencies\nDiscard high frequencies', 
           ha='center', va='center', fontsize=12,
           bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow"))
    
    # Key advantage
    ax.text(0.5, 0.08, '✓ Fast computation\n✓ Good for smooth images', 
           ha='center', va='center', fontsize=11, color='darkgreen')
    
    plt.tight_layout()
    plt.savefig('fourier_diagram_square.png', dpi=150, bbox_inches='tight')
    plt.show()

create_clean_square_fourier()