import matplotlib.pyplot as plt
import numpy as np

def create_fft_spectrum():
    # Create a simple FFT magnitude spectrum
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Simulate FFT magnitude (low frequencies in center)
    size = 100
    center = size // 2
    radius = 10
    
    # Create a 2D array representing FFT magnitude
    fft_mag = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i-center)**2 + (j-center)**2)
            if dist < radius:
                fft_mag[i,j] = np.exp(-dist/5)
    
    im = ax.imshow(fft_mag, cmap='viridis', extent=[-1, 1, -1, 1])
    ax.set_title('FFT Magnitude Spectrum\n(Energy Concentrated in Low Frequencies)', 
                fontweight='bold', fontsize=12)
    ax.set_xlabel('Frequency u')
    ax.set_ylabel('Frequency v')
    ax.grid(False)
    
    # Add a circle to highlight low frequencies
    circle = plt.Circle((0, 0), radius/size*2, fill=False, color='red', linewidth=2)
    ax.add_artist(circle)
    ax.text(0, 0.3, 'Low Frequencies\n(Preserved)', ha='center', va='center', 
           color='red', fontweight='bold', fontsize=10)
    
    plt.colorbar(im, ax=ax, label='Magnitude')
    plt.tight_layout()
    plt.savefig('fourier_spectrum.png', dpi=150, bbox_inches='tight')
    plt.show()

create_fft_spectrum()