import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from sklearn.datasets import load_sample_image
from skimage import data
import os
from PIL import Image

def fourier_circular_compress(image, keep_fraction=0.1):
    """
    Compress image using Fourier Transform with circular mask (better quality)
    """
    fft_original = fft2(image)
    fft_shifted = fftshift(fft_original)
    
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create circular mask for frequency selection (better than square)
    y, x = np.ogrid[:rows, :cols]
    mask = (x - ccol)**2 + (y - crow)**2 <= (min(rows, cols) * np.sqrt(keep_fraction) / 2)**2
    
    # Apply mask and reconstruct
    fft_compressed = fft_shifted * mask
    fft_ishifted = ifftshift(fft_compressed)
    image_compressed = np.real(ifft2(fft_ishifted))
    image_compressed = np.clip(image_compressed, 0, 255).astype(np.uint8)
    
    return image_compressed, mask

def calculate_psnr(original, compressed):
    """Calculate PSNR between original and compressed images"""
    mse = np.mean((original.astype(float) - compressed.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def create_square_fourier_results():
    """Create a square Fourier compression results image using built-in images"""
    print("Creating Fourier compression results with built-in images...")
    
    # Load test images using your existing function
    def load_test_images():
        images = {}
        try:
            china = load_sample_image("china.jpg")
            images['china'] = np.mean(china, axis=2).astype(np.uint8)
        except:
            print("Could not load china image")
        
        # SKImage grayscale images
        images['camera'] = data.camera()
        images['coins'] = data.coins()
        
        return images
    
    test_images = load_test_images()
    
    if not test_images:
        print("No test images found!")
        return
    
    # Create the square results figure - FIXED: 2x3 grid means columns 0,1,2 only
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Fourier Transform Image Compression Results', fontsize=16, fontweight='bold')
    
    # Test different compression levels - FIXED: Only 2 levels to fit in 3 columns
    compression_levels = [0.3, 0.1]  # Keep 30%, 10% of frequencies (2 levels + original = 3 columns)
    
    # Use camera image for demonstration (consistent size)
    original_img = test_images['camera']
    
    # Top row: Spatial domain comparisons - FIXED INDEXING
    # Column 0: Original image
    axes[0, 0].imshow(original_img, cmap='gray')
    axes[0, 0].set_title(f'Original Image\nCamera\n{original_img.shape}', 
                       fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Columns 1 and 2: Compressed versions
    for comp_idx, level in enumerate(compression_levels):
        compressed, mask = fourier_circular_compress(original_img, keep_fraction=level)
        psnr = calculate_psnr(original_img, compressed)
        compression_ratio = original_img.size / np.sum(mask)
        
        axes[0, comp_idx+1].imshow(compressed, cmap='gray')  # comp_idx+1 gives columns 1 and 2
        axes[0, comp_idx+1].set_title(f'Keep {level*100:.0f}% frequencies\nPSNR: {psnr:.1f} dB\nCompression: {compression_ratio:.1f}x', 
                                    fontsize=11)
        axes[0, comp_idx+1].axis('off')
    
    # Bottom row: Frequency domain visualizations - FIXED INDEXING
    # Column 0: Original FFT spectrum
    fft_orig = fftshift(fft2(original_img))
    axes[1, 0].imshow(np.log(1 + np.abs(fft_orig)), cmap='viridis')
    axes[1, 0].set_title('Original FFT Spectrum\n(Log magnitude)', 
                       fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Columns 1 and 2: Compressed FFT spectra
    for comp_idx, level in enumerate(compression_levels):
        compressed, mask = fourier_circular_compress(original_img, keep_fraction=level)
        fft_comp = fftshift(fft2(compressed))
        
        # Create visualization with mask overlay
        fft_viz = np.log(1 + np.abs(fft_comp))
        axes[1, comp_idx+1].imshow(fft_viz, cmap='viridis')  # comp_idx+1 gives columns 1 and 2
        
        # Draw the actual mask boundary
        rows, cols = original_img.shape
        crow, ccol = rows // 2, cols // 2
        radius = min(rows, cols) * np.sqrt(level) / 2
        
        # Create circle for mask boundary
        circle = plt.Circle((ccol, crow), radius, fill=False, color='red', 
                          linewidth=3, linestyle='--', alpha=0.8)
        axes[1, comp_idx+1].add_artist(circle)
        
        # Add text inside the circle
        axes[1, comp_idx+1].text(ccol, crow, 'Kept\nFrequencies', 
                               ha='center', va='center', fontsize=10, 
                               color='white', fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor="red", alpha=0.7))
        
        axes[1, comp_idx+1].set_title(f'FFT: Keep {level*100:.0f}% frequencies\n(Red circle = kept region)', 
                                    fontsize=11)
        axes[1, comp_idx+1].axis('off')
    
    # Add compression insights at the bottom
    plt.figtext(0.5, 0.01, 
                'Fourier Compression Insights: • Lower frequencies contain most image energy • Higher compression = more blurring • Circular mask prevents directional artifacts', 
                ha='center', fontsize=12, style='italic',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)  # Make space for the footer
    
    # Save high-quality result
    plt.savefig('fourier_results.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✓ Saved: fourier_results.png")
    
    plt.show()

# Run the function
if __name__ == "__main__":
    create_square_fourier_results()