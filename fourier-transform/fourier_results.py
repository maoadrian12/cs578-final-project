import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from skimage import data, transform, img_as_ubyte

def download_bsds_sample():
    """Download sample images from BSDS500 dataset"""
    print("Downloading BSDS500 sample images...")
    urls = [
        "https://github.com/BIDS/BSDS500/raw/master/BSDS500/data/images/train/101027.jpg",
        "https://github.com/BIDS/BSDS500/raw/master/BSDS500/data/images/train/106024.jpg", 
        "https://github.com/BIDS/BSDS500/raw/master/BSDS500/data/images/train/253027.jpg"
    ]
    
    downloaded_images = []
    
    for i, url in enumerate(urls):
        try:
            response = requests.get(url, timeout=10)
            filename = f'bsds_{i}.jpg'
            with open(filename, 'wb') as f:
                f.write(response.content)
            downloaded_images.append(filename)
            print(f"✓ Downloaded: {filename}")
        except Exception as e:
            print(f"✗ Failed to download {url}: {e}")
    
    return downloaded_images

def load_test_images():
    """Load test images, prioritizing BSDS500, with fallback to built-in images"""
    images = []
    image_names = []
    
    # Try to load downloaded BSDS500 images
    bsds_files = [f for f in os.listdir('.') if f.startswith('bsds_') and f.endswith('.jpg')]
    
    if bsds_files:
        print("Using BSDS500 images:")
        for file in bsds_files[:2]:  # Use first 2 BSDS images
            try:
                img = Image.open(file).convert('L')
                img = img.resize((256, 256))  # Resize to square
                images.append(np.array(img))
                image_names.append(f"BSDS500_{file.split('_')[1].split('.')[0]}")
                print(f"✓ Loaded: {file}")
            except Exception as e:
                print(f"✗ Error loading {file}: {e}")
    else:
        print("No BSDS500 images found, using built-in images")
        # Fallback to built-in images
        builtin_imgs = [
            (data.camera(), "Camera"),
            (transform.resize(data.coins(), (256, 256)), "Coins"),
            (transform.resize(data.text(), (256, 256)), "Text")
        ]
        
        for img, name in builtin_imgs[:2]:  # Use first 2 built-in images
            if img.dtype == np.float64:
                img = img_as_ubyte(img)
            images.append(img)
            image_names.append(name)
            print(f"✓ Using built-in: {name}")
    
    return images, image_names

def fourier_compress(image, keep_fraction=0.1):
    """Compress image using Fourier Transform with circular mask"""
    fft_original = fft2(image)
    fft_shifted = fftshift(fft_original)
    
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create circular mask for frequency selection
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
    """Create a square Fourier compression results image"""
    print("\nCreating Fourier compression results...")
    
    # Download and load images
    download_bsds_sample()
    test_images, image_names = load_test_images()
    
    if not test_images:
        print("Error: No test images available!")
        return
    
    # Create the square results figure
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Fourier Transform Image Compression Results', fontsize=16, fontweight='bold')
    
    # Test different compression levels
    compression_levels = [0.3, 0.1, 0.05]  # Keep 30%, 10%, 5% of frequencies
    
    # Use first image for demonstration
    original_img = test_images[0]
    img_name = image_names[0]
    
    # Top row: Spatial domain comparisons
    for comp_idx, level in enumerate(compression_levels):
        if comp_idx == 0:
            # Original image
            axes[0, 0].imshow(original_img, cmap='gray')
            axes[0, 0].set_title(f'Original Image\n{img_name}\n{original_img.shape}', 
                               fontsize=11, fontweight='bold')
            axes[0, 0].axis('off')
        
        # Compressed versions
        compressed, mask = fourier_compress(original_img, keep_fraction=level)
        psnr = calculate_psnr(original_img, compressed)
        
        axes[0, comp_idx+1].imshow(compressed, cmap='gray')
        axes[0, comp_idx+1].set_title(f'Keep {level*100:.0f}% frequencies\nPSNR: {psnr:.1f} dB\nCompression: {1/level:.1f}x', 
                                    fontsize=11)
        axes[0, comp_idx+1].axis('off')
    
    # Bottom row: Frequency domain visualizations
    for comp_idx, level in enumerate(compression_levels):
        if comp_idx == 0:
            # Original FFT spectrum
            fft_orig = fftshift(fft2(original_img))
            axes[1, 0].imshow(np.log(1 + np.abs(fft_orig)), cmap='viridis')
            axes[1, 0].set_title('Original FFT Spectrum\n(Log magnitude)', 
                               fontsize=11, fontweight='bold')
            axes[1, 0].axis('off')
        else:
            # Compressed FFT spectrum with mask visualization
            compressed, mask = fourier_compress(original_img, keep_fraction=level)
            fft_comp = fftshift(fft2(compressed))
            
            # Create visualization with mask overlay
            fft_viz = np.log(1 + np.abs(fft_comp))
            axes[1, comp_idx+1].imshow(fft_viz, cmap='viridis')
            
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
    
    # Display file info
    file_size = os.path.getsize('fourier_results.png') / 1024  # KB
    print(f"✓ File size: {file_size:.1f} KB")
    
    plt.show()

def main():
    """Main function to run the Fourier compression analysis"""
    print("=" * 60)
    print("Fourier Transform Image Compression Analysis")
    print("Using BSDS500 dataset + fallback to built-in images")
    print("=" * 60)
    
    create_square_fourier_results()
    
    print("\n" + "=" * 60)
    print("Analysis complete! Check 'fourier_results.png'")
    print("=" * 60)

if __name__ == "__main__":
    main()