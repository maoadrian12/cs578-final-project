import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from sklearn.datasets import load_sample_image
from skimage import data, img_as_float
import os
from PIL import Image

def load_test_images():
    """Load multiple test images for comparison"""
    images = {}
    
    # Method 1: Built-in sample images
    try:
        china = load_sample_image("china.jpg")
        images['china'] = np.mean(china, axis=2).astype(np.uint8)
    except:
        print("Could not load china image")
    
    # Method 2: SKImage grayscale images
    images['camera'] = data.camera()
    images['coins'] = data.coins()
    
    # Method 3: Check for local images
    if os.path.exists('test_image.jpg'):
        local_img = Image.open('test_image.jpg').convert('L')
        images['local'] = np.array(local_img)
    
    return images

def fourier_low_freq_compress(image, keep_fraction=0.1):
    """
    Compress image by keeping only low-frequency components
    
    Parameters:
    image: 2D numpy array (grayscale)
    keep_fraction: proportion of coefficients to keep (0.0 to 1.0)
    
    Returns:
    compressed_image, compression_ratio, coefficients_kept
    """
    # Step 1: Compute 2D FFT
    fft_original = fft2(image)
    
    # Step 2: Shift zero frequency to center
    fft_shifted = fftshift(fft_original)
    
    # Step 3: Create mask for low-frequency components
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    # Calculate keep radius (square region)
    r_keep = int(rows * np.sqrt(keep_fraction) / 2)
    c_keep = int(cols * np.sqrt(keep_fraction) / 2)
    
    # Ensure we keep at least 1 pixel
    r_keep = max(1, r_keep)
    c_keep = max(1, c_keep)
    
    # Create mask
    mask = np.zeros((rows, cols), dtype=bool)
    mask[crow-r_keep:crow+r_keep, ccol-c_keep:ccol+c_keep] = True
    
    # Step 4: Apply mask (zero out high frequencies)
    fft_compressed = fft_shifted * mask
    
    # Step 5: Inverse shift and inverse FFT
    fft_ishifted = ifftshift(fft_compressed)
    image_compressed = np.real(ifft2(fft_ishifted))
    
    # Clip to valid range
    image_compressed = np.clip(image_compressed, 0, 255).astype(np.uint8)
    
    # Calculate compression metrics
    total_coefficients = rows * cols
    coefficients_kept = np.sum(mask)
    compression_ratio = total_coefficients / coefficients_kept
    
    return image_compressed, compression_ratio, coefficients_kept

def fourier_threshold_compress(image, threshold_ratio=0.1):
    """
    Compress by thresholding small coefficients
    
    Parameters:
    image: 2D numpy array (grayscale)
    threshold_ratio: fraction of max magnitude to use as threshold
    
    Returns:
    compressed_image, compression_ratio, coefficients_kept
    """
    # Step 1: Compute 2D FFT and shift
    fft_original = fft2(image)
    fft_shifted = fftshift(fft_original)
    
    # Step 2: Calculate magnitude and threshold
    magnitude = np.abs(fft_shifted)
    threshold = np.max(magnitude) * threshold_ratio
    
    # Step 3: Zero out coefficients below threshold
    mask = magnitude > threshold
    fft_compressed = fft_shifted * mask
    
    # Step 4: Reconstruct image
    fft_ishifted = ifftshift(fft_compressed)
    image_compressed = np.real(ifft2(fft_ishifted))
    image_compressed = np.clip(image_compressed, 0, 255).astype(np.uint8)
    
    # Calculate compression metrics
    total_coefficients = image.size
    coefficients_kept = np.sum(mask)
    compression_ratio = total_coefficients / coefficients_kept
    
    return image_compressed, compression_ratio, coefficients_kept

def calculate_metrics(original, compressed):
    """Calculate MSE and PSNR between original and compressed images"""
    # Convert to float for calculation
    orig_float = original.astype(np.float64)
    comp_float = compressed.astype(np.float64)
    
    mse = np.mean((orig_float - comp_float) ** 2)
    
    if mse == 0:
        return 0, float('inf')
    
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return mse, psnr

def run_comparison():
    """Main function to run all comparisons"""
    print("Loading test images...")
    test_images = load_test_images()
    
    if not test_images:
        print("No test images found! Please add some images to the directory.")
        return
    
    # Test different compression levels
    compression_levels = [0.01, 0.05, 0.1, 0.2, 0.3]
    
    for image_name, original_image in test_images.items():
        print(f"\n{'='*50}")
        print(f"Testing on: {image_name} (Size: {original_image.shape})")
        print(f"{'='*50}")
        
        results_low_freq = []
        results_threshold = []
        
        # Test low-frequency method
        print(f"\nLow-Frequency Compression Method:")
        print("Level | Keep % | MSE    | PSNR (dB) | Comp Ratio")
        print("-" * 50)
        
        for level in compression_levels:
            compressed, comp_ratio, coeffs_kept = fourier_low_freq_compress(
                original_image, keep_fraction=level
            )
            mse, psnr = calculate_metrics(original_image, compressed)
            
            results_low_freq.append({
                'level': level,
                'mse': mse,
                'psnr': psnr,
                'comp_ratio': comp_ratio,
                'image': compressed
            })
            
            print(f"{level:5.2f} | {level*100:6.1f}% | {mse:6.1f} | {psnr:8.2f} | {comp_ratio:8.2f}x")
        
        # Test threshold method
        print(f"\nThreshold Compression Method:")
        print("Thresh | MSE    | PSNR (dB) | Comp Ratio")
        print("-" * 45)
        
        for level in compression_levels:
            compressed, comp_ratio, coeffs_kept = fourier_threshold_compress(
                original_image, threshold_ratio=level
            )
            mse, psnr = calculate_metrics(original_image, compressed)
            
            results_threshold.append({
                'level': level,
                'mse': mse,
                'psnr': psnr,
                'comp_ratio': comp_ratio,
                'image': compressed
            })
            
            print(f"{level:6.2f} | {mse:6.1f} | {psnr:8.2f} | {comp_ratio:8.2f}x")
        
        # Create visualization
        create_visualization(original_image, results_low_freq, results_threshold, image_name)

def create_visualization(original, results_low, results_thresh, image_name):
    """Create comparison plots for the results"""
    
    # Create separate figures for each method
    
    # Figure 1: Low-frequency compression results (the one that works well)
    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))
    fig1.suptitle(f'Low-Frequency Fourier Compression - {image_name}', fontsize=16)
    
    # Row 1: Original and compressed images
    axes1[0,0].imshow(original, cmap='gray')
    axes1[0,0].set_title('Original Image')
    axes1[0,0].axis('off')
    
    # Show 2 compression levels instead of 3
    display_indices = [0, 2]  # Just show 2 examples: low and medium compression
    for i, idx in enumerate(display_indices):
        result = results_low[idx]
        axes1[0,i+1].imshow(result['image'], cmap='gray')
        axes1[0,i+1].set_title(f'Keep {result["level"]*100:.1f}%\nPSNR: {result["psnr"]:.1f} dB')
        axes1[0,i+1].axis('off')
    
    # Leave the last spot in row 1 empty or show FFT
    fft_original = fftshift(fft2(original))
    axes1[0,2].imshow(np.log(1 + np.abs(fft_original)), cmap='viridis')
    axes1[0,2].set_title('Original FFT (log scale)')
    axes1[0,2].axis('off')
    
    # Row 2: Metrics plots
    levels = [r['level'] for r in results_low]
    psnr_low = [r['psnr'] for r in results_low]
    comp_ratios = [r['comp_ratio'] for r in results_low]
    mse_values = [r['mse'] for r in results_low]
    
    # Plot 1: PSNR vs Compression Level
    axes1[1,0].plot(levels, psnr_low, 'bo-', linewidth=2, markersize=8)
    axes1[1,0].set_xlabel('Keep Fraction')
    axes1[1,0].set_ylabel('PSNR (dB)')
    axes1[1,0].set_title('Quality vs Compression Level')
    axes1[1,0].grid(True)
    
    # Plot 2: PSNR vs Compression Ratio
    axes1[1,1].plot(comp_ratios, psnr_low, 'ro-', linewidth=2, markersize=8)
    axes1[1,1].set_xlabel('Compression Ratio')
    axes1[1,1].set_ylabel('PSNR (dB)')
    axes1[1,1].set_title('Quality vs Compression Ratio')
    axes1[1,1].grid(True)
    
    # Plot 3: MSE vs Compression Level
    axes1[1,2].plot(levels, mse_values, 'go-', linewidth=2, markersize=8)
    axes1[1,2].set_xlabel('Keep Fraction')
    axes1[1,2].set_ylabel('MSE')
    axes1[1,2].set_title('Error vs Compression Level')
    axes1[1,2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'low_freq_results_{image_name}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Optional: Create a simple threshold visualization if you want to see it
    if len(results_thresh) > 0:
        fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
        fig2.suptitle(f'Threshold Method Examples - {image_name}', fontsize=14)
        
        # Show worst and best of threshold method
        if len(results_thresh) >= 2:
            axes2[0].imshow(results_thresh[0]['image'], cmap='gray')
            axes2[0].set_title(f'Thresh: {results_thresh[0]["level"]:.2f}\nPSNR: {results_thresh[0]["psnr"]:.1f} dB')
            axes2[0].axis('off')
            
            axes2[1].imshow(results_thresh[-1]['image'], cmap='gray')
            axes2[1].set_title(f'Thresh: {results_thresh[-1]["level"]:.2f}\nPSNR: {results_thresh[-1]["psnr"]:.1f} dB')
            axes2[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'threshold_results_{image_name}.png', dpi=150, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    print("Fourier Transform Image Compression Analysis")
    print("Christina Zhang - Image Processing Project")
    run_comparison()