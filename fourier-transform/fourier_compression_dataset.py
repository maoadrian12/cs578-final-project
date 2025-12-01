import os
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import tarfile
import zipfile
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from PIL import Image
from io import BytesIO
from tqdm import tqdm  # For progress bar
import pandas as pd

# Install required packages if not already installed
# pip install tqdm pandas

class BSDS500Dataset:
    def __init__(self):
        self.dataset_url = "https://github.com/BIDS/BSDS500/archive/refs/heads/master.zip"
        self.data_dir = "BSDS500_data"
        self.images = []
        self.image_names = []
        
    def download_and_extract(self):
        """Download and extract the BSDS500 dataset"""
        print("Downloading BSDS500 dataset...")
        
        # Create directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        zip_path = os.path.join(self.data_dir, "BSDS500.zip")
        
        # Download the dataset
        try:
            urllib.request.urlretrieve(self.dataset_url, zip_path)
            print("✓ Dataset downloaded")
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            print("✓ Dataset extracted")
            
            # Remove the zip file to save space
            os.remove(zip_path)
            
        except Exception as e:
            print(f"✗ Error downloading dataset: {e}")
            print("Using fallback - built-in images only")
            return False
        
        return True
    
    def load_all_images(self, max_images=None):
        """Load all images from the BSDS500 dataset"""
        print("\nLoading BSDS500 images...")
        
        # Paths to the image directories
        train_path = os.path.join(self.data_dir, "BSDS500-master", "BSDS500", "data", "images", "train")
        test_path = os.path.join(self.data_dir, "BSDS500-master", "BSDS500", "data", "images", "test")
        val_path = os.path.join(self.data_dir, "BSDS500-master", "BSDS500", "data", "images", "val")
        
        all_image_paths = []
        
        # Collect all image paths
        for path in [train_path, test_path, val_path]:
            if os.path.exists(path):
                for file in os.listdir(path):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        all_image_paths.append(os.path.join(path, file))
        
        if not all_image_paths:
            print("No images found in dataset! Using built-in images as fallback.")
            return self.load_fallback_images(max_images)
        
        # Limit number of images if specified
        if max_images:
            all_image_paths = all_image_paths[:max_images]
        
        # Load images with progress bar
        self.images = []
        self.image_names = []
        
        for img_path in tqdm(all_image_paths, desc="Loading images"):
            try:
                # Load and convert to grayscale
                img = Image.open(img_path).convert('L')
                img_array = np.array(img)
                
                # Resize if too large (for memory efficiency)
                if img_array.shape[0] > 1024 or img_array.shape[1] > 1024:
                    img = img.resize((512, 512))
                    img_array = np.array(img)
                
                self.images.append(img_array)
                self.image_names.append(os.path.basename(img_path))
                
            except Exception as e:
                print(f"\n✗ Error loading {img_path}: {e}")
                continue
        
        print(f"✓ Successfully loaded {len(self.images)} images from BSDS500")
        return len(self.images) > 0
    
    def load_fallback_images(self, max_images=None):
        """Load built-in images as fallback"""
        from skimage import data
        
        print("Loading built-in images as fallback...")
        
        builtin_images = [
            data.camera(),
            data.coins(),
            data.text(),
            data.hubble_deep_field()[::4, ::4],  # Downsample for speed
            data.moon(),
            data.chelsea()[:,:,0]  # Red channel as grayscale
        ]
        
        self.images = builtin_images[:max_images] if max_images else builtin_images
        self.image_names = [f"builtin_{i}" for i in range(len(self.images))]
        
        print(f"✓ Loaded {len(self.images)} built-in images")
        return len(self.images) > 0

class FourierAnalyzer:
    def __init__(self):
        self.results = []
    
    def fourier_circular_compress(self, image, keep_fraction=0.1):
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
    
    def calculate_psnr(self, original, compressed):
        """Calculate PSNR between original and compressed images"""
        # Handle float images (0-1 range) vs uint8 (0-255)
        if original.dtype == np.float32 or original.dtype == np.float64:
            original = (original * 255).astype(np.uint8)
        if compressed.dtype == np.float32 or compressed.dtype == np.float64:
            compressed = (compressed * 255).astype(np.uint8)
        
        mse = np.mean((original.astype(float) - compressed.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))
    
    def analyze_dataset(self, images, image_names, compression_levels=[0.1, 0.2, 0.5]):
        """Analyze Fourier compression on entire dataset"""
        print(f"\nAnalyzing Fourier compression on {len(images)} images...")
        
        self.results = []
        
        for idx, (image, name) in enumerate(tqdm(zip(images, image_names), 
                                                 total=len(images), 
                                                 desc="Processing images")):
            
            # Convert to uint8 if needed
            if image.dtype == np.float32 or image.dtype == np.float64:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            image_result = {
                'name': name,
                'size': f"{image.shape[0]}x{image.shape[1]}",
                'total_pixels': image.size
            }
            
            for level in compression_levels:
                compressed, mask = self.fourier_circular_compress(image, keep_fraction=level)
                psnr = self.calculate_psnr(image, compressed)
                compression_ratio = image.size / np.sum(mask)
                
                image_result[f'keep_{int(level*100)}%_psnr'] = psnr
                image_result[f'keep_{int(level*100)}%_ratio'] = compression_ratio
                image_result[f'keep_{int(level*100)}%_kept'] = np.sum(mask)
            
            self.results.append(image_result)
        
        print("✓ Analysis complete!")
        return self.results
    
    def generate_summary_statistics(self):
        """Generate comprehensive statistics from all results"""
        if not self.results:
            print("No results to analyze!")
            return None
        
        df = pd.DataFrame(self.results)
        
        compression_levels = [0.1, 0.2, 0.5]
        
        print("\n" + "="*60)
        print("FOURIER COMPRESSION - COMPREHENSIVE DATASET ANALYSIS")
        print("="*60)
        
        for level in compression_levels:
            level_key = f'keep_{int(level*100)}%_psnr'
            ratio_key = f'keep_{int(level*100)}%_ratio'
            
            if level_key in df.columns:
                psnr_values = df[level_key]
                ratio_values = df[ratio_key]
                
                print(f"\n--- Keep {int(level*100)}% of frequencies ---")
                print(f"  PSNR Statistics:")
                print(f"    Min:    {psnr_values.min():.2f} dB")
                print(f"    Max:    {psnr_values.max():.2f} dB")
                print(f"    Mean:   {psnr_values.mean():.2f} dB")
                print(f"    Median: {psnr_values.median():.2f} dB")
                print(f"    Std:    {psnr_values.std():.2f} dB")
                
                print(f"  Compression Ratio Statistics:")
                print(f"    Min:    {ratio_values.min():.2f}x")
                print(f"    Max:    {ratio_values.max():.2f}x")
                print(f"    Mean:   {ratio_values.mean():.2f}x")
                print(f"    Median: {ratio_values.median():.2f}x")
        
        # Generate final summary table
        print("\n" + "="*60)
        print("SUMMARY TABLE FOR PRESENTATION:")
        print("="*60)
        print("\nKeep % | PSNR Range (dB) | Avg PSNR | Avg Compression")
        print("-" * 55)
        
        for level in compression_levels:
            level_key = f'keep_{int(level*100)}%_psnr'
            ratio_key = f'keep_{int(level*100)}%_ratio'
            
            if level_key in df.columns:
                psnr_min = df[level_key].min()
                psnr_max = df[level_key].max()
                psnr_mean = df[level_key].mean()
                ratio_mean = df[ratio_key].mean()
                
                print(f"  {int(level*100):2d}%   | {psnr_min:5.1f}-{psnr_max:5.1f} | {psnr_mean:7.1f} | {ratio_mean:6.1f}x")
        
        return df
    
    def create_visual_summary(self, images, image_names, save_path="fourier_dataset_summary.png"):
        """Create a visual summary of results"""
        if not images:
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle('Fourier Compression - BSDS500 Dataset Analysis', fontsize=16, fontweight='bold')
        
        # Use first 3 images for visualization
        for i in range(min(3, len(images))):
            original_img = images[i]
            
            # Original image
            axes[i, 0].imshow(original_img, cmap='gray')
            axes[i, 0].set_title(f'{image_names[i]}\n{original_img.shape}', fontsize=10)
            axes[i, 0].axis('off')
            
            # Compressed versions (10%, 20%, 50%)
            for j, level in enumerate([0.1, 0.2, 0.5]):
                compressed, _ = self.fourier_circular_compress(original_img, keep_fraction=level)
                psnr = self.calculate_psnr(original_img, compressed)
                
                axes[i, j+1].imshow(compressed, cmap='gray')
                axes[i, j+1].set_title(f'Keep {level*100:.0f}%\nPSNR: {psnr:.1f} dB', fontsize=10)
                axes[i, j+1].axis('off')
        
        # Add FFT visualization for first image
        if len(images) > 0:
            original_img = images[0]
            fft_orig = fftshift(fft2(original_img))
            
            axes[2, 0].imshow(np.log(1 + np.abs(fft_orig)), cmap='viridis')
            axes[2, 0].set_title('FFT Spectrum (log)\nFirst Image', fontsize=10)
            axes[2, 0].axis('off')
        
        # Remove empty subplots
        for i in range(3):
            for j in range(4):
                if not axes[i, j].has_data():
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visual summary saved to {save_path}")
        
        plt.show()

def main():
    """Main function to run complete Fourier analysis on BSDS500 dataset"""
    print("="*70)
    print("FOURIER COMPRESSION ANALYSIS - BSDS500 DATASET")
    print("="*70)
    
    # Initialize dataset
    dataset = BSDS500Dataset()
    
    # Try to download and load dataset
    if dataset.download_and_extract():
        # Load all images (set max_images to None for all, or e.g., 50 for testing)
        dataset.load_all_images(max_images=50)  # Use 50 images for testing
    else:
        # Fallback to built-in images
        dataset.load_fallback_images(max_images=10)
    
    if not dataset.images:
        print("Error: No images available!")
        return
    
    # Initialize analyzer
    analyzer = FourierAnalyzer()
    
    # Analyze all images
    analyzer.analyze_dataset(dataset.images, dataset.image_names)
    
    # Generate comprehensive statistics
    df = analyzer.generate_summary_statistics()
    
    # Create visual summary
    analyzer.create_visual_summary(dataset.images[:3], dataset.image_names[:3])
    
    # Save detailed results to CSV
    if df is not None:
        df.to_csv('fourier_dataset_results.csv', index=False)
        print("\n✓ Detailed results saved to 'fourier_dataset_results.csv'")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()