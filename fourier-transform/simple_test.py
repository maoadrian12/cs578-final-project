import numpy as np
import matplotlib.pyplot as plt
from fourier_compression import fourier_low_freq_compress, calculate_metrics
from skimage import data

# Quick test with one image
print("Quick Fourier Compression Test")
print("=" * 40)

# Load test image
image = data.camera()

# Test single compression level
compressed, ratio, coeffs = fourier_low_freq_compress(image, keep_fraction=0.1)
mse, psnr = calculate_metrics(image, compressed)

print(f"Original size: {image.shape}")
print(f"Compression ratio: {ratio:.2f}x")
print(f"Coefficients kept: {coeffs}/{image.size} ({coeffs/image.size*100:.1f}%)")
print(f"MSE: {mse:.2f}")
print(f"PSNR: {psnr:.2f} dB")

# Display results
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(compressed, cmap='gray')
plt.title(f'Compressed (PSNR: {psnr:.1f} dB)')
plt.axis('off')

plt.tight_layout()
plt.savefig('quick_test_result.png', dpi=150)
plt.show()