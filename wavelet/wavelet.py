import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pywt

# --- 1. The Compression Function (From previous step) ---
def calculate_entropy(signal):
    vals, counts = np.unique(signal, return_counts=True)
    probs = counts / len(signal)
    return -np.sum(probs * np.log2(probs + 1e-10))

def compress_wavelet(image_path, quant_step):
    # Load as Grayscale (L) for simplicity in this project
    try:
        img = Image.open(image_path).convert('L')
    except Exception as e:
        print(f"Skipping {image_path}: {e}")
        return None, None
        
    img_arr = np.array(img)
    
    # Wavelet Transform (using 'bior4.4' standard)
    coeffs = pywt.wavedec2(img_arr, wavelet='bior4.4', level=2)
    
    # Quantize
    coeffs_quant = []
    for c in coeffs:
        if isinstance(c, tuple):
            coeffs_quant.append(tuple(np.round(subband / quant_step) for subband in c))
        else:
            coeffs_quant.append(np.round(c / quant_step))

    # Calculate Rate (Entropy)
    all_coeffs = np.concatenate([c.flatten() for c_group in coeffs_quant 
                                 for c in (c_group if isinstance(c_group, tuple) else [c_group])])
    entropy = calculate_entropy(all_coeffs)
    
    # Reconstruction
    coeffs_recon = []
    for c in coeffs_quant:
        if isinstance(c, tuple):
            coeffs_recon.append(tuple(subband * quant_step for subband in c))
        else:
            coeffs_recon.append(c * quant_step)
            
    img_recon = pywt.waverec2(coeffs_recon, wavelet='bior4.4')
    
    # Calculate Distortion (MSE)
    # Crop reconstruction to match original size (padding sometimes happens in DWT)
    h, w = img_arr.shape
    img_recon = img_recon[:h, :w]
    mse = np.mean((img_arr - img_recon) ** 2)
    
    return mse, entropy, img_recon

# --- 2. The Batch Processing Loop ---

def run_experiment(image_folder):
    # Get list of all jpg/png files
    # Note: Update valid extensions if your files are different
    files = glob.glob(os.path.join(image_folder, "*.jpg")) + \
            glob.glob(os.path.join(image_folder, "*.png"))
    output_dir = "compressed_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    if not files:
        print("No images found! Check your path.")
        return

    print(f"Found {len(files)} images. Starting compression...")

    # We test different "Qualities". 
    # Smaller step = Higher Quality (Low MSE, High Bits)
    # Larger step = Lower Quality (High MSE, Low Bits)
    quant_steps = [10, 20, 30, 50, 75, 100, 150]
    
    avg_mses = []
    avg_rates = []

    for q in quant_steps:
        batch_mse = []
        batch_rate = []
        
        for i, f in enumerate(files):
            mse, rate, recon_img = compress_wavelet(f, q)
            if mse is not None:
                batch_mse.append(mse)
                batch_rate.append(rate)
                if i == 0:
                    # Convert numpy array back to PIL Image to save it
                    # We must clip values to 0-255 and cast to uint8
                    recon_img_clipped = np.clip(recon_img, 0, 255).astype(np.uint8)
                    im_out = Image.fromarray(recon_img_clipped)
                        
                    # Save with a clear filename
                    save_name = f"sample_q{q}.png"
                    save_path = os.path.join(output_dir, save_name)
                    im_out.save(save_path)
                    print(f"   --> Saved visual sample: {save_path}")
        
        # Calculate average for this quantization step
        current_avg_mse = np.mean(batch_mse)
        current_avg_rate = np.mean(batch_rate)
        
        avg_mses.append(current_avg_mse)
        avg_rates.append(current_avg_rate)
        
        print(f"Quant Step {q}: Avg MSE = {current_avg_mse:.2f}, Avg Bitrate = {current_avg_rate:.2f}")

    return avg_rates, avg_mses

# --- 3. Execution and Plotting ---

folder_path = "../images/test/" 

rates, mses = run_experiment(folder_path)

if rates:
    plt.figure(figsize=(10, 6))
    plt.plot(rates, mses, marker='o', linestyle='-', color='b', label='Wavelet (Bior4.4)')
    
    plt.title("Rate-Distortion Curve (BSDS Dataset)")
    plt.xlabel("Estimated Bitrate (Entropy)")
    plt.ylabel("Distortion (MSE)")
    plt.grid(True)
    plt.legend()
    plt.savefig("rd_curve.png")
    plt.show()
    print("Plot saved as 'rd_curve.png'")