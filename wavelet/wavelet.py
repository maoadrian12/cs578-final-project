import numpy as np
import pywt
from PIL import Image
import os

# --- 1. Helper Functions ---

def calculate_entropy_bpp(signal):
    """Returns estimated Bits Per Pixel (bpp) based on entropy."""
    vals, counts = np.unique(signal, return_counts=True)
    probs = counts / len(signal)
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    return entropy

def calculate_mse(original, reconstructed):
    """Calculate Mean Squared Error."""
    return np.mean((original - reconstructed) ** 2)

def calculate_psnr(original, reconstructed):
    mse = calculate_mse(original, reconstructed)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def compress_measure_single(img_arr, quant_step):
    """
    Runs one compression cycle.
    Returns: (bpp, mse, psnr, reconstructed_image)
    """
    # 1. Wavelet Transform (Biorthogonal 4.4 is standard)
    coeffs = pywt.wavedec2(img_arr, wavelet='bior4.4', level=3)
    
    # 2. Quantize
    coeffs_quant = []
    for c in coeffs:
        if isinstance(c, tuple):
            coeffs_quant.append(tuple(np.round(subband / quant_step) for subband in c))
        else:
            coeffs_quant.append(np.round(c / quant_step))

    # 3. Calculate Bitrate (Entropy)
    all_coeffs = np.concatenate([c.flatten() for c_group in coeffs_quant 
                                 for c in (c_group if isinstance(c_group, tuple) else [c_group])])
    bpp = calculate_entropy_bpp(all_coeffs)

    # 4. Reconstruct
    coeffs_recon = []
    for c in coeffs_quant:
        if isinstance(c, tuple):
            coeffs_recon.append(tuple(subband * quant_step for subband in c))
        else:
            coeffs_recon.append(c * quant_step)
            
    img_recon = pywt.waverec2(coeffs_recon, wavelet='bior4.4')
    
    # Fix dimensions
    h, w = img_arr.shape
    img_recon = img_recon[:h, :w]
    
    # Calculate MSE and PSNR
    mse = calculate_mse(img_arr, img_recon)
    psnr = calculate_psnr(img_arr, img_recon)
    
    return bpp, mse, psnr, img_recon

# --- 2. Main Search Logic ---

def find_psnr_for_ratio(image_path, target_ratio=10.0):
    print(f"Processing: {image_path}")
    print(f"Target Compression Ratio: {target_ratio}x")
    
    # Load Image
    try:
        img = Image.open(image_path).convert('L')
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    img_arr = np.array(img)
    
    # Standard grayscale is 8 bits per pixel.
    # So 10x compression means we want roughly 0.8 bits per pixel.
    original_bpp = 8.0
    target_bpp = original_bpp / target_ratio
    
    print(f"Target Bitrate: ~{target_bpp:.3f} bpp")
    print("Searching for best match...")

    best_match = {
        'diff': float('inf'),
        'q': 0,
        'bpp': 0,
        'psnr': 0,
        'recon': None
    }
    
    # Sweep through quantization steps (q)
    # We use a finer step (0.5) to get closer to the exact 10x mark
    for q in np.arange(1.0, 200.0, 0.5):
        bpp, psnr, recon = compress_measure_single(img_arr, q)
        
        # How far are we from 10x compression?
        diff = abs(bpp - target_bpp)
        
        if diff < best_match['diff']:
            best_match['diff'] = diff
            best_match['q'] = q
            best_match['bpp'] = bpp
            best_match['psnr'] = psnr
            best_match['recon'] = recon

    # --- 3. Output Results ---
    
    if best_match['recon'] is None:
        print("Could not find a match.")
        return

    final_ratio = original_bpp / best_match['bpp']
    
    print("\n" + "="*40)
    print(f" RESULT AT ~{target_ratio}x COMPRESSION")
    print("="*40)
    print(f"Quantization Step (q): {best_match['q']}")
    print(f"Actual Ratio:          {final_ratio:.2f}x")
    print(f"Actual Bitrate:        {best_match['bpp']:.3f} bpp")
    print("-" * 40)
    print(f"Resulting PSNR:        {best_match['psnr']:.2f} dB")
    print("="*40)
    
    # Save the image
    output_dir = "ratio_study"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"result_{target_ratio}x_ratio.png")
    
    recon_clipped = np.clip(best_match['recon'], 0, 255).astype(np.uint8)
    Image.fromarray(recon_clipped).save(save_path)
    print(f"\nImage saved to: {save_path}")

# --- Run Config ---
def process_all_images_batch():
    """
    Process all images in ./images/test folder for multiple compression ratios
    and compute average PSNR values. Saves 10 random images to saved_images folder.
    """
    test_folder = "./images/test"
    target_ratios = [5.0, 10.0, 20.0]
    output_folder = "./saved_images"
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Dictionary to store results
    results = {ratio: {'psnr_values': [], 'mse_values': [], 'avg_psnr': 0, 'avg_mse': 0} 
               for ratio in target_ratios}
    
    # Get all jpg files from test folder
    image_files = [f for f in os.listdir(test_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(image_files)} images in {test_folder}")
    print(f"Processing for compression ratios: {target_ratios}")
    print("="*60)
    
    for idx, img_file in enumerate(image_files, 1):
        img_path = os.path.join(test_folder, img_file)
        print(f"\n[{idx}/{len(image_files)}] Processing: {img_file}")
        
        # Load Image
        try:
            img = Image.open(img_path).convert('L')
        except Exception as e:
            print(f"  Error loading image: {e}")
            continue
        
        img_arr = np.array(img)
        original_bpp = 8.0
        
        # Test each compression ratio
        image_results = {}
        for target_ratio in target_ratios:
            target_bpp = original_bpp / target_ratio
            
            best_match = {
                'diff': float('inf'),
                'q': 0,
                'bpp': 0,
                'mse': 0,
                'psnr': 0,
                'recon': None
            }
            
            # Sweep through quantization steps
            for q in np.arange(1.0, 200.0, 0.5):
                bpp, mse, psnr, recon = compress_measure_single(img_arr, q)
                diff = abs(bpp - target_bpp)
                
                if diff < best_match['diff']:
                    best_match['diff'] = diff
                    best_match['q'] = q
                    best_match['bpp'] = bpp
                    best_match['mse'] = mse
                    best_match['psnr'] = psnr
                    best_match['recon'] = recon
            
            if best_match['recon'] is not None:
                results[target_ratio]['psnr_values'].append(best_match['psnr'])
                results[target_ratio]['mse_values'].append(best_match['mse'])
                image_results[target_ratio] = best_match
                print(f"  {target_ratio}x: MSE = {best_match['mse']:.4f}, PSNR = {best_match['psnr']:.2f} dB")
        
        # If this is one of the first 10 images, save compressed results immediately
        if image_results and idx <= 10:
            name_base = os.path.splitext(img_file)[0]
            for ratio, res in image_results.items():
                recon_img = res['recon']
                recon_clipped = np.clip(recon_img, 0, 255).astype(np.uint8)
                save_name = f"{name_base}_{ratio}x.png"
                save_path = os.path.join(output_folder, save_name)
                Image.fromarray(recon_clipped).save(save_path)
                mse = res.get('mse', None)
                psnr = res.get('psnr', None)
                print(f"[{idx}/10] Saved: {save_name} (MSE: {mse:.4f}, PSNR: {psnr:.2f} dB)")
    
    # (Saved first 10 images were written during processing.)
    
    # --- Print Summary Results ---
    print("\n" + "="*60)
    print("SUMMARY: AVERAGE MSE AND PSNR ACROSS ALL IMAGES")
    print("="*60)
    
    for target_ratio in target_ratios:
        psnr_vals = results[target_ratio]['psnr_values']
        mse_vals = results[target_ratio]['mse_values']
        if psnr_vals and mse_vals:
            avg_psnr = np.mean(psnr_vals)
            std_psnr = np.std(psnr_vals)
            min_psnr = np.min(psnr_vals)
            max_psnr = np.max(psnr_vals)
            
            avg_mse = np.mean(mse_vals)
            std_mse = np.std(mse_vals)
            min_mse = np.min(mse_vals)
            max_mse = np.max(mse_vals)
            
            results[target_ratio]['avg_psnr'] = avg_psnr
            results[target_ratio]['avg_mse'] = avg_mse
            
            print(f"\n{target_ratio}x Compression Ratio:")
            print(f"  PSNR:")
            print(f"    Average:   {avg_psnr:.2f} dB")
            print(f"    Std Dev:   {std_psnr:.2f} dB")
            print(f"    Min:       {min_psnr:.2f} dB")
            print(f"    Max:       {max_psnr:.2f} dB")
            print(f"  MSE:")
            print(f"    Average:   {avg_mse:.4f}")
            print(f"    Std Dev:   {std_mse:.4f}")
            print(f"    Min:       {min_mse:.4f}")
            print(f"    Max:       {max_mse:.4f}")
            print(f"  Num images:    {len(psnr_vals)}")
        else:
            print(f"\n{target_ratio}x Compression Ratio: No results")
    
    print("\n" + "="*60)
    
    return results

if __name__ == "__main__":
    process_all_images_batch()