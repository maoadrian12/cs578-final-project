import os
import sys
import argparse
import numpy as np
from PIL import Image

# Try importing compress_measure_single from local wavelet module
try:
    from wavelet import compress_measure_single
except Exception:
    sys.path.insert(0, os.path.dirname(__file__))
    from wavelet import compress_measure_single


def find_best_quant(img_arr, target_ratio, q_min=1.0, q_max=200.0, q_step=0.5):
    """Search quantization steps and return best (bpp,mse,psnr,recon,q) matching target ratio."""
    original_bpp = 8.0
    target_bpp = original_bpp / target_ratio

    best = {'diff': float('inf'), 'q': None, 'bpp': None, 'mse': None, 'psnr': None, 'recon': None}

    for q in np.arange(q_min, q_max, q_step):
        bpp, mse, psnr, recon = compress_measure_single(img_arr, q)
        diff = abs(bpp - target_bpp)
        if diff < best['diff']:
            best.update({'diff': diff, 'q': q, 'bpp': bpp, 'mse': mse, 'psnr': psnr, 'recon': recon})

    return best


def compress_and_save(image_path, ratio, outdir='./saved_images'):
    if ratio <= 0:
        raise ValueError('Compression ratio must be > 0')

    if not os.path.exists(image_path):
        raise FileNotFoundError(f'Image not found: {image_path}')

    os.makedirs(outdir, exist_ok=True)

    img = Image.open(image_path).convert('L')
    img_arr = np.array(img)

    print(f'Compressing "{os.path.basename(image_path)}" to ~{ratio}x...')
    best = find_best_quant(img_arr, ratio)

    if best['recon'] is None:
        print('No reconstruction produced.')
        return None

    # Save reconstructed image
    name_base = os.path.splitext(os.path.basename(image_path))[0]
    save_name = f"{name_base}_{ratio}x.png"
    save_path = os.path.join(outdir, save_name)

    recon_clipped = np.clip(best['recon'], 0, 255).astype(np.uint8)
    Image.fromarray(recon_clipped).save(save_path)

    achieved_ratio = 8.0 / best['bpp'] if best['bpp'] else float('nan')

    print('Saved:', save_path)
    print(f"Quant step: {best['q']}")
    print(f"Achieved bitrate (bpp): {best['bpp']:.4f}")
    print(f"Achieved ratio: {achieved_ratio:.2f}x")
    print(f"MSE: {best['mse']:.4f}")
    print(f"PSNR: {best['psnr']:.2f} dB")

    return save_path


def find_default_image():
    """Try common locations for `100007` in the repo."""
    base = os.path.dirname(__file__)
    candidates = [
        os.path.join(base, 'images', 'test', '100007.png'),
        os.path.join(base, 'images', 'test', '100007.jpg'),
        os.path.join(os.getcwd(), 'wavelet', 'images', 'test', '100007.png'),
        os.path.join(os.getcwd(), 'wavelet', 'images', 'test', '100007.jpg'),
        os.path.join(os.getcwd(), 'images', 'test', '100007.png'),
        os.path.join(os.getcwd(), 'images', 'test', '100007.jpg'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compress an image to a target compression ratio using wavelet quantization sweep')
    parser.add_argument('-i', '--image', type=str, default=None, help='Path to input image (grayscale will be used)')
    parser.add_argument('-r', '--ratio', type=float, required=True, help='Desired compression ratio (e.g., 5, 10, 20)')
    parser.add_argument('-o', '--outdir', type=str, default='./saved_images', help='Output directory')
    args = parser.parse_args()

    img_path = args.image
    if img_path is None:
        img_path = find_default_image()
        if img_path is None:
            print('No default image found. Please provide --image path.')
            sys.exit(1)

    try:
        compress_and_save(img_path, args.ratio, args.outdir)
    except Exception as e:
        print('Error:', e)
        sys.exit(1)
