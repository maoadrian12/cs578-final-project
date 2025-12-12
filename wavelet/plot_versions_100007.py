import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Import the compression helper from the existing module in the same folder
# This file sits next to `wavelet.py` so `import wavelet` will import that module.
try:
    from wavelet import compress_measure_single
except Exception:
    # If run with a different working directory, try an alternative import
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from wavelet import compress_measure_single


def find_best_for_ratios(img_arr, ratios):
    """Return best compression results (quantization sweep) for each ratio."""
    original_bpp = 8.0
    results = {}
    for ratio in ratios:
        target_bpp = original_bpp / ratio
        best = {'diff': float('inf')}
        for q in np.arange(1.0, 200.0, 0.5):
            bpp, mse, psnr, recon = compress_measure_single(img_arr, q)
            diff = abs(bpp - target_bpp)
            if diff < best['diff']:
                best.update({
                    'diff': diff,
                    'q': q,
                    'bpp': bpp,
                    'mse': mse,
                    'psnr': psnr,
                    'recon': recon,
                })
        results[ratio] = best
    return results


def plot_and_save(image_path, outpath=None):
    img = Image.open(image_path).convert('L')
    img_arr = np.array(img)

    ratios = [5.0, 10.0, 20.0]
    results = find_best_for_ratios(img_arr, ratios)

    ncols = 1 + len(ratios)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))

    # Original
    axes[0].imshow(img_arr, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Compressed versions
    for i, ratio in enumerate(ratios, start=1):
        res = results[ratio]
        recon = np.clip(res['recon'], 0, 255).astype(np.uint8)
        axes[i].imshow(recon, cmap='gray', vmin=0, vmax=255)
        title = f"{ratio}x\nPSNR={res['psnr']:.2f} dB\nMSE={res['mse']:.2f}\nActual={8.0/res['bpp']:.2f}x"
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.tight_layout()

    if outpath is None:
        outdir = os.path.join(os.getcwd(), 'saved_images')
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, '100007_versions.png')
    else:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)

    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved figure to: {outpath}")


if __name__ == '__main__':
    # Try several likely locations for 100007 (jpg/png) depending on cwd
    candidates = [
        os.path.join(os.path.dirname(__file__), 'images', 'test', '100007.png'),
        os.path.join(os.path.dirname(__file__), 'images', 'test', '100007.jpg'),
        os.path.join(os.getcwd(), 'wavelet', 'images', 'test', '100007.png'),
        os.path.join(os.getcwd(), 'wavelet', 'images', 'test', '100007.jpg'),
        os.path.join(os.getcwd(), 'images', 'test', '100007.png'),
        os.path.join(os.getcwd(), 'images', 'test', '100007.jpg'),
    ]

    image_path = None
    for p in candidates:
        if os.path.exists(p):
            image_path = p
            break

    if image_path is None:
        print('Could not find 100007 image in expected locations. Please provide the path to the image.')
        print('Candidates tried:')
        for p in candidates:
            print(' -', p)
    else:
        outpath = os.path.join(os.getcwd(), 'saved_images', '100007_versions.png')
        plot_and_save(image_path, outpath)
