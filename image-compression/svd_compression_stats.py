from pathlib import Path
from typing import Iterable, List, Dict
import csv
import numpy as np
from PIL import Image


RAW_DIR = Path("data/raw")
RESULTS_CSV = Path("svd_results_grayscale.csv")
PCT_RANKS = [5,10,15,20,25,30,35,40,45,50,60]


def iter_image_paths(root, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
    """Yield all image paths under root with given extensions."""
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in exts: yield p


def load_image_gray(path):
    """
    Load an image as grayscale float32 in [0, 1].
    Shape: (H, W)
    """
    img = Image.open(path).convert("L")  # grayscale
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def mse_psnr(orig, recon, max_val=1.0):
    """
    Compute MSE and PSNR between two images (both assumed in [0, 1]).
    """
    diff = orig - recon
    mse = float(np.mean(diff ** 2, dtype=np.float64))
    if mse == 0.0:
        psnr = float("inf")
    else:
        psnr = float(10.0 * np.log10((max_val * max_val) / mse))
    return mse, psnr


def compression_ratio(height, width, k):
    """
    Compression ratio for rank-k SVD on a single grayscale image.

    Original: H * W numbers.
    SVD representation (rank k):
      U_k: H x k
      s_k: k singular values
      V_k^T: k x W
      -> total = k(H + W + 1)
    """
    n_orig = height * width
    n_svd = k * (height + width + 1)
    return float(n_orig) / float(n_svd)


def ranks_from_percentages(height, width, pct_ranks):
    """
    Convert %rank list into concrete k values based on max rank = min(H, W).
    """
    r_max = min(height, width)
    ks = set()
    for p in pct_ranks:
        k = int(round(r_max * (p / 100.0)))
        if k < 1:
            k = 1
        if k > r_max:
            k = r_max
        ks.add(k)
    return sorted(ks)


def reconstruct_from_svd(U: np.ndarray, s: np.ndarray, Vt: np.ndarray, k: int) -> np.ndarray:
    """
    Given full SVD (U, s, Vt) of shape (H, W),
    reconstruct rank-k approximation.
    """
    k_eff = min(k, len(s), U.shape[1], Vt.shape[0])
    U_k = U[:, :k_eff]
    s_k = s[:k_eff]
    Vt_k = Vt[:k_eff, :]

    # (U_k * s_k) @ Vt_k is equivalent to U_k @ diag(s_k) @ Vt_k
    A_k = (U_k * s_k) @ Vt_k
    return A_k


def run_svd_experiments_grayscale():
    rows = []

    image_paths = list(iter_image_paths(RAW_DIR))
    if not image_paths:
        raise RuntimeError(f"No images found under {RAW_DIR}")

    print(f"Found {len(image_paths)} images in {RAW_DIR}")

    for idx, img_path in enumerate(image_paths, start=1):
        print(f"[{idx}/{len(image_paths)}] {img_path.name}")

        # Load grayscale image in [0, 1]
        img = load_image_gray(img_path)   # (H, W)
        H, W = img.shape

        # Compute SVD once per image
        U, s, Vt = np.linalg.svd(img, full_matrices=False)

        # Decide k values from %rank
        ks = ranks_from_percentages(H, W, PCT_RANKS)

        for k in ks:
            recon = reconstruct_from_svd(U, s, Vt, k)
            # Clip just in case of tiny numerical overshoot
            recon = np.clip(recon, 0.0, 1.0)

            mse, psnr = mse_psnr(img, recon)
            cr = compression_ratio(H, W, k)
            pct_rank = 100.0 * k / float(min(H, W))

            rows.append({
                "image": img_path.name,
                "height": H,
                "width": W,
                "k": k,
                "pct_rank": pct_rank,
                "compression_ratio": cr,
                "mse": mse,
                "psnr": psnr,
            })

    # Save to CSV
    fieldnames = [
        "image", "height", "width",
        "k", "pct_rank", "compression_ratio", "mse", "psnr"
    ]

    with RESULTS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved results to {RESULTS_CSV}")


if __name__ == "__main__":
    run_svd_experiments_grayscale()