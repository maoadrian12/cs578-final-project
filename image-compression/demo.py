from pathlib import Path
import numpy as np
from PIL import Image


DEMO_DIR = Path("data/demo")
INPUT_PATH = DEMO_DIR / "demo.jpg"

# Output filenames (PNG to avoid extra JPEG artifacts)
OUT_GRAY = DEMO_DIR / "image_grayscale.png"
OUT_10   = DEMO_DIR / "image_10.png"
OUT_25   = DEMO_DIR / "image_25.png"
OUT_50   = DEMO_DIR / "image_50.png"

PCT_RANKS = [10, 25, 50]


def load_image_gray(path: Path) -> np.ndarray:
    """Load an image as grayscale float32 in [0,1], shape (H,W)."""
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def save_gray_image(array: np.ndarray, path: Path) -> None:
    """Save a grayscale float array in [0,1] to disk as uint8 PNG."""
    arr = np.clip(array, 0.0, 1.0) * 255.0
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    img.save(path)


def mse_psnr(orig: np.ndarray, recon: np.ndarray, max_val: float = 1.0):
    """Compute MSE and PSNR for two images in [0,1]."""
    diff = orig - recon
    mse = float(np.mean(diff**2, dtype=np.float64))
    if mse == 0.0:
        psnr = float("inf")
    else:
        psnr = float(10.0 * np.log10((max_val * max_val) / mse))
    return mse, psnr


def compression_ratio(H: int, W: int, k: int) -> float:
    """
    Compression ratio for grayscale rank-k SVD.

    Original: H*W numbers
    SVD: U_k (H x k), s_k (k), V_k^T (k x W) -> k(H + W + 1) numbers
    """
    n_orig = H * W
    n_svd = k * (H + W + 1)
    return float(n_orig) / float(n_svd)


def reconstruct_from_svd(U: np.ndarray, s: np.ndarray, Vt: np.ndarray, k: int):
    """Reconstruct rank-k approximation from full SVD."""
    k_eff = min(k, len(s), U.shape[1], Vt.shape[0])
    U_k = U[:, :k_eff]
    s_k = s[:k_eff]
    Vt_k = Vt[:k_eff, :]
    # (U_k * s_k) @ Vt_k == U_k @ diag(s_k) @ Vt_k, but faster
    A_k = (U_k * s_k) @ Vt_k
    return A_k


def main():
    # Load grayscale demo image
    A = load_image_gray(INPUT_PATH)  # shape (H,W), [0,1]
    H, W = A.shape

    # Save the pure grayscale version (just in case original wasn't grayscale)
    save_gray_image(A, OUT_GRAY)
    print(f"Saved grayscale image to {OUT_GRAY}")

    # Compute SVD once
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    r_max = min(H, W)

    # Header for results table
    print("\nResults for demo image:")
    print(f"{'k':>6} {'%rank':>8} {'CR':>10} {'MSE':>14} {'PSNR (dB)':>12}")
    print("-" * 54)

    for pct in PCT_RANKS:
        # Convert %rank to k
        k = int(round(r_max * (pct / 100.0)))
        k = max(1, min(k, r_max))

        # Reconstruct
        A_k = reconstruct_from_svd(U, s, Vt, k)
        A_k = np.clip(A_k, 0.0, 1.0)

        # Save image
        if pct == 10:
            out_path = OUT_10
        elif pct == 25:
            out_path = OUT_25
        elif pct == 50:
            out_path = OUT_50
        else:
            out_path = DEMO_DIR / f"image_{pct}.png"

        save_gray_image(A_k, out_path)

        # Metrics
        mse, psnr = mse_psnr(A, A_k)
        cr = compression_ratio(H, W, k)
        pct_rank = 100.0 * k / r_max

        print(f"{k:6d} {pct_rank:8.1f} {cr:10.3f} {mse:14.6e} {psnr:12.3f}")

    print("\nCompressed demo images saved in:", DEMO_DIR)


if __name__ == "__main__":
    main()
