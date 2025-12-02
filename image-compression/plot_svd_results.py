import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = "svd_results_grayscale.csv"
GROUPED_CSV_PATH = "svd_results_grayscale_grouped.csv"

def main():
    df = pd.read_csv(CSV_PATH)

    grouped = (
        df.groupby("pct_rank")
          .agg(
              # Compression ratio stats
              mean_cr=("compression_ratio", "mean"),
              min_cr=("compression_ratio", "min"),
              q1_cr=("compression_ratio", lambda x: x.quantile(0.25)),
              q3_cr=("compression_ratio", lambda x: x.quantile(0.75)),
              max_cr=("compression_ratio", "max"),

              # MSE stats
              mean_mse=("mse", "mean"),
              min_mse=("mse", "min"),
              q1_mse=("mse", lambda x: x.quantile(0.25)),
              q3_mse=("mse", lambda x: x.quantile(0.75)),
              max_mse=("mse", "max"),

              # PSNR stats
              mean_psnr=("psnr", "mean"),
              min_psnr=("psnr", "min"),
              q1_psnr=("psnr", lambda x: x.quantile(0.25)),
              q3_psnr=("psnr", lambda x: x.quantile(0.75)),
              max_psnr=("psnr", "max"),
          )
          .reset_index()
          .sort_values("pct_rank")
    )

    print(grouped)

    # 🔹 Save grouped stats
    grouped.to_csv(GROUPED_CSV_PATH, index=False)
    print(f"Saved grouped stats to {GROUPED_CSV_PATH}")

    x = grouped["pct_rank"].values

    fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)

    # --- plotting code stays the same ---
    # 1) Compression ratio
    ax = axes[0]
    ax.fill_between(x, grouped["q1_cr"], grouped["q3_cr"], alpha=0.2, label="IQR (25th–75th)")
    ax.plot(x, grouped["mean_cr"], marker="o", label="Mean")
    ax.plot(x, grouped["min_cr"], linestyle="--", marker="v", label="Min")
    ax.plot(x, grouped["max_cr"], linestyle="--", marker="^", label="Max")
    ax.set_ylabel("Compression ratio")
    ax.legend(loc="best")

    # 2) MSE
    ax = axes[1]
    ax.fill_between(x, grouped["q1_mse"], grouped["q3_mse"], alpha=0.2, label="IQR (25th–75th)")
    ax.plot(x, grouped["mean_mse"], marker="o", label="Mean")
    ax.plot(x, grouped["min_mse"], linestyle="--", marker="v", label="Min")
    ax.plot(x, grouped["max_mse"], linestyle="--", marker="^", label="Max")
    ax.set_ylabel("MSE")
    ax.legend(loc="best")

    # 3) PSNR
    ax = axes[2]
    ax.fill_between(x, grouped["q1_psnr"], grouped["q3_psnr"], alpha=0.2, label="IQR (25th–75th)")
    ax.plot(x, grouped["mean_psnr"], marker="o", label="Mean")
    ax.plot(x, grouped["min_psnr"], linestyle="--", marker="v", label="Min")
    ax.plot(x, grouped["max_psnr"], linestyle="--", marker="^", label="Max")
    ax.set_ylabel("PSNR (dB)")
    ax.set_xlabel("% rank (k / min(H, W) × 100)")
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig("svd_results_summary.png", dpi=600, bbox_inches="tight", transparent=True)
    plt.show()


if __name__ == "__main__":
    main()
