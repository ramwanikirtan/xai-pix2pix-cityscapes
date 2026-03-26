"""Phase 1 evaluation script: compute FID, SSIM, PSNR on generated vs real images.

Usage:
    python scripts/evaluate_metrics.py --results_dir results/cityscapes_pix2pix/test_latest

Expects:
    results_dir/images/ containing files like *_real_B.png and *_fake_B.png
"""
import argparse
import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def load_image(path):
    """Load image as numpy float32 array in [0, 1]."""
    img = Image.open(path).convert('RGB')
    return np.array(img).astype(np.float32) / 255.0


def compute_ssim_psnr(results_dir):
    """Compute SSIM and PSNR between real and fake images."""
    images_dir = Path(results_dir) / 'images'
    if not images_dir.exists():
        # Try without /images subfolder
        images_dir = Path(results_dir)

    # Find all real/fake pairs
    real_files = sorted(images_dir.glob('*_real_B.png'))
    if not real_files:
        real_files = sorted(images_dir.glob('*_real_B.jpg'))

    ssim_scores = []
    psnr_scores = []

    for real_path in real_files:
        # Derive fake path from real path
        fake_path = str(real_path).replace('_real_B', '_fake_B')
        if not os.path.exists(fake_path):
            print(f"Warning: no matching fake for {real_path}")
            continue

        real_img = load_image(real_path)
        fake_img = load_image(fake_path)

        # SSIM (multichannel)
        s = ssim(real_img, fake_img, channel_axis=2, data_range=1.0)
        ssim_scores.append(s)

        # PSNR
        p = psnr(real_img, fake_img, data_range=1.0)
        psnr_scores.append(p)

    return ssim_scores, psnr_scores


def main():
    parser = argparse.ArgumentParser(description='Evaluate FID, SSIM, PSNR')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to test results directory')
    parser.add_argument('--real_dir', type=str, default=None,
                        help='Path to real images for FID (optional, auto-detected)')
    parser.add_argument('--fake_dir', type=str, default=None,
                        help='Path to fake images for FID (optional, auto-detected)')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    print(f"Evaluating results in: {results_dir}")

    # --- SSIM and PSNR ---
    print("\n--- Computing SSIM and PSNR ---")
    ssim_scores, psnr_scores = compute_ssim_psnr(results_dir)

    if ssim_scores:
        mean_ssim = np.mean(ssim_scores)
        std_ssim = np.std(ssim_scores)
        mean_psnr = np.mean(psnr_scores)
        std_psnr = np.std(psnr_scores)

        print(f"SSIM:  {mean_ssim:.4f} +/- {std_ssim:.4f}  (n={len(ssim_scores)})")
        print(f"PSNR:  {mean_psnr:.2f} +/- {std_psnr:.2f} dB  (n={len(psnr_scores)})")

        # Check publishability targets
        print("\n--- Publishability Check ---")
        print(f"SSIM > 0.55:  {'PASS' if mean_ssim > 0.55 else 'FAIL'} ({mean_ssim:.4f})")
        print(f"PSNR > 18 dB: {'PASS' if mean_psnr > 18 else 'FAIL'} ({mean_psnr:.2f})")
    else:
        print("No image pairs found! Check results directory structure.")

    # --- FID ---
    print("\n--- Computing FID ---")
    print("Run FID separately with:")
    print(f"  python -m pytorch_fid <real_images_dir> <fake_images_dir>")

    # Try to compute FID if pytorch_fid is available
    try:
        from pytorch_fid import fid_score
        # Auto-detect real/fake directories
        images_dir = results_dir / 'images'
        if not images_dir.exists():
            images_dir = results_dir

        # Create temporary directories with only real/fake images
        import tempfile
        import shutil

        with tempfile.TemporaryDirectory() as tmpdir:
            real_tmp = Path(tmpdir) / 'real'
            fake_tmp = Path(tmpdir) / 'fake'
            real_tmp.mkdir()
            fake_tmp.mkdir()

            for f in images_dir.glob('*_real_B.*'):
                shutil.copy2(f, real_tmp / f.name)
            for f in images_dir.glob('*_fake_B.*'):
                shutil.copy2(f, fake_tmp / f.name)

            if list(real_tmp.iterdir()) and list(fake_tmp.iterdir()):
                fid_value = fid_score.calculate_fid_given_paths(
                    [str(real_tmp), str(fake_tmp)],
                    batch_size=50, device='cuda', dims=2048
                )
                print(f"FID:   {fid_value:.2f}")
                print(f"FID < 100: {'PASS' if fid_value < 100 else 'FAIL'} ({fid_value:.2f})")
            else:
                print("Could not auto-detect real/fake image directories for FID.")
    except ImportError:
        print("pytorch_fid not installed. Install with: pip install pytorch-fid")

    # Save results
    log_path = Path('./logs/phase1_metrics.txt')
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, 'w') as f:
        f.write("Phase 1 Baseline Metrics\n")
        f.write("=" * 40 + "\n")
        if ssim_scores:
            f.write(f"SSIM: {mean_ssim:.4f} +/- {std_ssim:.4f}\n")
            f.write(f"PSNR: {mean_psnr:.2f} +/- {std_psnr:.2f} dB\n")
            f.write(f"Num images: {len(ssim_scores)}\n")
    print(f"\nMetrics saved to {log_path}")


if __name__ == '__main__':
    main()
