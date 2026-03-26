"""Phase 2 evaluation: compute pairwise LPIPS diversity across MC-dropout samples.

Usage:
    python scripts/evaluate_diversity.py --results_dir results/cityscapes_pix2pix_mc/test_latest

Expects results_dir/images/ with files like:
    0_fake_B_s0.png, 0_fake_B_s1.png, ..., 0_fake_B_sN.png
"""
import argparse
import re
from pathlib import Path
from itertools import combinations
import numpy as np
import torch
import lpips
from PIL import Image


def load_tensor(path, device):
    """Load image as tensor in [-1, 1] for LPIPS."""
    img = Image.open(path).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0
    img = img * 2 - 1  # [0,1] -> [-1,1]
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def main():
    parser = argparse.ArgumentParser(description='Evaluate MC-dropout diversity via pairwise LPIPS')
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    images_dir = Path(args.results_dir) / 'images'
    if not images_dir.exists():
        images_dir = Path(args.results_dir)

    # Find all MC sample files: *_fake_B_s*.png
    sample_files = sorted(images_dir.glob('*_fake_B_s*.png'))
    if not sample_files:
        sample_files = sorted(images_dir.glob('*_fake_B_s*.jpg'))
    if not sample_files:
        print("No MC-dropout sample files found (expected *_fake_B_s*.png)")
        return

    # Group by image index
    groups = {}
    for f in sample_files:
        match = re.match(r'(\d+)_fake_B_s(\d+)', f.stem)
        if match:
            img_idx = int(match.group(1))
            groups.setdefault(img_idx, []).append(f)

    print(f"Found {len(groups)} images with MC-dropout samples")
    n_samples_per = [len(v) for v in groups.values()]
    print(f"Samples per image: min={min(n_samples_per)}, max={max(n_samples_per)}")

    # Compute pairwise LPIPS
    device = args.device
    lpips_fn = lpips.LPIPS(net='vgg', verbose=False).to(device).eval()

    all_distances = []
    for img_idx in sorted(groups.keys()):
        paths = sorted(groups[img_idx])
        if len(paths) < 2:
            continue
        tensors = [load_tensor(p, device) for p in paths]
        for (i, t1), (j, t2) in combinations(enumerate(tensors), 2):
            with torch.no_grad():
                dist = lpips_fn(t1, t2).item()
            all_distances.append(dist)

    if not all_distances:
        print("Need at least 2 samples per image to compute diversity.")
        return

    mean_lpips = np.mean(all_distances)
    std_lpips = np.std(all_distances)
    print(f"\n--- MC-Dropout Diversity Results ---")
    print(f"Pairwise LPIPS: {mean_lpips:.4f} +/- {std_lpips:.4f}  (n={len(all_distances)} pairs)")
    print(f"LPIPS > 0.08:   {'PASS' if mean_lpips > 0.08 else 'FAIL'} ({mean_lpips:.4f})")

    # Save results
    log_path = Path('./logs/phase2_diversity.txt')
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, 'w') as f:
        f.write("Phase 2 MC-Dropout Diversity Metrics\n")
        f.write("=" * 40 + "\n")
        f.write(f"Pairwise LPIPS: {mean_lpips:.4f} +/- {std_lpips:.4f}\n")
        f.write(f"Num pairs: {len(all_distances)}\n")
        f.write(f"Num images: {len(groups)}\n")
        f.write(f"Samples per image: {n_samples_per[0]}\n")
    print(f"Results saved to {log_path}")


if __name__ == '__main__':
    main()
