"""Phase 3 XAI: Grad-CAM on U-Net skip connections + PatchGAN explanation maps.

Usage:
    python scripts/xai_explain.py \
        --dataroot ./datasets/cityscapes \
        --name cityscapes_pix2pix_mc \
        --direction BtoA \
        --num_images 50 \
        --output_dir ./logs/xai_results

Produces per-image:
    - gradcam_skip{1..5}.png  : Grad-CAM heatmaps at 5 U-Net skip connection scales
    - patchgan_map.png         : PatchGAN spatial confidence map (30x30 upsampled to 256x256)
    - overlay.png              : Grad-CAM deepest skip overlaid on input segmentation
    - composite.png            : Side-by-side: input | output | gradcam | patchgan
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import util.util as util


# ---------------------------------------------------------------------------
# Grad-CAM engine
# ---------------------------------------------------------------------------

class GradCAM:
    """Grad-CAM using tensor-level hooks to avoid inplace op conflicts."""

    def __init__(self, model, target_layer):
        self.activations = None
        self.gradients = None
        self._grad_hook = None
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, input, output):
        # Clone to avoid inplace modification issues, register grad hook on tensor
        self.activations = output.clone()
        self._grad_hook = self.activations.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad.detach()

    def remove(self):
        self._fwd_hook.remove()
        if self._grad_hook is not None:
            self._grad_hook.remove()

    def compute(self, loss):
        """Call after forward pass. Backprop loss, then compute CAM."""
        loss.backward(retain_graph=True)
        if self.gradients is None or self.activations is None:
            return None
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations.detach()).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam_min = cam.flatten(1).min(1)[0].view(-1, 1, 1, 1)
        cam_max = cam.flatten(1).max(1)[0].view(-1, 1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tensor_to_rgb(t):
    """Convert [-1,1] tensor [1,C,H,W] -> uint8 numpy [H,W,3]."""
    arr = t[0].cpu().float().numpy()
    arr = (np.transpose(arr, (1, 2, 0)) + 1.0) / 2.0 * 255.0
    return arr.clip(0, 255).astype(np.uint8)


def cam_to_heatmap(cam_tensor, size=(256, 256)):
    """Convert CAM tensor [1,1,H,W] -> coloured heatmap uint8 [H,W,3]."""
    cam_np = cam_tensor[0, 0].cpu().numpy()
    cam_resized = np.array(Image.fromarray((cam_np * 255).astype(np.uint8)).resize(size, Image.BILINEAR))
    # Apply jet colormap manually
    r = np.clip(1.5 - np.abs(4 * cam_resized / 255.0 - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * cam_resized / 255.0 - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * cam_resized / 255.0 - 1), 0, 1)
    heatmap = (np.stack([r, g, b], axis=2) * 255).astype(np.uint8)
    return heatmap


def overlay_cam(image_rgb, cam_tensor, alpha=0.5):
    """Blend Grad-CAM heatmap over image."""
    heatmap = cam_to_heatmap(cam_tensor, size=(image_rgb.shape[1], image_rgb.shape[0]))
    return (alpha * image_rgb.astype(np.float32) + (1 - alpha) * heatmap.astype(np.float32)).clip(0, 255).astype(np.uint8)


def collect_skip_layers(netG):
    """Return list of downconv layers from U-Net skip connection blocks (outermost->innermost)."""
    layers = []
    for module in netG.modules():
        if module.__class__.__name__ == 'UnetSkipConnectionBlock':
            # model[0] is either downconv (non-outermost) or downconv for outermost
            sub = module.model
            if len(sub) > 0 and isinstance(sub[0], torch.nn.Conv2d):
                layers.append(sub[0])
    return layers  # ~8 layers, ordered outermost->innermost


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--direction', type=str, default='BtoA')
    parser.add_argument('--num_images', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='./logs/xai_results')
    parser.add_argument('--skip_layers', type=str, default='1,2,3,4,5',
                        help='Which skip layer indices to compute Grad-CAM for (1=outermost)')
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build options compatible with TestOptions
    sys.argv = [
        'xai_explain.py',
        '--dataroot', args.dataroot,
        '--name', args.name,
        '--model', 'pix2pix',
        '--direction', args.direction,
        '--dataset_mode', 'aligned',
        '--norm', 'batch',
        '--netG', 'unet_256',
        '--no_dropout',
        '--num_threads', '0',
        '--batch_size', '1',
        '--serial_batches',
    ]
    opt = TestOptions().parse()
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt.eval = False  # keep BatchNorm in train mode for Grad-CAM stability

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)

    # Unwrap DDP if needed
    netG = model.netG.module if hasattr(model.netG, 'module') else model.netG
    device = next(netG.parameters()).device

    # Load discriminator separately for PatchGAN maps
    from models import networks as nets
    netD = nets.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                         opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)
    D_path = f'checkpoints/{opt.name}/latest_net_D.pth'
    netD.load_state_dict(torch.load(D_path, map_location=device))
    netD = netD.to(device).eval()

    # Put in train mode so BatchNorm stats are live (better Grad-CAM)
    netG.train()
    netD.eval()

    # Collect skip connection layers
    skip_layers = collect_skip_layers(netG)
    selected_indices = [int(i) - 1 for i in args.skip_layers.split(',')]
    selected_layers = [skip_layers[i] for i in selected_indices if i < len(skip_layers)]

    print(f"Found {len(skip_layers)} skip layers, using indices: {args.skip_layers}")
    print(f"Saving XAI results to: {out_dir}")

    results_summary = []

    for idx, data in enumerate(dataset):
        if idx >= args.num_images:
            break

        model.set_input(data)
        real_A = model.real_A  # [1, 3, 256, 256]
        real_B = model.real_B
        img_name = Path(model.get_image_paths()[0]).stem

        img_out_dir = out_dir / img_name
        img_out_dir.mkdir(exist_ok=True)

        # ---- Grad-CAM on skip connections ----
        cams = {}
        for layer_idx, layer in zip(selected_indices, selected_layers):
            real_A_req = real_A.clone().requires_grad_(True)

            # Attach Grad-CAM hooks
            gcam = GradCAM(netG, layer)

            # Forward pass
            fake_B = netG(real_A_req)

            # Loss: L1 reconstruction (drives gradients back through generator)
            loss = F.l1_loss(fake_B, real_B)
            netG.zero_grad()
            cam = gcam.compute(loss)
            gcam.remove()

            if cam is not None:
                cams[layer_idx + 1] = cam  # 1-indexed
                heatmap = cam_to_heatmap(cam)
                Image.fromarray(heatmap).save(img_out_dir / f'gradcam_skip{layer_idx + 1}.png')

        # ---- PatchGAN explanation map ----
        netG.eval()
        with torch.no_grad():
            fake_B_clean = netG(real_A)

        fake_AB = torch.cat([real_A, fake_B_clean], dim=1)
        real_AB = torch.cat([real_A, real_B], dim=1)

        with torch.no_grad():
            patch_fake = torch.sigmoid(netD(fake_AB))  # [1, 1, 30, 30] -> [0,1]
            patch_real = torch.sigmoid(netD(real_AB))

        # Upsample to 256x256
        patch_fake_up = F.interpolate(patch_fake, size=(256, 256), mode='bilinear', align_corners=False)
        patch_real_up = F.interpolate(patch_real, size=(256, 256), mode='bilinear', align_corners=False)

        patchgan_heatmap = cam_to_heatmap(patch_fake_up)
        Image.fromarray(patchgan_heatmap).save(img_out_dir / 'patchgan_map.png')

        # ---- Overlay: deepest selected skip on real_A ----
        real_A_rgb = tensor_to_rgb(real_A)
        if cams:
            deepest_cam = cams[max(cams.keys())]
            overlay = overlay_cam(real_A_rgb, deepest_cam, alpha=0.6)
            Image.fromarray(overlay).save(img_out_dir / 'overlay.png')

        # ---- Composite: input | output | gradcam_deepest | patchgan ----
        fake_B_rgb = tensor_to_rgb(fake_B_clean)
        real_B_rgb = tensor_to_rgb(real_B)
        if cams:
            gcam_rgb = cam_to_heatmap(cams[max(cams.keys())])
        else:
            gcam_rgb = np.zeros_like(real_A_rgb)

        composite = np.concatenate([real_A_rgb, fake_B_rgb, real_B_rgb, gcam_rgb, patchgan_heatmap], axis=1)
        Image.fromarray(composite).save(img_out_dir / 'composite.png')

        # Per-image stats
        mean_conf_fake = patch_fake.mean().item()
        mean_conf_real = patch_real.mean().item()
        results_summary.append({
            'image': img_name,
            'patchgan_fake_conf': mean_conf_fake,
            'patchgan_real_conf': mean_conf_real,
            'fooling_ratio': mean_conf_fake / (mean_conf_real + 1e-8),
        })

        if (idx + 1) % 10 == 0:
            print(f"[{idx+1}/{args.num_images}] {img_name} | D(fake)={mean_conf_fake:.3f} D(real)={mean_conf_real:.3f}")

        netG.train()  # restore for next iter

    # ---- Save summary ----
    summary_path = out_dir / 'xai_summary.txt'
    with open(summary_path, 'w') as f:
        f.write('XAI Phase 3 Summary\n')
        f.write('=' * 50 + '\n')
        f.write(f'Images analyzed: {len(results_summary)}\n')
        if results_summary:
            mean_fool = np.mean([r['fooling_ratio'] for r in results_summary])
            mean_fake_conf = np.mean([r['patchgan_fake_conf'] for r in results_summary])
            mean_real_conf = np.mean([r['patchgan_real_conf'] for r in results_summary])
            f.write(f'Mean D(G(A)) confidence: {mean_fake_conf:.4f}\n')
            f.write(f'Mean D(real) confidence: {mean_real_conf:.4f}\n')
            f.write(f'Mean fooling ratio: {mean_fool:.4f}\n')
            f.write('\nPer-image results:\n')
            for r in results_summary:
                f.write(f"  {r['image']}: fake={r['patchgan_fake_conf']:.3f} real={r['patchgan_real_conf']:.3f}\n")

    print(f"\nDone. Summary saved to {summary_path}")


if __name__ == '__main__':
    main()
