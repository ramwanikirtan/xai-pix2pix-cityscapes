"""Print U-Net generator layer index map for Grad-CAM targeting (Phase 3).

Saves the full named_modules() listing to a text file for reference.
This identifies the skip connection layers that will be hooked in Phase 3.
"""
import sys
sys.path.insert(0, '.')

import torch
from models.networks import UnetGenerator, NLayerDiscriminator, get_norm_layer

def print_unet_layers():
    norm_layer = get_norm_layer(norm_type='batch')

    # Create U-Net 256 generator (same as pix2pix default)
    G = UnetGenerator(
        input_nc=3, output_nc=3, num_downs=8,
        ngf=64, norm_layer=norm_layer, use_dropout=True
    )

    print("=" * 70)
    print("U-NET GENERATOR — FULL LAYER MAP")
    print("=" * 70)

    lines = []
    for name, module in G.named_modules():
        line = f"{name:60s} | {module.__class__.__name__}"
        lines.append(line)
        print(line)

    print("\n" + "=" * 70)
    print("SKIP CONNECTION TARGET LAYERS FOR GRAD-CAM (Phase 3)")
    print("=" * 70)

    # The U-Net is built recursively. Each UnetSkipConnectionBlock has:
    #   - model[0] or model[0:2/3] = downsampling (encoder)
    #   - model[middle] = submodule (inner block)
    #   - model[-3:] or model[-2:] = upsampling (decoder)
    #
    # Skip connections happen in the forward() method:
    #   return torch.cat([x, self.model(x)], 1)
    #
    # To hook skip connections, we need to capture the OUTPUT of each
    # UnetSkipConnectionBlock's downconv (the encoder part).

    skip_info = []
    for name, module in G.named_modules():
        if module.__class__.__name__ == 'UnetSkipConnectionBlock':
            skip_info.append((name, module))

    for name, block in skip_info:
        is_outer = getattr(block, 'outermost', False)
        is_inner = hasattr(block, 'model') and not any(
            isinstance(m, type(block)) for m in block.model.children()
            if m is not block
        )
        label = " (outermost)" if is_outer else ""
        print(f"  {name:50s}{label}")

    print("\n" + "=" * 70)
    print("PATCHGAN DISCRIMINATOR — LAYER MAP")
    print("=" * 70)

    D = NLayerDiscriminator(input_nc=6, ndf=64, n_layers=3, norm_layer=norm_layer)
    for name, module in D.named_modules():
        line = f"{name:60s} | {module.__class__.__name__}"
        lines.append(line)
        print(line)

    # Verify discriminator output shape
    dummy_input = torch.randn(1, 6, 256, 256)
    with torch.no_grad():
        out = D(dummy_input)
    print(f"\nPatchGAN output shape for 256x256 input: {list(out.shape)}")
    print(f"Expected: [1, 1, 30, 30]")

    # Print VRAM estimate
    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"\nGenerator params: {g_params / 1e6:.2f}M")
    print(f"Discriminator params: {d_params / 1e6:.2f}M")
    print(f"Estimated fp16 model memory: {(g_params + d_params) * 2 / 1e9:.2f} GB")

    # Save to file
    output_path = './logs/unet_layer_map.txt'
    import os
    os.makedirs('./logs', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("U-NET GENERATOR — FULL LAYER MAP\n")
        f.write("=" * 70 + "\n")
        for name, module in G.named_modules():
            f.write(f"{name:60s} | {module.__class__.__name__}\n")
        f.write("\nPATCHGAN DISCRIMINATOR — LAYER MAP\n")
        f.write("=" * 70 + "\n")
        for name, module in D.named_modules():
            f.write(f"{name:60s} | {module.__class__.__name__}\n")
        f.write(f"\nPatchGAN output shape: {list(out.shape)}\n")
        f.write(f"Generator params: {g_params / 1e6:.2f}M\n")
        f.write(f"Discriminator params: {d_params / 1e6:.2f}M\n")

    print(f"\nLayer map saved to {output_path}")


if __name__ == '__main__':
    print_unet_layers()
