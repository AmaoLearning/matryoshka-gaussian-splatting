#!/usr/bin/env python3
"""Evaluate a trained MGS checkpoint at multiple operating points.

Reports PSNR, SSIM, LPIPS, and FPS for each Gaussian budget ratio.
Evaluation is split into two passes:
  1. FPS pass: rasterization only, no tqdm, no saving, no metrics.
  2. Metrics and save pass: rasterize once per frame; compute PSNR/SSIM/LPIPS and save images (if --save_images).
FPS is measured with torch.cuda.synchronize(), excluding the first 3 frames.

Usage:
    python eval.py \
        --ckpt ../checkpoint/bicycle/ckpts/ckpt_49999_rank0.pt \
        --data_dir ../benchmark/MipNeRF360/360_v2/bicycle \
        --data_factor 4 \
        --output_dir ../prediction-image/bicycle

    # Custom operating points
    python eval.py \
        --ckpt ../checkpoint/bicycle/ckpts/ckpt_49999_rank0.pt \
        --data_dir ../benchmark/MipNeRF360/360_v2/bicycle \
        --data_factor 4 \
        --output_dir ../prediction-image/bicycle \
        --ratios 1.0 0.5 0.1
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import imageio
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from datasets.auto import build_parser_and_datasets
from gsplat.rendering import rasterization
from mgs.sorting import SplatSorter
from mgs.deformation import apply_deformation, HexPlaneField, DeformationModule


PAPER_RATIOS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MGS checkpoint at multiple operating points"
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to scene data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--data_factor", type=int, default=4, help="Image downsample factor")
    parser.add_argument("--sh_degree", type=int, default=3, help="SH degree")
    parser.add_argument("--test_every", type=int, default=8, help="Every N-th image is test")
    parser.add_argument(
        "--ratios", type=float, nargs="+", default=PAPER_RATIOS,
        help="Gaussian budget ratios to evaluate (default: paper operating points)",
    )
    parser.add_argument("--save_images", action="store_true", help="Save rendered images")
    parser.add_argument("--sort_strategy", type=str, default="by_opacity_descending",
                        help="Sorting strategy for prefix ordering")
    parser.add_argument("--enable_deformation", action="store_true", 
                        help="Enable deformation module for dynamic scenes")
    parser.add_argument("--deformation_resolution", type=int, nargs="+", default=[64, 64, 64, 150],
                        help="HexPlane resolution [res_x, res_y, res_z, res_t]")
    parser.add_argument("--deformation_feature_dim", type=int, default=16,
                        help="HexPlane feature dimension")
    parser.add_argument("--deformation_multires", type=int, nargs="+", default=[1, 2],
                        help="Multi-resolution levels")
    return parser.parse_args()


def load_checkpoint(path, device):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    if "splats" in ckpt:
        splats = ckpt["splats"]
    else:
        splats = ckpt
    return splats, ckpt.get("step", -1)


def build_subset(splats, subset_indices, hexplane=None, deformation_module=None, time_coord=None):
    """Pre-slice and activate splat tensors for a given subset.

    Applies exp/sigmoid/cat once per ratio so the per-frame timing loop
    measures only the rasterization kernel.
    
    Args:
        splats: ParameterDict of all splat parameters
        subset_indices: Indices of splats to include
        hexplane: HexPlaneField module (optional, for deformation)
        deformation_module: DeformationModule (optional, for deformation)
        time_coord: Time coordinate scalar (optional, for deformation)
    """
    idx = subset_indices
    subset = {
        "means": torch.index_select(splats["means"], 0, idx),
        "quats": torch.index_select(splats["quats"], 0, idx),
        "scales": torch.exp(torch.index_select(splats["scales"], 0, idx)),
        "opacities": torch.sigmoid(torch.index_select(splats["opacities"], 0, idx)),
        "colors": torch.cat(
            [torch.index_select(splats["sh0"], 0, idx),
             torch.index_select(splats["shN"], 0, idx)],
            dim=1,
        ),
    }
    
    # Apply deformation if modules are provided
    if hexplane is not None and deformation_module is not None and time_coord is not None:
        N = subset["means"].shape[0]
        
        # Normalize coordinates for HexPlane
        coords = torch.cat([
            subset["means"].unsqueeze(0),  # (1, N, 3)
            torch.full((1, N, 1), time_coord, device=subset["means"].device)  # (1, N, 1)
        ], dim=-1)  # (1, N, 4)
        
        # Normalize spatial coordinates
        coords[..., :3] = coords[..., :3] / 2.0  # Simplified normalization
        
        # Query HexPlane and get deformation
        features = hexplane(coords)[0]  # (N, feature_dim)
        dx, ds, dr, do, dsh = deformation_module(features)
        
        # Apply deformation
        deformed = apply_deformation(
            means=subset["means"],
            scales=torch.log(subset["scales"]),  # Convert back to log space
            quats=subset["quats"],
            opacities=torch.logit(subset["opacities"]),  # Convert back to logit space
            sh_coeffs=subset["colors"][:, 1:, :],  # Higher order SH
            dx=dx,
            ds=ds,
            dr=dr,
            do=do,
            dsh=dsh,
        )
        
        subset["means"], subset["scales"], subset["quats"], subset["opacities"], sh_deformed = deformed
        subset["scales"] = torch.exp(subset["scales"])  # Exp for rasterization
        subset["opacities"] = torch.sigmoid(subset["opacities"])  # Sigmoid for rasterization
        if sh_deformed is not None:
            subset["colors"] = torch.cat([subset["colors"][:, :1, :], sh_deformed], dim=1)
    
    return subset


def rasterize_subset(subset, camtoworlds, Ks, width, height, sh_degree, packed=False):
    render_colors, render_alphas, info = rasterization(
        means=subset["means"],
        quats=subset["quats"],
        scales=subset["scales"],
        opacities=subset["opacities"],
        colors=subset["colors"],
        viewmats=torch.linalg.inv(camtoworlds),
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=sh_degree,
        packed=packed,
    )
    return render_colors, render_alphas, info


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.ckpt}")
    splats_data, step = load_checkpoint(args.ckpt, device)

    splats = torch.nn.ParameterDict()
    for k, v in splats_data.items():
        splats[k] = torch.nn.Parameter(v.to(device), requires_grad=False)

    total_splats = len(splats["means"])
    print(f"Checkpoint: {total_splats:,} Gaussians, step {step}")

    # Load deformation module if enabled
    hexplane = None
    deformation_module = None
    if args.enable_deformation:
        print("Loading deformation module...")
        ckpt_full, _ = load_checkpoint(args.ckpt, device)
        
        if "hexplane" in ckpt_full:
            hexplane = HexPlaneField(
                resolution=args.deformation_resolution,
                feature_dim=args.deformation_feature_dim,
                multires=args.deformation_multires,
                device=str(device),
            ).to(device)
            hexplane.load_state_dict(ckpt_full["hexplane"])
            hexplane.eval()
            
            # Compute feature dimension: 6 planes * feature_dim * len(multires)
            feature_dim_total = 6 * args.deformation_feature_dim * len(args.deformation_multires)
            deformation_module = DeformationModule(
                feature_dim=feature_dim_total,
                device=str(device),
            ).to(device)
            deformation_module.load_state_dict(ckpt_full["deformation_module"])
            deformation_module.eval()
            
            print(f"Deformation module loaded: HexPlane + MLP")
        else:
            print("WARNING: Checkpoint does not contain deformation module, but --enable_deformation is set")

    print(f"Loading dataset: {args.data_dir}")
    parser_obj, _, valset = build_parser_and_datasets(
        data_dir=args.data_dir,
        factor=args.data_factor,
        normalize=True,
        test_every=args.test_every,
        patch_size=None,
        load_depths=False,
    )
    dataloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=1)
    print(f"Test set: {len(valset)} images")

    sorter = SplatSorter(strategy=args.sort_strategy)
    sort_indices = sorter.argsort(splats)

    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)

    ratios = sorted(args.ratios, reverse=True)
    all_results = {}

    # --- Pass 1: FPS only (no tqdm, no save, no metrics) ---
    print("Pass 1: FPS evaluation (rasterization only)...")
    with torch.no_grad():
        for ratio in ratios:
            n_keep = max(1, min(total_splats, int(round(ratio * total_splats))))
            subset_idx = sort_indices[:n_keep].to(device)
            
            # For FPS pass, use time=0 or average over time
            time_coord = 0.0
            subset = build_subset(splats, subset_idx, hexplane, deformation_module, time_coord=time_coord)
            times = []
            for i, data in enumerate(dataloader):
                camtoworlds = data["camtoworld"].to(device)
                Ks = data["K"].to(device)
                height, width = data["image"].shape[1:3]
                
                # Get time coordinate from data if available
                if hexplane is not None and "time" in data:
                    time_coord_i = data["time"].item() if "time" in data else data.get("frame_id", 0.0)
                    subset = build_subset(splats, subset_idx, hexplane, deformation_module, time_coord=time_coord_i)

                torch.cuda.synchronize()
                tic = time.time()
                rasterize_subset(
                    subset, camtoworlds, Ks, width, height, args.sh_degree
                )
                torch.cuda.synchronize()
                elapsed = max(time.time() - tic, 1e-10)
                times.append(elapsed)

            if len(times) > 3:
                avg_time = float(np.mean(times[3:]))
            else:
                avg_time = float(np.mean(times)) if times else 0.0
            fps = 1.0 / avg_time if avg_time > 0 else 0.0
            all_results[f"{ratio:.2f}"] = {
                "ratio": ratio,
                "num_gaussians": n_keep,
                "fps": fps,
                "time_per_image": avg_time,
                "num_test_images": len(valset),
            }
            print(f"  ratio={ratio:.2f} | GS={n_keep:>9,} | FPS={fps:.1f}")

    # --- Pass 2: Metrics and save images ---
    print("Pass 2: Metrics and save images...")
    with torch.no_grad():
        for ratio in ratios:
            n_keep = max(1, min(total_splats, int(round(ratio * total_splats))))
            subset_idx = sort_indices[:n_keep].to(device)
            metrics = defaultdict(list)
            if args.save_images:
                img_dir = output_dir / f"ratio_{ratio:.2f}"
                img_dir.mkdir(exist_ok=True)
            for i, data in enumerate(tqdm.tqdm(dataloader, desc=f"ratio={ratio:.2f}")):
                camtoworlds = data["camtoworld"].to(device)
                Ks = data["K"].to(device)
                pixels = data["image"].to(device) / 255.0
                height, width = pixels.shape[1:3]
                
                # Get time coordinate for this frame
                time_coord = 0.0
                if hexplane is not None:
                    if "time" in data:
                        time_coord = data["time"].item()
                    elif "frame_id" in data:
                        time_coord = data["frame_id"].item()
                
                # Build subset with deformation for this frame's time
                subset = build_subset(splats, subset_idx, hexplane, deformation_module, time_coord=time_coord)
                
                render_colors, _, _ = rasterize_subset(
                    subset, camtoworlds, Ks, width, height, args.sh_degree
                )
                colors = torch.clamp(render_colors[..., :3], 0.0, 1.0)
                if args.save_images:
                    pred = (colors.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                    imageio.imwrite(str(img_dir / f"{i:04d}.png"), pred)
                gt = pixels.permute(0, 3, 1, 2)
                pred_p = colors.permute(0, 3, 1, 2)
                metrics["psnr"].append(psnr_fn(pred_p, gt).item())
                metrics["ssim"].append(ssim_fn(pred_p, gt).item())
                metrics["lpips"].append(lpips_fn(pred_p, gt).item())
            all_results[f"{ratio:.2f}"].update({
                "psnr": float(np.mean(metrics["psnr"])),
                "ssim": float(np.mean(metrics["ssim"])),
                "lpips": float(np.mean(metrics["lpips"])),
            })
            r = all_results[f"{ratio:.2f}"]
            print(
                f"  ratio={ratio:.2f} | GS={r['num_gaussians']:>9,} | "
                f"PSNR={r['psnr']:.2f} | SSIM={r['ssim']:.4f} | "
                f"LPIPS={r['lpips']:.4f} | FPS={r['fps']:.1f}"
            )

    summary_path = output_dir / "eval_results.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
