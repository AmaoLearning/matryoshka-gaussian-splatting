#!/usr/bin/env python3
"""Quick validation script for 4DGS HexPlane integration.

This script tests:
1. HexPlane forward pass
2. DeformationModule forward pass
3. apply_deformation function
4. Integration with SimpleTrainer config
"""

import torch
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mgs.deformation import (
    HexPlaneField,
    DeformationModule,
    apply_deformation,
    rotation_6d_to_quaternion,
    quaternion_multiply,
)
from mgs.train.simple_trainer import Config


def test_hexplane():
    """Test HexPlaneField forward pass."""
    print("Testing HexPlaneField...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hexplane = HexPlaneField(
        resolution=[64, 64, 64, 150],
        feature_dim=16,
        multires=[1, 2],
        device=device,
    ).to(device)
    
    # Test with batch of 4D coordinates
    B, N = 1, 1000
    coords = torch.rand((B, N, 4), device=device) * 2 - 1  # [-1, 1]
    
    with torch.no_grad():
        features = hexplane(coords)
    
    # Expected: 6 planes * feature_dim * len(multires) = 6 * 16 * 2 = 192
    expected_dim = 6 * 16 * 2
    assert features.shape == (B, N, expected_dim), f"Expected {(B, N, expected_dim)}, got {features.shape}"
    print(f"  ✓ HexPlane forward: {features.shape}")
    
    # Test gradient flow
    coords.requires_grad_(True)
    features = hexplane(coords)
    loss = features.sum()
    loss.backward()
    assert coords.grad is not None, "Gradient should flow to coords"
    print(f"  ✓ HexPlane backward: gradient computed")
    
    print("✓ HexPlaneField test passed\n")


def test_deformation_module():
    """Test DeformationModule forward pass."""
    print("Testing DeformationModule...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Expected: 6 planes * feature_dim * len(multires) = 6 * 16 * 2 = 192
    feature_dim = 6 * 16 * 2
    
    deform_module = DeformationModule(
        feature_dim=feature_dim,
        hidden_dim=256,
        num_layers=2,
        predict_opacity=False,
        predict_sh=False,
        device=device,
    ).to(device)
    
    # Test with batch of features
    B, N = 1, 1000
    features = torch.rand((B, N, feature_dim), device=device)
    
    with torch.no_grad():
        dx, ds, dr, do, dsh = deform_module(features)
    
    assert dx.shape == (B, N, 3), f"Expected dx {(B, N, 3)}, got {dx.shape}"
    assert ds.shape == (B, N, 3), f"Expected ds {(B, N, 3)}, got {ds.shape}"
    assert dr.shape == (B, N, 6), f"Expected dr {(B, N, 6)}, got {dr.shape}"
    assert do is None, "do should be None when predict_opacity=False"
    assert dsh is None, "dsh should be None when predict_sh=False"
    print(f"  ✓ Deformation forward: dx={dx.shape}, ds={ds.shape}, dr={dr.shape}")
    
    # Test with opacity and SH prediction
    deform_module_full = DeformationModule(
        feature_dim=feature_dim,
        hidden_dim=256,
        num_layers=2,
        predict_opacity=True,
        predict_sh=True,
        sh_degree=3,
        device=device,
    ).to(device)
    
    with torch.no_grad():
        dx, ds, dr, do, dsh = deform_module_full(features)
    
    assert do.shape == (B, N, 1), f"Expected do {(B, N, 1)}, got {do.shape}"
    num_sh = (3 + 1) ** 2  # (sh_degree+1)^2
    assert dsh.shape == (B, N, num_sh * 3), f"Expected dsh {(B, N, num_sh*3)}, got {dsh.shape}"
    print(f"  ✓ Deformation full: do={do.shape}, dsh={dsh.shape}")
    
    print("✓ DeformationModule test passed\n")


def test_apply_deformation():
    """Test apply_deformation function."""
    print("Testing apply_deformation...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N = 1000
    
    # Create dummy Gaussian parameters
    means = torch.randn((N, 3), device=device)
    scales = torch.randn((N, 3), device=device)  # log space
    quats = torch.randn((N, 4), device=device)
    opacities = torch.randn((N, 1), device=device)  # logit space
    sh_coeffs = torch.randn((N, 16, 3), device=device)  # sh_degree=3 -> (3+1)^2=16
    
    # Create dummy deformation offsets
    dx = torch.randn((N, 3), device=device)
    ds = torch.randn((N, 3), device=device)
    dr = torch.randn((N, 6), device=device)
    do = torch.randn((N, 1), device=device)
    dsh = torch.randn((N, 16 * 3), device=device)
    
    # Apply deformation
    deformed_means, deformed_scales, deformed_quats, deformed_opacities, deformed_sh = apply_deformation(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh_coeffs=sh_coeffs,
        dx=dx,
        ds=ds,
        dr=dr,
        do=do,
        dsh=dsh,
    )
    
    assert deformed_means.shape == (N, 3), f"Expected {(N, 3)}, got {deformed_means.shape}"
    assert deformed_scales.shape == (N, 3), f"Expected {(N, 3)}, got {deformed_scales.shape}"
    assert deformed_quats.shape == (N, 4), f"Expected {(N, 4)}, got {deformed_quats.shape}"
    assert deformed_opacities.shape == (N, 1), f"Expected {(N, 1)}, got {deformed_opacities.shape}"
    assert deformed_sh.shape == (N, 16, 3), f"Expected {(N, 16, 3)}, got {deformed_sh.shape}"
    
    print(f"  ✓ apply_deformation: all parameters deformed correctly")
    
    # Test without optional parameters
    deformed2 = apply_deformation(
        means=means,
        scales=scales,
        quats=quats,
        opacities=None,
        sh_coeffs=None,
        dx=dx,
        ds=ds,
        dr=dr,
        do=None,
        dsh=None,
    )
    assert deformed2[3] is None, "opacities should be None when input is None"
    assert deformed2[4] is None, "sh should be None when input is None"
    print(f"  ✓ apply_deformation: optional parameters handled correctly")
    
    print("✓ apply_deformation test passed\n")


def test_rotation_utils():
    """Test rotation utility functions."""
    print("Testing rotation utilities...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N = 100
    
    # Test 6D to quaternion
    rot_6d = torch.randn((N, 6), device=device)
    quat = rotation_6d_to_quaternion(rot_6d)
    
    assert quat.shape == (N, 4), f"Expected {(N, 4)}, got {quat.shape}"
    quat_norm = torch.norm(quat, dim=-1)
    assert torch.allclose(quat_norm, torch.ones_like(quat_norm), atol=1e-5), "Quaternions should be unit"
    print(f"  ✓ rotation_6d_to_quaternion: unit quaternions")
    
    # Test quaternion multiplication
    q1 = torch.randn((N, 4), device=device)
    q1 = q1 / torch.norm(q1, dim=-1, keepdim=True)
    q2 = torch.randn((N, 4), device=device)
    q2 = q2 / torch.norm(q2, dim=-1, keepdim=True)
    
    q_mult = quaternion_multiply(q1, q2)
    assert q_mult.shape == (N, 4), f"Expected {(N, 4)}, got {q_mult.shape}"
    q_mult_norm = torch.norm(q_mult, dim=-1)
    assert torch.allclose(q_mult_norm, torch.ones_like(q_mult_norm), atol=1e-5), "Result should be unit"
    print(f"  ✓ quaternion_multiply: unit quaternions")
    
    print("✓ Rotation utilities test passed\n")


def test_config():
    """Test Config has deformation parameters."""
    print("Testing Config deformation parameters...")
    
    cfg = Config()
    
    # Check deformation parameters exist
    assert hasattr(cfg, "enable_deformation"), "Config missing enable_deformation"
    assert hasattr(cfg, "deformation_resolution"), "Config missing deformation_resolution"
    assert hasattr(cfg, "deformation_feature_dim"), "Config missing deformation_feature_dim"
    assert hasattr(cfg, "deformation_multires"), "Config missing deformation_multires"
    assert hasattr(cfg, "deformation_lr"), "Config missing deformation_lr"
    assert hasattr(cfg, "deformation_reg_weight"), "Config missing deformation_reg_weight"
    assert hasattr(cfg, "deformation_time_smooth_weight"), "Config missing deformation_time_smooth_weight"
    
    # Check default values
    assert cfg.enable_deformation == False, "Default should be False"
    assert cfg.deformation_resolution == [64, 64, 64, 150], f"Got {cfg.deformation_resolution}"
    assert cfg.deformation_feature_dim == 16, f"Got {cfg.deformation_feature_dim}"
    assert cfg.deformation_multires == [1, 2], f"Got {cfg.deformation_multires}"
    assert cfg.deformation_lr == 1.6e-4, f"Got {cfg.deformation_lr}"
    
    print(f"  ✓ Config has all deformation parameters with correct defaults")
    print("✓ Config test passed\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("4DGS HexPlane Integration Validation")
    print("=" * 60 + "\n")
    
    try:
        test_hexplane()
        test_deformation_module()
        test_apply_deformation()
        test_rotation_utils()
        test_config()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        return 0
    except Exception as e:
        print("=" * 60)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
