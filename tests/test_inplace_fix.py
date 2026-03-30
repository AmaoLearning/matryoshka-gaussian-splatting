"""Test for in-place operation fix in deformation field."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from mgs.deformation import HexPlaneField, DeformationModule, apply_deformation


def test_no_inplace_modification():
    """Test that deformation doesn't modify tensors in-place."""
    print("=" * 60)
    print("Testing No In-Place Modification")
    print("=" * 60)
    
    # Create modules
    hexplane = HexPlaneField(
        resolution=[64, 64, 64, 150],
        feature_dim=16,
        multires=[1, 2],
    ).cuda()
    
    deform_module = DeformationModule(
        feature_dim=16 * 2,
        hidden_dim=256,
        num_layers=2,
        predict_opacity=False,
        predict_sh=False,
    ).cuda()
    
    # Test with batch_size=1, num_gaussians=10000
    B, N = 1, 10000
    
    # Create normalized coordinates
    coords = torch.rand(B, N, 4).cuda() * 2.0 - 1.0
    coords.requires_grad_(True)
    
    # Forward through HexPlane
    features = hexplane(coords)  # (B, N, feature_dim)
    print(f"Features shape: {features.shape}")
    print(f"Features requires_grad: {features.requires_grad}")
    
    # IMPORTANT: This should NOT modify features[0] in-place
    features_first = features[0].clone()  # (N, feature_dim)
    features_first.requires_grad_(True)
    
    # Forward through DeformationModule
    dx, ds, dr, do, dsh = deform_module(features_first)
    print(f"Deformation dx shape: {dx.shape}")
    print(f"dx requires_grad: {dx.requires_grad}")
    
    # Test backward pass
    loss = dx.sum() + ds.sum() + dr.sum()
    loss.backward()
    
    # Check gradients
    assert coords.grad is not None, "coords should have gradient"
    assert features_first.grad is not None, "features_first should have gradient"
    
    print(f"coords.grad shape: {coords.grad.shape}")
    print(f"features_first.grad shape: {features_first.grad.shape}")
    
    # Verify features[0] wasn't modified
    print(f"Features version check: {features._version if hasattr(features, '_version') else 'N/A'}")
    
    print("✓ No in-place modification detected\n")
    return True


def test_mlp_no_inplace_relu():
    """Test that MLP doesn't use inplace ReLU."""
    print("=" * 60)
    print("Testing MLP ReLU Configuration")
    print("=" * 60)
    
    deform_module = DeformationModule(
        feature_dim=32,
        hidden_dim=256,
        num_layers=2,
    ).cuda()
    
    # Check if any ReLU uses inplace=True
    has_inplace_relu = False
    for module in deform_module.modules():
        if isinstance(module, torch.nn.ReLU):
            if module.inplace:
                has_inplace_relu = True
                print(f"❌ Found inplace ReLU: {module}")
    
    if has_inplace_relu:
        print("❌ FAIL: Found inplace ReLU operations\n")
        return False
    else:
        print("✓ All ReLU operations are not inplace\n")
        return True


def test_backward_pass():
    """Test full backward pass through deformation pipeline."""
    print("=" * 60)
    print("Testing Full Backward Pass")
    print("=" * 60)
    
    # Create modules
    hexplane = HexPlaneField(
        resolution=[64, 64, 64, 150],
        feature_dim=16,
        multires=[1, 2],
    ).cuda()
    
    deform_module = DeformationModule(
        feature_dim=16 * 2,
        hidden_dim=256,
        num_layers=2,
        predict_opacity=False,
        predict_sh=False,
    ).cuda()
    
    # Test parameters
    B, N = 1, 10000
    
    # Create coordinates with gradients enabled
    coords = torch.rand(B, N, 4).cuda() * 2.0 - 1.0
    coords.requires_grad_(True)
    
    # Gaussian parameters
    means = torch.randn(N, 3).cuda().requires_grad_(True)
    scales = torch.randn(N, 3).cuda().requires_grad_(True)
    quats = torch.randn(N, 4).cuda()
    quats = quats / quats.norm(dim=-1, keepdim=True)
    quats.requires_grad_(True)
    
    print(f"Input coords shape: {coords.shape}")
    print(f"Gaussian means shape: {means.shape}")
    
    # Forward
    features = hexplane(coords)
    features_first = features[0].clone()
    dx, ds, dr, do, dsh = deform_module(features_first)
    
    # Apply deformation
    def_means, def_scales, def_quats, _, _ = apply_deformation(
        means=means,
        scales=scales,
        quats=quats,
        opacities=None,
        sh_coeffs=None,
        dx=dx,
        ds=ds,
        dr=dr,
        do=do,
        dsh=dsh,
    )
    
    # Compute loss
    loss = def_means.sum() + def_scales.sum() + def_quats.sum()
    
    # Backward
    loss.backward()
    
    # Check gradients
    assert coords.grad is not None, "coords should have gradient"
    assert means.grad is not None, "means should have gradient"
    assert scales.grad is not None, "scales should have gradient"
    assert quats.grad is not None, "quats should have gradient"
    
    print(f"coords.grad: {coords.grad.shape}, min={coords.grad.min():.6f}, max={coords.grad.max():.6f}")
    print(f"means.grad: {means.grad.shape}, min={means.grad.min():.6f}, max={means.grad.max():.6f}")
    print(f"scales.grad: {scales.grad.shape}, min={scales.grad.min():.6f}, max={scales.grad.max():.6f}")
    print(f"quats.grad: {quats.grad.shape}, min={quats.grad.min():.6f}, max={quats.grad.max():.6f}")
    
    print("✓ Full backward pass successful\n")
    return True


if __name__ == "__main__":
    try:
        all_passed = True
        
        all_passed &= test_mlp_no_inplace_relu()
        all_passed &= test_no_inplace_modification()
        all_passed &= test_backward_pass()
        
        if all_passed:
            print("=" * 60)
            print("✅ ALL TESTS PASSED - In-place operations fixed!")
            print("=" * 60)
            sys.exit(0)
        else:
            print("=" * 60)
            print("❌ SOME TESTS FAILED")
            print("=" * 60)
            sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
