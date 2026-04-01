"""
Test script to verify the inplace operation fix for HexPlane deformation field.
This script tests the gradient computation with multiple subset iterations.
"""

import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mgs.deformation import HexPlaneField, DeformationModule, apply_deformation


def test_hexplane_gradient_no_inplace_error():
    """Test that HexPlane can be called multiple times without inplace errors."""
    print("[TEST] Testing HexPlane multiple calls...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hexplane = HexPlaneField(
        resolution=[64, 64, 64, 150],
        feature_dim=16,
        multires=[1, 2],
        device=device,
    ).to(device)
    
    # Simulate multiple subset iterations
    num_subsets = 4
    total_loss = 0.0
    
    for i in range(num_subsets):
        # Create random coords for this "subset"
        coords = torch.rand((100, 4), device=device) * 2 - 1
        coords.requires_grad_(True)
        
        # Query HexPlane
        features = hexplane(coords.unsqueeze(0))[0]
        
        # Compute a simple loss
        loss = features.sum()
        total_loss = total_loss + loss
        
        print(f"  Subset {i+1}/{num_subsets}: loss={loss.item():.4f}")
    
    # Backward pass - this should NOT raise an inplace error
    try:
        total_loss.backward()
        print("✓ SUCCESS: Backward pass completed without inplace error")
        return True
    except RuntimeError as e:
        if "inplace operation" in str(e):
            print(f"✗ FAILED: Inplace operation error detected: {e}")
            return False
        else:
            raise


def test_deformation_module_gradient():
    """Test that DeformationModule works correctly with HexPlane."""
    print("\n[TEST] Testing DeformationModule integration...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    hexplane = HexPlaneField(
        resolution=[64, 64, 64, 150],
        feature_dim=16,
        multires=[1, 2],
        device=device,
    ).to(device)
    
    deformation = DeformationModule(
        feature_dim=192,  # 16 * 6 * 2 (multires=[1,2]) = 192
        hidden_dim=256,
        num_layers=2,
        device=device,
    ).to(device)
    
    # Simulate forward pass with deformation
    coords = torch.rand((1000, 4), device=device) * 2 - 1
    coords.requires_grad_(True)
    
    features = hexplane(coords.unsqueeze(0))[0]
    dx, ds, dr, do, dsh = deformation(features)
    
    # Apply deformation to dummy Gaussians
    means = torch.randn((1000, 3), device=device, requires_grad=True)
    scales = torch.randn((1000, 3), device=device, requires_grad=True)
    quats = torch.randn((1000, 4), device=device, requires_grad=True)
    opacities = torch.randn((1000, 1), device=device, requires_grad=True)
    
    deformed_means, deformed_scales, deformed_quats, deformed_opacities, _ = apply_deformation(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh_coeffs=None,
        dx=dx,
        ds=ds,
        dr=dr,
        do=do,
        dsh=None,
    )
    
    # Compute loss
    loss = deformed_means.sum() + deformed_scales.sum() + deformed_quats.sum()
    
    try:
        loss.backward()
        print("✓ SUCCESS: DeformationModule backward pass completed")
        return True
    except RuntimeError as e:
        if "inplace operation" in str(e):
            print(f"✗ FAILED: Inplace operation error: {e}")
            return False
        else:
            raise


def test_regularization_in_loop():
    """Test the new regularization computation inside the loop."""
    print("\n[TEST] Testing regularization inside loop...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    hexplane = HexPlaneField(
        resolution=[64, 64, 64, 150],
        feature_dim=16,
        multires=[1, 2],
        device=device,
    ).to(device)
    
    num_subsets = 4
    total_loss = 0.0
    
    deformation_reg_weight = 1e-5
    deformation_time_smooth_weight = 1e-4
    
    for i in range(num_subsets):
        # Simulate main loss for this subset
        coords_main = torch.rand((100, 4), device=device) * 2 - 1
        features_main = hexplane(coords_main.unsqueeze(0))[0]
        main_loss = features_main.sum()
        
        # Add PlaneTV regularization
        num_samples = 100
        coords = torch.rand((num_samples, 4), device=device) * 2 - 1
        coords.requires_grad_(True)
        
        features = hexplane(coords.unsqueeze(0))[0]
        
        # PlaneTV: simplified first-order approximation (no second-order gradients)
        plane_tv_loss = features.abs().mean()
        total_loss = total_loss + main_loss + (deformation_reg_weight / num_subsets) * plane_tv_loss
        
        # Add time smoothness regularization
        num_samples = 50
        xyz = torch.rand((num_samples, 3), device=device) * 2 - 1
        t1 = torch.rand((num_samples, 1), device=device)
        t2 = t1 + 0.1
        
        coords1 = torch.cat([xyz, t1], dim=-1)
        coords2 = torch.cat([xyz, t2], dim=-1)
        
        features1 = hexplane(coords1.unsqueeze(0))[0]
        features2 = hexplane(coords2.unsqueeze(0))[0]
        
        time_smooth_loss = (features1 - features2).abs().mean()
        total_loss = total_loss + (deformation_time_smooth_weight / num_subsets) * time_smooth_loss
        
        print(f"  Subset {i+1}/{num_subsets}: total_loss={total_loss.item():.4f}")
    
    try:
        total_loss.backward()
        print("✓ SUCCESS: Loop regularization backward pass completed")
        return True
    except RuntimeError as e:
        if "inplace operation" in str(e):
            print(f"✗ FAILED: Inplace operation error: {e}")
            return False
        else:
            raise


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing HexPlane Inplace Operation Fix")
    print("=" * 60)
    
    results = []
    
    results.append(("HexPlane multiple calls", test_hexplane_gradient_no_inplace_error()))
    results.append(("DeformationModule integration", test_deformation_module_gradient()))
    results.append(("Regularization in loop", test_regularization_in_loop()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
