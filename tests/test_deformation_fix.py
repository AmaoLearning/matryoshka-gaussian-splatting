"""Quick test for HexPlane deformation field fix."""

import torch
from mgs.deformation import HexPlaneField, DeformationModule, apply_deformation

def test_hexplane_forward():
    """Test HexPlane forward pass with correct dimensions."""
    print("=" * 60)
    print("Testing HexPlane Field Forward Pass")
    print("=" * 60)
    
    # Create HexPlane with default config
    hexplane = HexPlaneField(
        resolution=[64, 64, 64, 150],
        feature_dim=16,
        multires=[1, 2],
    ).cuda()
    
    # Test with batch_size=1, num_gaussians=100000
    B, N = 1, 100000
    
    # Create normalized coordinates in [-1, 1]
    coords = torch.rand(B, N, 4).cuda() * 2.0 - 1.0  # (B, N, 4) in [-1, 1]
    
    print(f"Input coords shape: {coords.shape}")
    print(f"Expected output shape: ({B}, {N}, {hexplane.feature_dim * len(hexplane.multires)})")
    
    # Forward pass
    with torch.cuda.amp.autocast(enabled=False):
        features = hexplane(coords)
    
    print(f"Output features shape: {features.shape}")
    
    # Verify shape
    expected_dim = hexplane.feature_dim * len(hexplane.multires)
    assert features.shape == (B, N, expected_dim), \
        f"Shape mismatch: expected ({B}, {N}, {expected_dim}), got {features.shape}"
    
    print("✓ HexPlane forward pass test PASSED\n")
    return True


def test_deformation_module():
    """Test DeformationModule forward pass."""
    print("=" * 60)
    print("Testing Deformation Module Forward Pass")
    print("=" * 60)
    
    # Create deformation module
    deform_module = DeformationModule(
        feature_dim=16 * 2,  # multires=[1, 2]
        hidden_dim=256,
        num_layers=2,
        predict_opacity=False,
        predict_sh=False,
    ).cuda()
    
    # Create dummy features
    B, N = 1, 100000
    features = torch.randn(B, N, 16 * 2).cuda()
    
    print(f"Input features shape: {features.shape}")
    
    # Forward pass (use first batch)
    with torch.cuda.amp.autocast(enabled=False):
        dx, ds, dr, do, dsh = deform_module(features[0])
    
    print(f"dx shape: {dx.shape}")  # (N, 3)
    print(f"ds shape: {ds.shape}")  # (N, 3)
    print(f"dr shape: {dr.shape}")  # (N, 6)
    print(f"do shape: {do.shape}")  # (N, 1) if enabled
    print(f"dsh shape: {dsh.shape}")  # (N, 48) if enabled
    
    # Verify shapes
    assert dx.shape == (N, 3), f"dx shape mismatch: expected ({N}, 3), got {dx.shape}"
    assert ds.shape == (N, 3), f"ds shape mismatch: expected ({N}, 3), got {ds.shape}"
    assert dr.shape == (N, 6), f"dr shape mismatch: expected ({N}, 6), got {dr.shape}"
    
    print("✓ Deformation module forward pass test PASSED\n")
    return True


def test_apply_deformation():
    """Test apply_deformation function."""
    print("=" * 60)
    print("Testing Apply Deformation Function")
    print("=" * 60)
    
    N = 100000
    
    # Create dummy Gaussian parameters
    means = torch.randn(N, 3).cuda()
    scales = torch.randn(N, 3).cuda()
    quats = torch.randn(N, 4).cuda()
    quats = quats / quats.norm(dim=-1, keepdim=True)  # Normalize
    opacities = torch.randn(N, 1).cuda()
    sh_coeffs = torch.randn(N, 8, 3).cuda()  # SH degree 2
    
    # Create deformation offsets
    dx = torch.randn(N, 3).cuda() * 0.1
    ds = torch.randn(N, 3).cuda() * 0.1
    dr = torch.randn(N, 6).cuda() * 0.01
    do = torch.randn(N, 1).cuda() * 0.01
    dsh = torch.randn(N, 8, 3).cuda() * 0.01
    
    print(f"Input means shape: {means.shape}")
    print(f"Deformation dx shape: {dx.shape}")
    
    # Apply deformation
    with torch.cuda.amp.autocast(enabled=False):
        deformed_means, deformed_scales, deformed_quats, deformed_opas, deformed_sh = \
            apply_deformation(
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
    
    print(f"Deformed means shape: {deformed_means.shape}")
    print(f"Deformed scales shape: {deformed_scales.shape}")
    print(f"Deformed quats shape: {deformed_quats.shape}")
    print(f"Deformed opacities shape: {deformed_opas.shape}")
    print(f"Deformed SH shape: {deformed_sh.shape if deformed_sh is not None else None}")
    
    # Verify shapes
    assert deformed_means.shape == means.shape
    assert deformed_scales.shape == scales.shape
    assert deformed_quats.shape == quats.shape
    assert deformed_opas.shape == opacities.shape
    if deformed_sh is not None:
        assert deformed_sh.shape == sh_coeffs.shape
    
    print("✓ Apply deformation test PASSED\n")
    return True


def test_full_pipeline():
    """Test full deformation pipeline."""
    print("=" * 60)
    print("Testing Full Deformation Pipeline")
    print("=" * 60)
    
    # Create HexPlane and deformation module
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
    B, N = 1, 100000
    
    # Create normalized coordinates
    coords = torch.rand(B, N, 4).cuda() * 2.0 - 1.0
    coords[..., :3] *= 0.5  # Scale down spatial coords
    
    # Gaussian parameters
    means = torch.randn(N, 3).cuda()
    scales = torch.randn(N, 3).cuda()
    quats = torch.randn(N, 4).cuda()
    quats = quats / quats.norm(dim=-1, keepdim=True)
    opacities = torch.randn(N, 1).cuda()
    
    print(f"HexPlane input: {coords.shape}")
    print(f"Gaussian means: {means.shape}")
    
    # Forward through HexPlane
    features = hexplane(coords)  # (B, N, feature_dim)
    print(f"HexPlane output: {features.shape}")
    
    # Get deformation from first batch
    dx, ds, dr, do, dsh = deform_module(features[0])
    print(f"Deformation offsets: dx={dx.shape}, ds={ds.shape}, dr={dr.shape}")
    
    # Apply deformation
    deformed = apply_deformation(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh_coeffs=None,
        dx=dx,
        ds=ds,
        dr=dr,
        do=do,
        dsh=dsh,
    )
    
    def_means, def_scales, def_quats, def_opas, _ = deformed
    print(f"Deformed means: {def_means.shape}")
    print(f"Deformed scales: {def_scales.shape}")
    
    # Verify
    assert def_means.shape == means.shape
    assert def_scales.shape == scales.shape
    
    print("✓ Full pipeline test PASSED\n")
    return True


if __name__ == "__main__":
    import sys
    
    try:
        all_passed = True
        
        all_passed &= test_hexplane_forward()
        all_passed &= test_deformation_module()
        all_passed &= test_apply_deformation()
        all_passed &= test_full_pipeline()
        
        if all_passed:
            print("=" * 60)
            print("✅ ALL TESTS PASSED")
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
