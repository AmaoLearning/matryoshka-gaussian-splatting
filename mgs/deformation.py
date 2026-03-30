"""
HexPlane deformation field for dynamic scene modeling.

This module implements the 4D Gaussian deformation field from "4D Gaussian Splatting for 
Real-Time Dynamic Scene Rendering" (CVPR 2024). The deformation field uses 6 planes 
(XY, XZ, YZ, XT, YT, ZT) to encode spatiotemporal features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Tuple, Literal


class HexPlaneField(nn.Module):
    """
    HexPlane feature grid for 4D coordinate encoding.
    
    Projects 4D points (x, y, z, t) onto 6 2D planes and aggregates features.
    """
    
    def __init__(
        self,
        resolution: List[int] = [64, 64, 64, 150],  # [res_x, res_y, res_z, res_t]
        feature_dim: int = 16,
        multires: List[int] = [1, 2],
        device: str = "cuda",
    ):
        super().__init__()
        self.resolution = resolution
        self.feature_dim = feature_dim
        self.multires = multires
        self.device = device
        
        # 6 planes: XY, XZ, YZ, XT, YT, ZT
        self.plane_dims = ["XY", "XZ", "YZ", "XT", "YT", "ZT"]
        
        # Multi-resolution feature grids
        self.feature_grids = nn.ModuleList()
        for level in multires:
            level_res = [r // level for r in resolution]
            # 6 planes, each with feature_dim channels
            planes = nn.ParameterList([
                nn.Parameter(torch.zeros(1, feature_dim, level_res[i], level_res[j]))
                for i, j in [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
            ])
            self.feature_grids.append(planes)
        
        # Initialize grids
        self._init_grids()
    
    def _init_grids(self):
        """Initialize feature grids with small random values."""
        for grids in self.feature_grids:
            for plane in grids:
                nn.init.xavier_uniform_(plane.data)
    
    def forward(self, coords: Tensor) -> Tensor:
        """
        Query features at 4D coordinates.
        
        Args:
            coords: 4D coordinates (B, N, 4) where 4 = (x, y, z, t), normalized to [-1, 1]
        
        Returns:
            features: (B, N, feature_dim * len(multires))
        """
        B, N, _ = coords.shape
        all_features = []
        
        for level_idx, grids in enumerate(self.feature_grids):
            level_features = []
            
            for plane_idx, (plane, dim_str) in enumerate(zip(grids, self.plane_dims)):
                # Get plane dimensions
                if dim_str == "XY":
                    grid_coords = coords[..., [0, 1]]  # x, y
                elif dim_str == "XZ":
                    grid_coords = coords[..., [0, 2]]  # x, z
                elif dim_str == "YZ":
                    grid_coords = coords[..., [1, 2]]  # y, z
                elif dim_str == "XT":
                    grid_coords = coords[..., [0, 3]]  # x, t
                elif dim_str == "YT":
                    grid_coords = coords[..., [1, 3]]  # y, t
                elif dim_str == "ZT":
                    grid_coords = coords[..., [2, 3]]  # z, t
                else:
                    raise ValueError(f"Unknown plane dimension: {dim_str}")
                
                # Reshape for grid_sample: (B, 1, N, 2)
                grid = grid_coords.unsqueeze(1).unsqueeze(1)
                
                # Sample from plane: (B, feature_dim, 1, N)
                sampled = F.grid_sample(
                    plane.expand(B, -1, -1, -1),
                    grid,
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=True
                )
                
                # Reshape to (B, N, feature_dim)
                sampled = sampled.squeeze(-2).transpose(1, 2)
                level_features.append(sampled)
            
            # Aggregate features from 6 planes using concatenation
            # Concatenate all 6 plane features: 6 * feature_dim
            concat_features = torch.cat(level_features, dim=-1)  # (B, N, 6 * feature_dim)
            all_features.append(concat_features)
        
        # Concatenate multi-resolution features
        features = torch.cat(all_features, dim=-1)  # (B, N, 6 * feature_dim * len(multires))
        return features


class DeformationModule(nn.Module):
    """
    Deformation MLP that decodes HexPlane features into Gaussian parameter offsets.
    
    Predicts offsets for: position (dx), scale (ds), rotation (dr).
    Optionally predicts opacity (do) and SH coefficients (dsh) offsets.
    """
    
    def __init__(
        self,
        feature_dim: int = 48,  # hexplane feature_dim * len(multires) = 16 * 3 = 48
        hidden_dim: int = 256,
        num_layers: int = 2,
        predict_opacity: bool = False,
        predict_sh: bool = False,
        sh_degree: int = 3,
        device: str = "cuda",
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.predict_opacity = predict_opacity
        self.predict_sh = predict_sh
        self.sh_degree = sh_degree
        self.device = device
        
        # Position offset (3D)
        self.pos_mlp = self._build_mlp(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=3
        )
        
        # Scale offset (3D) - in log space
        self.scale_mlp = self._build_mlp(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=3
        )
        
        # Rotation offset (6D representation)
        self.rot_mlp = self._build_mlp(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=6
        )
        
        # Optional: opacity offset (1D)
        if predict_opacity:
            self.opa_mlp = self._build_mlp(
                input_dim=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=1
            )
        
        # Optional: SH offset (K*3 where K=(sh_degree+1)^2)
        if predict_sh:
            num_sh = (sh_degree + 1) ** 2
            self.sh_mlp = self._build_mlp(
                input_dim=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=num_sh * 3
            )
    
    def _build_mlp(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int
    ) -> nn.Sequential:
        """Build MLP with ReLU activation."""
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ])
        layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)
    
    def forward(
        self,
        features: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Decode features into deformation offsets.
        
        Args:
            features: (B, N, feature_dim) from HexPlaneField
        
        Returns:
            dx: position offset (..., 3)
            ds: scale offset (..., 3)
            dr: rotation offset in 6D (..., 6)
            do: opacity offset (..., 1) or None
            dsh: SH offset (..., K*3) or None
        """
        dx = self.pos_mlp(features)
        ds = self.scale_mlp(features)
        dr = self.rot_mlp(features)
        
        do = self.opa_mlp(features) if self.predict_opacity else None
        dsh = self.sh_mlp(features) if self.predict_sh else None
        
        return dx, ds, dr, do, dsh


def apply_deformation(
    means: Tensor,
    scales: Tensor,
    quats: Tensor,
    opacities: Optional[Tensor],
    sh_coeffs: Optional[Tensor],
    dx: Tensor,
    ds: Tensor,
    dr: Tensor,
    do: Optional[Tensor] = None,
    dsh: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    """
    Apply deformation offsets to Gaussian parameters.
    
    Args:
        means: (N, 3) positions
        scales: (N, 3) scale factors (in log space)
        quats: (N, 4) quaternions (w, x, y, z)
        opacities: (N, 1) opacities
        sh_coeffs: (N, K, 3) SH coefficients
        dx: (N, 3) position offset
        ds: (N, 3) scale offset
        dr: (N, 6) rotation offset (6D representation)
        do: (N, 1) opacity offset or None
        dsh: (N, K*3) SH offset or None
    
    Returns:
        deformed_means, deformed_scales, deformed_quats, deformed_opacities, deformed_sh_coeffs
    """
    # Position: simple addition
    deformed_means = means + dx
    
    # Scale: addition in log space
    deformed_scales = scales + ds
    
    # Rotation: convert 6D to quaternion, then multiply
    dr_quat = rotation_6d_to_quaternion(dr)
    deformed_quats = quaternion_multiply(quats, dr_quat)
    
    # Opacity: optional addition
    deformed_opacities = opacities + do if do is not None else opacities
    
    # SH coefficients: optional addition
    deformed_sh = sh_coeffs + dsh.view(-1, (dsh.shape[-1] // 3), 3) if dsh is not None else sh_coeffs
    
    return deformed_means, deformed_scales, deformed_quats, deformed_opacities, deformed_sh


def rotation_6d_to_quaternion(rot_6d: Tensor) -> Tensor:
    """
    Convert 6D rotation representation to unit quaternion.
    
    Args:
        rot_6d: (..., 6) 6D rotation representation
    
    Returns:
        quaternion: (..., 4) unit quaternion (w, x, y, z)
    """
    # 6D representation: [a1, a2] where a1, a2 are orthogonal 3D vectors
    # Convert to rotation matrix first
    a1 = rot_6d[..., 0:3]
    a2 = rot_6d[..., 3:6]
    
    # Gram-Schmidt orthogonalization
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    
    # Rotation matrix: [b1, b2, b3]
    rot_mat = torch.stack([b1, b2, b3], dim=-1)  # (..., 3, 3)
    
    # Convert rotation matrix to quaternion
    return matrix_to_quaternion(rot_mat)


def matrix_to_quaternion(R: Tensor) -> Tensor:
    """
    Convert rotation matrix to quaternion.
    
    Args:
        R: (..., 3, 3) rotation matrix
    
    Returns:
        q: (..., 4) quaternion (w, x, y, z)
    """
    # Standard conversion formula
    batch_dims = R.shape[:-2]
    R = R.view(-1, 3, 3)
    
    # Compute trace
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    
    # Initialize quaternion
    q = torch.zeros(R.shape[0], 4, device=R.device, dtype=R.dtype)
    
    # Case 1: trace > 0
    mask = tr > 0
    S = torch.sqrt(tr[mask] + 1.0) * 2
    q[mask, 0] = 0.25 * S
    q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / S
    q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / S
    q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / S
    
    # Case 2: R[0,0] is largest diagonal
    mask = (~mask) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    S = torch.sqrt(1.0 + R[mask, 0, 0] - R[mask, 1, 1] - R[mask, 2, 2]) * 2
    q[mask, 0] = (R[mask, 2, 1] - R[mask, 1, 2]) / S
    q[mask, 1] = 0.25 * S
    q[mask, 2] = (R[mask, 0, 1] + R[mask, 1, 0]) / S
    q[mask, 3] = (R[mask, 0, 2] + R[mask, 2, 0]) / S
    
    # Case 3: R[1,1] is largest diagonal
    mask = (~mask) & (R[:, 1, 1] > R[:, 2, 2])
    S = torch.sqrt(1.0 + R[mask, 1, 1] - R[mask, 0, 0] - R[mask, 2, 2]) * 2
    q[mask, 0] = (R[mask, 0, 2] - R[mask, 2, 0]) / S
    q[mask, 1] = (R[mask, 0, 1] + R[mask, 1, 0]) / S
    q[mask, 2] = 0.25 * S
    q[mask, 3] = (R[mask, 1, 2] + R[mask, 2, 1]) / S
    
    # Case 4: R[2,2] is largest diagonal
    mask = ~mask
    S = torch.sqrt(1.0 + R[mask, 2, 2] - R[mask, 0, 0] - R[mask, 1, 1]) * 2
    q[mask, 0] = (R[mask, 1, 0] - R[mask, 0, 1]) / S
    q[mask, 1] = (R[mask, 0, 2] + R[mask, 2, 0]) / S
    q[mask, 2] = (R[mask, 1, 2] + R[mask, 2, 1]) / S
    q[mask, 3] = 0.25 * S
    
    # Normalize to ensure unit quaternion
    q = F.normalize(q, dim=-1)
    
    return q.view(*batch_dims, 4)


def quaternion_multiply(q1: Tensor, q2: Tensor) -> Tensor:
    """
    Multiply two quaternions.
    
    Args:
        q1: (..., 4) quaternion (w, x, y, z)
        q2: (..., 4) quaternion (w, x, y, z)
    
    Returns:
        q: (..., 4) quaternion (w, x, y, z)
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([w, x, y, z], dim=-1)
