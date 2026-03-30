"""
D-NeRF Dataset Parser for Matryoshka Gaussian Splatting.

D-NeRF (Dynamic Neural Radiance Fields) uses a format similar to Blender/NeRF-Synthetic,
but includes time information for each frame to support dynamic scene modeling.

Dataset format:
    data/dnerf/{scene}/
    ├── transforms_train.json
    ├── transforms_val.json
    ├── transforms_test.json (optional)
    └── images/ (or rgba/, rgb/ depending on dataset)

Each frame in transforms JSON contains:
    - file_path: path to image file
    - transform_matrix: 4x4 camera pose
    - time: timestamp in [0, 1] (or computed from frame index)

Reference:
    - D-NeRF: https://github.com/albertpumarola/D-NeRF
    - 4DGaussians: https://github.com/hustvl/4DGaussians
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from .blender import Dataset as BlenderDataset
from .blender import Parser as BlenderParser


@dataclass
class _SplitData:
    """Container for split data with time information."""
    image_paths: List[str]
    camtoworlds: np.ndarray  # [N, 4, 4]
    times: np.ndarray  # [N,] timestamps in [0, 1]


class Parser(BlenderParser):
    """
    D-NeRF Parser extending Blender parser with time support.
    
    This parser handles dynamic datasets with temporal information.
    Time values are normalized to [0, 1] range.
    """
    
    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
    ):
        """
        Initialize D-NeRF parser.
        
        Args:
            data_dir: Path to dataset directory
            factor: Downsample factor for images
            normalize: Whether to normalize camera poses
            test_every: Interval for test frames (kept for compatibility)
        """
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        self.dataset_type = "dnerf"
        
        train_json = os.path.join(data_dir, "transforms_train.json")
        if not os.path.exists(train_json):
            raise ValueError(
                f"Expected D-NeRF dataset at {data_dir}, missing transforms_train.json"
            )
        
        # Load metadata
        train_meta = self._load_json(train_json)
        
        # Prefer explicit val split, fallback to test
        val_json = os.path.join(data_dir, "transforms_val.json")
        test_json = os.path.join(data_dir, "transforms_test.json")
        
        val_meta = self._load_json(val_json) if os.path.exists(val_json) else None
        test_meta = self._load_json(test_json) if os.path.exists(test_json) else None
        
        # Validate camera parameters
        angle_x = float(train_meta.get("camera_angle_x", 0.0))
        if angle_x <= 0.0:
            raise ValueError("D-NeRF transforms JSON missing camera_angle_x.")
        
        # Parse splits with time information
        train_split = self._parse_split(train_meta)
        if val_meta is not None:
            val_split = self._parse_split(val_meta)
        elif test_meta is not None:
            val_split = self._parse_split(test_meta)
        else:
            raise ValueError(
                "D-NeRF dataset missing transforms_val.json or transforms_test.json."
            )
        
        self.splits: Dict[str, _SplitData] = {
            "train": train_split,
            "val": val_split,
        }
        
        # Combine train/val for trajectory rendering
        self.camtoworlds = np.concatenate(
            [self.splits["train"].camtoworlds, self.splits["val"].camtoworlds],
            axis=0
        )
        
        # Combine times for all frames
        self.times = np.concatenate(
            [self.splits["train"].times, self.splits["val"].times],
            axis=0
        )
        
        # Normalize cameras if requested
        if normalize:
            from .normalize import normalize as normalize_cameras_only
            self.camtoworlds, self.transform = normalize_cameras_only(self.camtoworlds)
            
            # Re-slice back into splits
            n_train = len(self.splits["train"].camtoworlds)
            self.splits["train"].camtoworlds = self.camtoworlds[:n_train]
            self.splits["val"].camtoworlds = self.camtoworlds[n_train:]
        else:
            self.transform = np.eye(4, dtype=np.float32)
        
        # Load intrinsics from first image
        import imageio.v2 as imageio
        from PIL import Image
        
        first_image = imageio.imread(self.splits["train"].image_paths[0])[..., :3]
        first_image = self._downscale_image(first_image, factor)
        height, width = first_image.shape[:2]
        
        focal = 0.5 * float(width) / np.tan(0.5 * angle_x)
        K = np.array([
            [focal, 0.0, width * 0.5],
            [0.0, focal, height * 0.5],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # Store metadata in COLMAP-compatible format
        self.image_names = [
            os.path.relpath(p, data_dir)
            for p in self.splits["train"].image_paths
        ] + [
            os.path.relpath(p, data_dir)
            for p in self.splits["val"].image_paths
        ]
        self.image_paths = (
            self.splits["train"].image_paths + self.splits["val"].image_paths
        )
        
        self.camera_ids = [0 for _ in self.image_paths]
        self.Ks_dict = {0: K}
        self.params_dict = {0: np.empty(0, dtype=np.float32)}
        self.imsize_dict = {0: (width, height)}
        self.mask_dict = {0: None}
        
        # Compute scene scale from camera extent
        cam_locs = self.camtoworlds[:, :3, 3]
        scene_center = np.mean(cam_locs, axis=0)
        dists = np.linalg.norm(cam_locs - scene_center, axis=1)
        self.scene_scale = float(np.max(dists)) if len(dists) else 1.0
        
        # Empty point cloud (D-NeRF doesn't provide SfM points)
        self.points = np.zeros((0, 3), dtype=np.float32)
        self.points_err = np.zeros((0,), dtype=np.float32)
        self.points_rgb = np.zeros((0, 3), dtype=np.uint8)
        self.point_indices: Dict[str, np.ndarray] = {}
        
        # Bounding boxes for rendering
        self.extconf = {"spiral_radius_scale": 1.0, "no_factor_suffix": True}
        self.bounds = np.array([
            0.01,
            max(self.scene_scale * 2.0, 1.0)
        ], dtype=np.float32)
        
        print(
            f"[DNeRFParser] train={len(self.splits['train'].image_paths)} "
            f"val={len(self.splits['val'].image_paths)} "
            f"factor={factor} size=({width}x{height}) "
            f"time_range=[{self.times.min():.3f}, {self.times.max():.3f}]"
        )
    
    def _load_json(self, path: str) -> Dict[str, Any]:
        """Load JSON file."""
        with open(path, "r") as f:
            return json.load(f)
    
    def _downscale_image(self, image: np.ndarray, factor: int) -> np.ndarray:
        """Downscale image by factor."""
        if factor <= 1:
            return image
        h, w = image.shape[:2]
        new_w = int(round(w / factor))
        new_h = int(round(h / factor))
        pil = Image.fromarray(image)
        pil = pil.resize((new_w, new_h), Image.BICUBIC)
        return np.asarray(pil)
    
    def _parse_split(self, meta: Dict[str, Any]) -> _SplitData:
        """
        Parse a single split with time information.
        
        Args:
            meta: Loaded JSON metadata
            
        Returns:
            _SplitData with image paths, poses, and times
        """
        frames = meta.get("frames", [])
        if not frames:
            raise ValueError("Transforms JSON has no frames.")
        
        image_paths: List[str] = []
        c2ws: List[np.ndarray] = []
        times: List[float] = []
        
        total_frames = len(frames)
        
        for idx, fr in enumerate(frames):
            # Parse file path (same as Blender)
            file_path = fr.get("file_path", "")
            if not file_path:
                raise ValueError("Frame missing file_path in transforms JSON.")
            
            img_path = self._resolve_blender_image_path(self.data_dir, file_path)
            if not os.path.exists(img_path):
                raise FileNotFoundError(
                    f"Missing image file referenced by transforms JSON: {img_path}"
                )
            image_paths.append(img_path)
            
            # Parse transform matrix (same as Blender)
            T = np.array(fr["transform_matrix"], dtype=np.float32)
            if T.shape != (4, 4):
                raise ValueError("transform_matrix must be 4x4.")
            
            # Convert NeRF-Synthetic (OpenGL) to OpenCV convention
            T[:3, 1:3] *= -1.0
            c2ws.append(T)
            
            # Parse time information
            if "time" in fr:
                # Use provided time value
                times.append(float(fr["time"]))
            else:
                # Compute from frame index: t = idx / (N-1)
                # This ensures time in [0, 1] range
                if total_frames > 1:
                    times.append(idx / (total_frames - 1))
                else:
                    times.append(0.0)
        
        # Validate time starts at 0
        if times and abs(times[0]) > 1e-6:
            print(
                f"[WARN] D-NeRF times should start at 0, but got {times[0]}. "
                "Normalizing time range to [0, 1]."
            )
            t_min = min(times)
            t_max = max(times)
            t_range = t_max - t_min
            if t_range > 1e-6:
                times = [(t - t_min) / t_range for t in times]
            else:
                times = [0.0 for _ in times]
        
        camtoworlds = np.stack(c2ws, axis=0)
        times_array = np.array(times, dtype=np.float32)
        
        return _SplitData(
            image_paths=image_paths,
            camtoworlds=camtoworlds,
            times=times_array
        )
    
    def _resolve_blender_image_path(self, data_dir: str, file_path: str) -> str:
        """Resolve image path from file_path in transforms JSON."""
        rel = file_path.lstrip("./")
        p = Path(data_dir) / rel
        
        # Try with .png extension (D-NeRF convention)
        if p.suffix == "":
            p = p.with_suffix(".png")
        
        # Fallback: search for actual file
        if not p.exists():
            for ext in [".png", ".jpg", ".jpeg", ".exr"]:
                candidate = p.with_suffix(ext)
                if candidate.exists():
                    return str(candidate)
        
        return str(p)


class Dataset(BlenderDataset):
    """
    D-NeRF Dataset extending Blender Dataset with time support.
    
    This dataset provides time information for each frame,
    which is used by the deformation module for temporal modeling.
    """
    
    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        """
        Initialize D-NeRF dataset.
        
        Args:
            parser: DNeRFParser instance
            split: "train" or "val"
            patch_size: Random crop size (experimental)
            load_depths: Whether to load depth (not supported)
        """
        if load_depths:
            raise ValueError(
                "Depth supervision is not supported for D-NeRF datasets."
            )
        
        if not isinstance(parser, Parser):
            raise TypeError("DNeRFDataset requires a DNeRFParser instance.")
        
        super().__init__(parser, split, patch_size, load_depths)
        
        # Store time information
        self.parser = parser
        self.split_data = parser.splits[split]
        self.times = self.split_data.times
    
    def __getitem__(self, idx: int):
        """
        Get a single data sample with time information.
        
        Args:
            idx: Index of sample
            
        Returns:
            dict with keys: image, camtoworld, K, camera_id, time
        """
        # Get base data from parent class
        item = super().__getitem__(idx)
        
        # Add time information
        item["time"] = torch.tensor(self.times[idx], dtype=torch.float32)
        
        return item
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.split_data.image_paths)
