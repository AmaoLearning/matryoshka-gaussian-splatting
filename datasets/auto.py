from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple


def _is_dnerf_dataset(data_dir: str) -> bool:
    """
    Check if directory contains a D-NeRF dataset.
    
    D-NeRF datasets have transforms_train.json with time information in frames.
    
    Args:
        data_dir: Path to dataset directory
        
    Returns:
        True if D-NeRF dataset is detected
    """
    p = Path(data_dir)
    train_json = p / "transforms_train.json"
    
    if not train_json.exists():
        return False
    
    # Check if frames contain time field
    try:
        with open(train_json, "r") as f:
            meta = json.load(f)
            frames = meta.get("frames", [])
            if frames and "time" in frames[0]:
                return True
    except (json.JSONDecodeError, KeyError, IndexError):
        pass
    
    return False


def _is_blender_dataset(data_dir: str) -> bool:
    p = Path(data_dir)
    return (p / "transforms_train.json").exists()


def build_parser_and_datasets(
    *,
    data_dir: str,
    factor: int,
    normalize: bool,
    test_every: int,
    patch_size: Optional[int],
    load_depths: bool,
    skip_t3: bool = False,
) -> Tuple[object, object, object]:
    """
    Construct (parser, trainset, valset) for:
    - D-NeRF datasets: expects transforms_{train,val,test}.json with time field
    - COLMAP datasets: expects {images/, sparse[/0]/}
    - Blender/NeRF-Synthetic datasets: expects transforms_{train,val,test}.json
    """
    # Check for D-NeRF dataset first (has time information)
    if _is_dnerf_dataset(data_dir):
        from .dnerf import Dataset, Parser  # local import to keep deps optional
        
        parser = Parser(
            data_dir=data_dir,
            factor=factor,
            normalize=normalize,
            test_every=test_every,
        )
        setattr(parser, "dataset_type", "dnerf")
        trainset = Dataset(
            parser,
            split="train",
            patch_size=patch_size,
            load_depths=load_depths,
        )
        valset = Dataset(parser, split="val")
        return parser, trainset, valset
    
    # Check for Blender dataset (no time information)
    if _is_blender_dataset(data_dir):
        from .blender import Dataset, Parser  # local import to keep deps optional

        parser = Parser(
            data_dir=data_dir,
            factor=factor,
            normalize=normalize,
            test_every=test_every,
        )
        setattr(parser, "dataset_type", "blender")
        trainset = Dataset(
            parser,
            split="train",
            patch_size=patch_size,
            load_depths=load_depths,
        )
        valset = Dataset(parser, split="val")
        return parser, trainset, valset

    # Default to COLMAP
    from .colmap import Dataset, Parser

    parser = Parser(
        data_dir=data_dir,
        factor=factor,
        normalize=normalize,
        test_every=test_every,
        skip_t3=skip_t3,
    )
    setattr(parser, "dataset_type", "colmap")
    trainset = Dataset(
        parser,
        split="train",
        patch_size=patch_size,
        load_depths=load_depths,
    )
    valset = Dataset(parser, split="val")
    return parser, trainset, valset

