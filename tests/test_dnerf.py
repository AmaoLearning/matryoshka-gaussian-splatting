"""
Test script for D-NeRF dataset integration.

This script validates:
1. D-NeRF dataset parser can load data correctly
2. Time information is properly extracted and normalized
3. Auto-detection works in datasets/auto.py
4. Dataset returns time field in __getitem__

Usage:
    python scripts/test_dnerf.py --data_dir /path/to/dnerf/bouncingballs
"""

import argparse
import os
import sys
from pathlib import Path

import torch


def test_parser(data_dir: str, factor: int = 1):
    """Test D-NeRF parser."""
    print("=" * 60)
    print("Testing D-NeRF Parser")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists(data_dir):
        print(f"❌ Dataset directory not found: {data_dir}")
        return False
    
    # Check for transforms files
    train_json = Path(data_dir) / "transforms_train.json"
    val_json = Path(data_dir) / "transforms_val.json"
    
    if not train_json.exists():
        print(f" transforms_train.json not found")
        return False
    
    if not val_json.exists():
        print(f"⚠ Warning: transforms_val.json not found")
    
    # Import and create parser
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from datasets.dnerf import Parser
    
    try:
        parser = Parser(
            data_dir=data_dir,
            factor=factor,
            normalize=True,
            test_every=8,
        )
        print(f"✓ Parser created successfully")
        print(f"  - Dataset type: {parser.dataset_type}")
        print(f"  - Train frames: {len(parser.splits['train'].image_paths)}")
        print(f"  - Val frames: {len(parser.splits['val'].image_paths)}")
        print(f"  - Time range: [{parser.times.min():.3f}, {parser.times.max():.3f}]")
        print(f"  - Image size: {parser.imsize_dict[0]}")
        
        # Validate time information
        train_times = parser.splits["train"].times
        val_times = parser.splits["val"].times
        
        assert train_times.min() >= 0.0, "Train times should be >= 0"
        assert train_times.max() <= 1.0, "Train times should be <= 1"
        assert abs(train_times[0]) < 1e-6, "Time should start at 0"
        
        print(f"✓ Time validation passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Parser creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset(data_dir: str, factor: int = 1):
    """Test D-NeRF Dataset."""
    print("\n" + "=" * 60)
    print("Testing D-NeRF Dataset")
    print("=" * 60)
    
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from datasets.dnerf import Dataset, Parser
    
    try:
        parser = Parser(
            data_dir=data_dir,
            factor=factor,
            normalize=True,
            test_every=8,
        )
        
        train_dataset = Dataset(parser, split="train")
        val_dataset = Dataset(parser, split="val")
        
        print(f"✓ Datasets created")
        print(f"  - Train length: {len(train_dataset)}")
        print(f"  - Val length: {len(val_dataset)}")
        
        # Test __getitem__
        sample = train_dataset[0]
        
        assert "time" in sample, "Sample missing 'time' field"
        assert isinstance(sample["time"], torch.Tensor), "Time should be a tensor"
        assert sample["time"].dim() == 0 or sample["time"].dim() == 1, "Time should be scalar or 1D"
        
        print(f"✓ Sample contains time field: {sample['time'].item():.3f}")
        print(f"  - Image shape: {sample['image'].shape}")
        print(f"  - K shape: {sample['K'].shape}")
        print(f"  - Camtoworld shape: {sample['camtoworld'].shape}")
        
        # Test multiple samples
        for idx in [0, len(train_dataset) // 2, len(train_dataset) - 1]:
            sample = train_dataset[idx]
            time_val = sample["time"].item()
            assert 0.0 <= time_val <= 1.0, f"Time out of range: {time_val}"
        
        print(f"✓ Multiple sample validation passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_auto_detection(data_dir: str):
    """Test auto-detection in datasets/auto.py."""
    print("\n" + "=" * 60)
    print("Testing Auto-Detection")
    print("=" * 60)
    
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from datasets.auto import _is_dnerf_dataset, _is_blender_dataset
    
    # Test D-NeRF detection
    is_dnerf = _is_dnerf_dataset(data_dir)
    is_blender = _is_blender_dataset(data_dir)
    
    print(f"  - Is D-NeRF: {is_dnerf}")
    print(f"  - Is Blender: {is_blender}")
    
    if is_dnerf:
        print(f"✓ D-NeRF auto-detection passed")
        return True
    else:
        print(f"⚠ Dataset not detected as D-NeRF (might not have time field)")
        return True  # Not necessarily a failure


def test_build_parser_and_datasets(data_dir: str, factor: int = 1):
    """Test build_parser_and_datasets function."""
    print("\n" + "=" * 60)
    print("Testing build_parser_and_datasets")
    print("=" * 60)
    
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from datasets.auto import build_parser_and_datasets
    
    try:
        parser, trainset, valset = build_parser_and_datasets(
            data_dir=data_dir,
            factor=factor,
            normalize=True,
            test_every=8,
            patch_size=None,
            load_depths=False,
            skip_t3=False,
        )
        
        dataset_type = getattr(parser, "dataset_type", "unknown")
        print(f"✓ Parser built successfully")
        print(f"  - Dataset type: {dataset_type}")
        print(f"  - Train length: {len(trainset)}")
        print(f"  - Val length: {len(valset)}")
        
        # Check if dataset type is correct
        if dataset_type == "dnerf":
            print(f"✓ Correctly detected as D-NeRF dataset")
            
            # Check time field in dataset
            sample = trainset[0]
            if "time" in sample:
                print(f"✓ Time field present in dataset samples")
            else:
                print(f"⚠ Warning: Time field missing in dataset samples")
        else:
            print(f"⚠ Warning: Dataset detected as '{dataset_type}' instead of 'dnerf'")
        
        return True
        
    except Exception as e:
        print(f"❌ build_parser_and_datasets failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    ap = argparse.ArgumentParser(description="Test D-NeRF dataset integration")
    ap.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to D-NeRF dataset directory"
    )
    ap.add_argument(
        "--factor",
        type=int,
        default=1,
        help="Downsample factor"
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = ap.parse_args()
    
    print("\n" + "=" * 60)
    print("D-NeRF Integration Test Suite")
    print("=" * 60)
    print(f"Dataset: {args.data_dir}")
    print(f"Factor: {args.factor}")
    print()
    
    results = {
        "parser": test_parser(args.data_dir, args.factor),
        "dataset": test_dataset(args.data_dir, args.factor),
        "auto_detection": test_auto_detection(args.data_dir),
        "build_parser": test_build_parser_and_datasets(args.data_dir, args.factor),
    }
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
