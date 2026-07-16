import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

# Add src directory to path
sys.path.insert(0, "/Users/stnava/code/syntx/src")

# Monkeypatch SwinUNETR to ignore img_size parameter
import monai.networks.nets
original_init = monai.networks.nets.SwinUNETR.__init__

def patched_init(self, *args, **kwargs):
    if 'img_size' in kwargs:
        # print(f"[Monkeypatch] Removing img_size={kwargs['img_size']} from SwinUNETR init")
        kwargs.pop('img_size')
    original_init(self, *args, **kwargs)

monai.networks.nets.SwinUNETR.__init__ = patched_init

from syntx.features import SwinUNETRExtractor

def test_instantiation():
    print("--- Test 1: Instantiation ---")
    try:
        extractor = SwinUNETRExtractor(feature_layers=[1, 2, 3, 4], weights_path="random", img_size=(96, 96, 96))
        print("Success: Extractor instantiated with monkeypatched SwinUNETR.")
        print(f"Extractor attributes: is_3d={extractor.is_3d}, in_channels={extractor.in_channels}")
        return extractor
    except Exception as e:
        print(f"Failure during instantiation: {e}")
        return None

def test_shapes_and_interpolation(extractor):
    print("\n--- Test 2: Shapes and Interpolation ---")
    if extractor is None:
        print("Skipping Test 2 as extractor is None.")
        return

    # Case 1: Native size 96x96x96
    print("\nCase 1: Input size matching img_size (96, 96, 96)")
    x_native = torch.randn(1, 1, 96, 96, 96)
    with torch.no_grad():
        feats_native = extractor.extract(x_native)
    
    expected_native_shapes = [
        (1, 96, 24, 24, 24),   # layer 1
        (1, 192, 12, 12, 12),  # layer 2
        (1, 384, 6, 6, 6),     # layer 3
        (1, 768, 3, 3, 3)      # layer 4
    ]
    
    for idx, feat in enumerate(feats_native):
        layer = idx + 1
        print(f"Layer {layer} output shape: {feat.shape} (Expected: {expected_native_shapes[idx]})")
        assert feat.shape == expected_native_shapes[idx], f"Shape mismatch at layer {layer}!"

    # Case 2: Input size 64x64x64
    print("\nCase 2: Input size differing from img_size (64, 64, 64)")
    x_64 = torch.randn(1, 1, 64, 64, 64)
    with torch.no_grad():
        feats_64 = extractor.extract(x_64)
    
    # Let's inspect the calculated expected_shape in the code:
    # expected_shape = [max(1, s // (2 ** layer)) for s in spatial_shape]
    # For s = 64:
    # layer 1: 64 // 2 = 32
    # layer 2: 64 // 4 = 16
    # layer 3: 64 // 8 = 8
    # layer 4: 64 // 16 = 4
    calculated_expected_shapes = [
        (1, 96, 32, 32, 32),
        (1, 192, 16, 16, 16),
        (1, 384, 8, 8, 8),
        (1, 768, 4, 4, 4)
    ]
    
    # Correct physical downsampling shapes (since layer 1 is /4, layer 2 is /8, layer 3 is /16, layer 4 is /32):
    # layer 1: 64 // 4 = 16
    # layer 2: 64 // 8 = 8
    # layer 3: 64 // 16 = 4
    # layer 4: 64 // 32 = 2
    correct_physical_shapes = [
        (1, 96, 16, 16, 16),
        (1, 192, 8, 8, 8),
        (1, 384, 4, 4, 4),
        (1, 768, 2, 2, 2)
    ]

    for idx, feat in enumerate(feats_64):
        layer = idx + 1
        print(f"Layer {layer} output shape: {feat.shape}")
        print(f"  -> Calculated by code: {calculated_expected_shapes[idx]}")
        print(f"  -> Correct physical:   {correct_physical_shapes[idx]}")
        
        # Check if code's output matches the calculated but incorrect formula:
        if feat.shape == calculated_expected_shapes[idx]:
            print(f"  [BUG CONFIRMED] Code outputs shape {feat.shape} due to incorrect downsampling factor formula (2**layer).")
        else:
            print(f"  Unexpected shape: {feat.shape}")

def test_performance(extractor):
    print("\n--- Test 3: Performance ---")
    if extractor is None:
        print("Skipping Test 3 as extractor is None.")
        return

    # Measure forward pass times
    x_native = torch.randn(1, 1, 96, 96, 96)
    x_64 = torch.randn(1, 1, 64, 64, 64)
    
    # Warmup
    with torch.no_grad():
        _ = extractor.extract(x_native)
        _ = extractor.extract(x_64)
        
    t0 = time.time()
    for _ in range(5):
        with torch.no_grad():
            _ = extractor.extract(x_native)
    t1 = time.time()
    native_time = (t1 - t0) / 5.0
    
    t0 = time.time()
    for _ in range(5):
        with torch.no_grad():
            _ = extractor.extract(x_64)
    t1 = time.time()
    interpolated_time = (t1 - t0) / 5.0
    
    print(f"Mean inference time (native 96^3): {native_time:.4f} seconds")
    print(f"Mean inference time (interpolated 64^3): {interpolated_time:.4f} seconds")

def test_offline_behavior():
    print("\n--- Test 4: Offline Behavior & Download Failure Handling ---")
    
    # Path that doesn't exist
    fake_path = "/tmp/non_existent_cache_dir/model_swinvit.pt"
    
    # Mock urllib retrieve to raise an exception (simulating offline mode)
    import urllib.request
    original_urlretrieve = urllib.request.urlretrieve
    
    def mock_urlretrieve(url, filename, reporthook=None, data=None):
        raise urllib.error.URLError("Connection timed out (Simulated Offline Mode)")
        
    urllib.request.urlretrieve = mock_urlretrieve
    
    # Try initializing SwinUNETRExtractor with fake_path (not "random")
    # This should trigger download, fail, warn, and initialize with random weights.
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            extractor = SwinUNETRExtractor(feature_layers=[4], weights_path=fake_path, img_size=(96, 96, 96))
            print("Successfully handled download failure!")
            print(f"Number of warnings caught: {len(w)}")
            for warning in w:
                print(f"Warning message: {warning.message}")
            print("Model weights initialized to random since download failed.")
    except Exception as e:
        print(f"Failure in handling offline mode/download failure: {e}")
    finally:
        urllib.request.urlretrieve = original_urlretrieve

if __name__ == "__main__":
    extractor = test_instantiation()
    test_shapes_and_interpolation(extractor)
    test_performance(extractor)
    test_offline_behavior()
