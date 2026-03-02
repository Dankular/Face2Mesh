"""
Test script to verify dependencies and setup
"""

import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def test_imports():
    """Test all required imports"""
    print("=" * 60)
    print("Testing Dependencies Installation")
    print("=" * 60)
    
    results = {}
    
    # Test PyTorch
    try:
        import torch
        print(f"\n[OK] PyTorch: {torch.__version__}")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - CUDA version: {torch.version.cuda}")
            print(f"  - GPU: {torch.cuda.get_device_name(0)}")
        results['torch'] = True
    except Exception as e:
        print(f"\n[FAIL] PyTorch: {e}")
        results['torch'] = False
    
    # Test Pillow
    try:
        from PIL import Image
        print(f"\n[OK] Pillow (PIL)")
        results['pillow'] = True
    except Exception as e:
        print(f"\n[FAIL] Pillow: {e}")
        results['pillow'] = False
    
    # Test NumPy
    try:
        import numpy as np
        print(f"\n[OK] NumPy: {np.__version__}")
        results['numpy'] = True
    except Exception as e:
        print(f"\n[FAIL] NumPy: {e}")
        results['numpy'] = False
    
    # Test Diffusers
    try:
        import diffusers
        print(f"\n[OK] Diffusers: {diffusers.__version__}")
        results['diffusers'] = True
    except Exception as e:
        print(f"\n[FAIL] Diffusers: {e}")
        results['diffusers'] = False
    
    # Test Transformers
    try:
        import transformers
        print(f"\n[OK] Transformers: {transformers.__version__}")
        results['transformers'] = True
    except Exception as e:
        print(f"\n[FAIL] Transformers: {e}")
        results['transformers'] = False
    
    # Test Accelerate
    try:
        import accelerate
        print(f"\n[OK] Accelerate: {accelerate.__version__}")
        results['accelerate'] = True
    except Exception as e:
        print(f"\n[FAIL] Accelerate: {e}")
        results['accelerate'] = False
    
    # Test Trimesh
    try:
        import trimesh
        print(f"\n[OK] Trimesh: {trimesh.__version__}")
        results['trimesh'] = True
    except Exception as e:
        print(f"\n[FAIL] Trimesh: {e}")
        results['trimesh'] = False
    
    # Test Plyfile
    try:
        import plyfile
        print(f"\n[OK] Plyfile")
        results['plyfile'] = True
    except Exception as e:
        print(f"\n[FAIL] Plyfile: {e}")
        results['plyfile'] = False
    
    # Test Open3D (optional)
    try:
        import open3d
        print(f"\n[OK] Open3D: {open3d.__version__} (optional)")
        results['open3d'] = True
    except Exception as e:
        print(f"\n[SKIP] Open3D: Not installed (optional)")
        results['open3d'] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    total = len([k for k in results.keys() if k != 'open3d'])
    passed = len([k for k, v in results.items() if v and k != 'open3d'])
    
    print(f"\nCore Dependencies: {passed}/{total} installed")
    
    if passed == total:
        print("\n[SUCCESS] All core dependencies installed successfully!")
        return True
    else:
        print("\n[WARNING] Some dependencies missing. Please install them.")
        return False

def test_image():
    """Test loading the face image"""
    print("\n" + "=" * 60)
    print("Testing Test Image")
    print("=" * 60)
    
    try:
        from PIL import Image
        
        image_path = "test_face.jpg"
        
        if not os.path.exists(image_path):
            print(f"\n[FAIL] Image not found: {image_path}")
            return False
        
        img = Image.open(image_path)
        print(f"\n[OK] Image loaded successfully")
        print(f"  - Path: {image_path}")
        print(f"  - Size: {img.size}")
        print(f"  - Mode: {img.mode}")
        print(f"  - Format: {img.format}")
        
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Failed to load image: {e}")
        return False

def test_pipeline_init():
    """Test initializing the pipeline"""
    print("\n" + "=" * 60)
    print("Testing Pipeline Initialization")
    print("=" * 60)
    
    try:
        # Try importing our pipeline
        import torch
        from pathlib import Path
        
        # Check if __init__.py exists
        init_path = Path("__init__.py")
        if not init_path.exists():
            print(f"\n[WARNING] Pipeline not found at {init_path}")
            return False
        
        print(f"\n[OK] Pipeline file found")
        print(f"  - Path: {init_path}")
        print(f"  - Size: {init_path.stat().st_size} bytes")
        
        # Don't actually import yet (Qwen models are large)
        print(f"\n[INFO] Skipping model loading (requires large download)")
        print(f"  - Qwen-Image-Edit-2511 model is several GB")
        print(f"  - Run actual pipeline test when ready")
        
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Pipeline initialization failed: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Face to 3D - Setup Test")
    print("=" * 60 + "\n")
    
    # Run tests
    deps_ok = test_imports()
    image_ok = test_image()
    pipeline_ok = test_pipeline_init()
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    if deps_ok and image_ok and pipeline_ok:
        print("\n[SUCCESS] All tests passed!")
        print("\nNext steps:")
        print("1. The first run will download Qwen models (~5-10 GB)")
        print("2. Run: python __init__.py --input test_face.jpg --output ./output --views 8")
        print("3. Or use the Python API in your own script")
        sys.exit(0)
    else:
        print("\n[WARNING] Some tests failed. Please review above.")
        sys.exit(1)
