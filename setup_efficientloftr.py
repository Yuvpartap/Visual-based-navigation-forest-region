"""
Setup and verification script for EfficientLoFTR implementation.

This script:
1. Checks dependencies
2. Verifies GPU availability
3. Tests model loading
4. Validates installation
5. Provides recommendations

Usage:
    python setup_efficientloftr.py
"""

import sys
import subprocess
import importlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def print_header(text):
    """Print formatted header."""
    logger.info("\n" + "=" * 70)
    logger.info(text)
    logger.info("=" * 70)


def print_section(text):
    """Print formatted section."""
    logger.info(f"\n{text}")
    logger.info("-" * 70)


def check_python_version():
    """Check Python version."""
    print_section("Checking Python Version")
    
    version = sys.version_info
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("❌ Python 3.8+ required")
        logger.info("Please upgrade Python")
        return False
    
    logger.info("✅ Python version OK")
    return True


def check_package(package_name, import_name=None, min_version=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        
        if min_version and version != 'unknown':
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                logger.warning(f"⚠️  {package_name} {version} (need {min_version}+)")
                return False
        
        logger.info(f"✅ {package_name} {version}")
        return True
    except ImportError:
        logger.error(f"❌ {package_name} not installed")
        return False


def check_dependencies():
    """Check all required dependencies."""
    print_section("Checking Dependencies")
    
    packages = [
        ('numpy', 'numpy', '1.24.0'),
        ('opencv-python', 'cv2', '4.7.0'),
        ('torch', 'torch', '2.0.0'),
        ('torchvision', 'torchvision', '0.15.0'),
        ('transformers', 'transformers', '4.30.0'),
        ('accelerate', 'accelerate', '0.20.0'),
        ('kornia', 'kornia', '0.7.0'),
    ]
    
    all_ok = True
    for package_name, import_name, min_version in packages:
        if not check_package(package_name, import_name, min_version):
            all_ok = False
    
    return all_ok


def check_gpu():
    """Check GPU availability."""
    print_section("Checking GPU")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"   CUDA version: {torch.version.cuda}")
            
            # Get GPU memory
            props = torch.cuda.get_device_properties(0)
            total_memory = props.total_memory / 1e9
            logger.info(f"   GPU memory: {total_memory:.1f} GB")
            
            # Check memory
            if total_memory < 4:
                logger.warning("⚠️  GPU has less than 4GB memory")
                logger.info("   Consider using smaller grid_size or resize_max")
            
            return True
        else:
            logger.warning("⚠️  CUDA not available - will use CPU")
            logger.info("   For better performance, install CUDA-enabled PyTorch:")
            logger.info("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            return False
    except ImportError:
        logger.error("❌ PyTorch not installed")
        return False


def test_model_loading():
    """Test loading EfficientLoFTR model."""
    print_section("Testing Model Loading")
    
    try:
        import torch
        from transformers import AutoModel
        
        logger.info("Attempting to load EfficientLoFTR model...")
        logger.info("(First time will download ~200MB)")
        
        model = AutoModel.from_pretrained(
            "zju-community/efficientloftr",
            trust_remote_code=True
        )
        
        logger.info("✅ EfficientLoFTR model loaded successfully")
        
        # Check model size
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"   Model parameters: {param_count / 1e6:.1f}M")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.info("   Check internet connection")
        logger.info("   Model will be downloaded on first use")
        return False


def test_matcher_creation():
    """Test creating matcher instance."""
    print_section("Testing Matcher Creation")
    
    try:
        from src.efficient_loftr_matcher import create_efficient_loftr_matcher
        
        logger.info("Creating EfficientLoFTR matcher...")
        
        matcher = create_efficient_loftr_matcher(
            use_gpu=True,
            resize_max=840,
            use_mixed_precision=True
        )
        
        logger.info("✅ Matcher created successfully")
        logger.info(f"   Device: {matcher.device}")
        logger.info(f"   Mixed precision: {matcher.use_mixed_precision}")
        logger.info(f"   Resize max: {matcher.resize_max}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create matcher: {e}")
        return False


def test_basic_matching():
    """Test basic matching functionality."""
    print_section("Testing Basic Matching")
    
    try:
        import numpy as np
        import cv2
        from src.efficient_loftr_matcher import create_efficient_loftr_matcher
        
        logger.info("Creating test images...")
        
        # Create simple test images
        img1 = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        img2 = img1.copy()  # Same image for guaranteed matches
        
        logger.info("Creating matcher...")
        matcher = create_efficient_loftr_matcher(
            use_gpu=True,
            resize_max=640,
            use_mixed_precision=False  # Disable for test
        )
        
        logger.info("Running matching...")
        src_pts, dst_pts, confidence, num_matches = matcher.match_images(
            img1, img2, min_matches=10
        )
        
        logger.info(f"✅ Matching successful")
        logger.info(f"   Matches found: {num_matches}")
        
        if num_matches > 0:
            logger.info(f"   Confidence range: {confidence.min():.3f} - {confidence.max():.3f}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Matching test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def provide_recommendations(results):
    """Provide recommendations based on test results."""
    print_section("Recommendations")
    
    if all(results.values()):
        logger.info("✅ All checks passed! You're ready to use EfficientLoFTR")
        logger.info("\nNext steps:")
        logger.info("1. Configure paths in main_efficient_loftr.py")
        logger.info("2. Run: python main_efficient_loftr.py")
        logger.info("3. Check results in results_efficient_loftr/")
    else:
        logger.info("⚠️  Some checks failed. Please address the issues above.")
        
        if not results['dependencies']:
            logger.info("\nInstall missing dependencies:")
            logger.info("  pip install -r requirements.txt")
        
        if not results['gpu']:
            logger.info("\nFor GPU support:")
            logger.info("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        
        if not results['model_loading']:
            logger.info("\nModel loading failed:")
            logger.info("  - Check internet connection")
            logger.info("  - Model will download on first use (~200MB)")
            logger.info("  - Try: huggingface-cli download zju-community/efficientloftr")


def print_system_info():
    """Print system information."""
    print_section("System Information")
    
    import platform
    
    logger.info(f"OS: {platform.system()} {platform.release()}")
    logger.info(f"Architecture: {platform.machine()}")
    logger.info(f"Processor: {platform.processor()}")
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"RAM: {memory.total / 1e9:.1f} GB")
    except ImportError:
        logger.info("RAM: (install psutil to see)")


def main():
    """Main setup and verification function."""
    print_header("EfficientLoFTR Setup and Verification")
    
    # Print system info
    print_system_info()
    
    # Run checks
    results = {
        'python': check_python_version(),
        'dependencies': check_dependencies(),
        'gpu': check_gpu(),
        'model_loading': test_model_loading(),
        'matcher_creation': test_matcher_creation(),
        'basic_matching': test_basic_matching(),
    }
    
    # Summary
    print_header("Summary")
    
    passed = sum(results.values())
    total = len(results)
    
    logger.info(f"\nTests passed: {passed}/{total}")
    
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        logger.info(f"{status} {test_name.replace('_', ' ').title()}")
    
    # Recommendations
    provide_recommendations(results)
    
    print_header("Setup Complete")
    
    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
