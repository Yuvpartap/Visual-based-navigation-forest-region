"""
CUDA management utilities for GPU acceleration.
Handles CUDA verification, optimization, and memory management.
"""

import torch
import logging
import gc

logger = logging.getLogger(__name__)


class CUDAManager:
    """Manages CUDA/GPU resources and optimizations."""
    
    @staticmethod
    def verify_and_setup():
        """
        Verify CUDA availability and setup optimizations.
        
        Returns:
            bool: True if CUDA is available and working, False otherwise
        """
        logger.info("\n" + "=" * 70)
        logger.info("CUDA VERIFICATION")
        
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA Available: {cuda_available}")
        
        if cuda_available:
            cuda_available = CUDAManager._test_gpu()
            if cuda_available:
                CUDAManager._enable_optimizations()
        else:
            CUDAManager._log_cpu_warning()
        
        logger.info("=" * 70 + "\n")
        return cuda_available
    
    @staticmethod
    def _test_gpu():
        """Test GPU functionality with a simple operation."""
        try:
            test_tensor = torch.randn(100, 100).cuda()
            _ = test_tensor @ test_tensor.T
            logger.info("✓ GPU test successful")
            del test_tensor
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            logger.error(f"✗ GPU test failed: {e}")
            return False
    
    @staticmethod
    def _log_cpu_warning():
        """Log warning when CUDA is not available."""
        logger.warning("✗ CUDA not available - will use CPU (much slower)")

    @staticmethod
    def _enable_optimizations():
        """Enable CUDA optimizations for better performance."""
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        logger.info("✓ CUDA optimizations enabled (TF32, cuDNN)")
    
    @staticmethod
    def clear_memory():
        """Clear CUDA cache and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
