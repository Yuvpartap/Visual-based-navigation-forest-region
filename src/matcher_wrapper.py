"""
Matcher wrapper and adaptive resizing utilities.
Provides optimized matching with performance tracking and caching.
"""

import torch
import time
import logging
from src.performance_tracker import PerformanceTracker
from src.cuda_manager import CUDAManager

logger = logging.getLogger(__name__)


class JetsonOptimizedLoFTRMatcher:
    """
    Optimized wrapper for LoFTR matcher with performance tracking.
    Supports mixed precision and feature caching.
    """

    def __init__(self, base_matcher, cuda_available, feature_cache_size=50):
        """
        Initialize optimized matcher.
        
        Args:
            base_matcher: Base matcher instance
            cuda_available: Whether CUDA is available
            feature_cache_size: Maximum number of cached features
        """
        self.matcher = base_matcher
        self.cuda_available = cuda_available
        self.use_mixed_precision = cuda_available
        self.tile_feature_cache = {}
        self.max_cache_size = feature_cache_size
        self.tracker = PerformanceTracker()
        
        if self.use_mixed_precision:
            logger.info("✓ FP16 mixed precision enabled")
        else:
            logger.info("✓ Using FP32 (CPU mode)")
    
    def match(self, frame, stitched, min_matches=20):
        """
        Perform optimized matching with timing.
        
        Args:
            frame: Query frame
            stitched: Reference stitched image
            min_matches: Minimum number of matches required
            
        Returns:
            Tuple of (src_pts, dst_pts, confidence, num_matches)
        """
        start_time = time.time()
        
        if self.use_mixed_precision:
            with torch.cuda.amp.autocast():
                result = self._do_match(frame, stitched, min_matches)
        else:
            result = self._do_match(frame, stitched, min_matches)
        
        self.tracker.record('total', time.time() - start_time)
        return result
    
    def _do_match(self, frame, stitched, min_matches):
        """Execute the actual matching operation."""
        match_start = time.time()
        src_pts, dst_pts, confidence, num_matches = self.matcher.match_images(
            frame, stitched, min_matches=min_matches
        )
        self.tracker.record('matching', time.time() - match_start)
        return src_pts, dst_pts, confidence, num_matches
    
    def compute_homography(self, src_pts, dst_pts, confidence, ransac_threshold):
        """
        Compute homography with timing.
        
        Args:
            src_pts: Source points
            dst_pts: Destination points
            confidence: Match confidence scores
            ransac_threshold: RANSAC threshold
            
        Returns:
            Tuple of (H, mask, num_inliers)
        """
        homo_start = time.time()
        H, mask, num_inliers = self.matcher.compute_homography_with_confidence(
            src_pts, dst_pts, confidence, ransac_threshold=ransac_threshold
        )
        self.tracker.record('homography', time.time() - homo_start)
        return H, mask, num_inliers
    
    def clear_cache(self, force=False):
        """
        Clear feature cache when needed.
        
        Args:
            force: Force cache clearing regardless of size
        """
        if force or len(self.tile_feature_cache) > self.max_cache_size:
            self.tile_feature_cache.clear()
            CUDAManager.clear_memory()


class AdaptiveResizer:
    """Adaptive image resizing utilities for optimal performance."""

    @staticmethod
    def calculate_resize_max(grid_size, tile_size=256, target_scale_ratio=2.0):
        """
        Calculate optimal resize_max with original algorithm.
        
        Args:
            grid_size: Size of tile grid
            tile_size: Size of individual tiles in pixels
            target_scale_ratio: Target scale ratio for resizing
            
        Returns:
            Optimal resize_max value
        """
        estimated_stitched_size = grid_size * tile_size
        resize_max = int(estimated_stitched_size / target_scale_ratio)
        resize_max = max(640, min(resize_max, 2048))
        resize_max = (resize_max // 8) * 8
        return resize_max
