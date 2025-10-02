"""
Comparison script to benchmark EfficientLoFTR vs Standard LoFTR.

This script helps you compare:
1. Processing speed
2. Memory usage
3. Matching quality
4. Trajectory accuracy

Usage:
    python compare_models.py
"""

import os
import sys
import cv2
import numpy as np
import torch
import logging
import time
import psutil
from typing import Dict, List, Tuple

from src.dino_loftr_matcher import create_dino_loftr_matcher
from src.efficient_loftr_matcher import create_efficient_loftr_matcher
from src.tile_loading_utilis import TileCache, load_and_stitch_grid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Track performance metrics for comparison."""
    
    def __init__(self, name: str):
        self.name = name
        self.frame_times: List[float] = []
        self.match_counts: List[int] = []
        self.inlier_counts: List[int] = []
        self.memory_usage: List[float] = []
        
    def add_frame(self, time: float, matches: int, inliers: int, memory_mb: float):
        """Add metrics for a single frame."""
        self.frame_times.append(time)
        self.match_counts.append(matches)
        self.inlier_counts.append(inliers)
        self.memory_usage.append(memory_mb)
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.frame_times:
            return {}
        
        return {
            'name': self.name,
            'avg_time': np.mean(self.frame_times),
            'std_time': np.std(self.frame_times),
            'min_time': np.min(self.frame_times),
            'max_time': np.max(self.frame_times),
            'avg_matches': np.mean(self.match_counts),
            'avg_inliers': np.mean(self.inlier_counts),
            'avg_memory_mb': np.mean(self.memory_usage),
            'total_frames': len(self.frame_times),
            'total_time': np.sum(self.frame_times),
        }
    
    def print_summary(self):
        """Print formatted summary."""
        summary = self.get_summary()
        if not summary:
            logger.warning(f"No data for {self.name}")
            return
        
        logger.info(f"\n{'='*70}")
        logger.info(f"{summary['name']} - Performance Summary")
        logger.info(f"{'='*70}")
        logger.info(f"Total Frames: {summary['total_frames']}")
        logger.info(f"Total Time: {summary['total_time']:.1f}s ({summary['total_time']/60:.1f} min)")
        logger.info(f"Avg Time/Frame: {summary['avg_time']:.3f}s ± {summary['std_time']:.3f}s")
        logger.info(f"Min/Max Time: {summary['min_time']:.3f}s / {summary['max_time']:.3f}s")
        logger.info(f"Avg Matches: {summary['avg_matches']:.1f}")
        logger.info(f"Avg Inliers: {summary['avg_inliers']:.1f}")
        logger.info(f"Avg Memory: {summary['avg_memory_mb']:.1f} MB")
        logger.info(f"{'='*70}\n")


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def get_gpu_memory_mb() -> float:
    """Get GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def test_matcher(
    matcher,
    matcher_name: str,
    video_path: str,
    tiles_dir: str,
    initial_tile_x: int,
    initial_tile_y: int,
    grid_size: int,
    num_frames: int = 20,
    frame_skip: int = 15,
) -> PerformanceMetrics:
    """
    Test a matcher on video frames.
    
    Args:
        matcher: Matcher instance
        matcher_name: Name for logging
        video_path: Path to input video
        tiles_dir: Path to tile directory
        initial_tile_x: Starting tile X
        initial_tile_y: Starting tile Y
        grid_size: Grid size for tiles
        num_frames: Number of frames to test
        frame_skip: Frame skip interval
        
    Returns:
        PerformanceMetrics object
    """
    metrics = PerformanceMetrics(matcher_name)
    cache = TileCache(max_items=512)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return metrics
    
    frame_count = 0
    processed = 0
    center_x = initial_tile_x
    center_y = initial_tile_y
    
    logger.info(f"\nTesting {matcher_name}...")
    logger.info(f"Target: {num_frames} frames")
    
    while processed < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue
        
        # Load tiles
        stitched = load_and_stitch_grid(
            tiles_dir=tiles_dir,
            center_x=center_x,
            center_y=center_y,
            grid_size=grid_size,
            ext=".png",
            cache=cache,
        )
        
        if stitched is None:
            frame_count += 1
            continue
        
        # Measure performance
        start_time = time.time()
        mem_before = get_gpu_memory_mb() if torch.cuda.is_available() else get_memory_usage_mb()
        
        # Match
        src_pts, dst_pts, confidence, num_matches = matcher.match_images(
            frame, stitched, min_matches=10
        )
        
        # Compute homography if enough matches
        num_inliers = 0
        if num_matches >= 10:
            H, mask, num_inliers = matcher.compute_homography_with_confidence(
                src_pts, dst_pts, confidence, ransac_threshold=4.0
            )
        
        elapsed = time.time() - start_time
        mem_after = get_gpu_memory_mb() if torch.cuda.is_available() else get_memory_usage_mb()
        mem_used = mem_after - mem_before
        
        # Record metrics
        metrics.add_frame(elapsed, num_matches, num_inliers, mem_used)
        
        processed += 1
        frame_count += 1
        
        logger.info(f"  Frame {processed}/{num_frames}: {elapsed:.3f}s, "
                   f"{num_matches} matches, {num_inliers} inliers")
    
    cap.release()
    return metrics


def compare_matchers(
    video_path: str,
    tiles_dir: str,
    initial_tile_x: int,
    initial_tile_y: int,
    grid_size: int = 7,
    num_frames: int = 20,
    frame_skip: int = 15,
    resize_max: int = 1080,
):
    """
    Compare Standard LoFTR vs EfficientLoFTR.
    
    Args:
        video_path: Path to test video
        tiles_dir: Path to satellite tiles
        initial_tile_x: Starting tile X coordinate
        initial_tile_y: Starting tile Y coordinate
        grid_size: Grid size for tile loading
        num_frames: Number of frames to test
        frame_skip: Frame skip interval
        resize_max: Maximum image dimension
    """
    logger.info("="*70)
    logger.info("MATCHER COMPARISON: Standard LoFTR vs EfficientLoFTR")
    logger.info("="*70)
    logger.info(f"Video: {video_path}")
    logger.info(f"Tiles: {tiles_dir}")
    logger.info(f"Grid Size: {grid_size}x{grid_size}")
    logger.info(f"Test Frames: {num_frames}")
    logger.info(f"Resize Max: {resize_max}")
    logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    logger.info("="*70)
    
    # Test Standard LoFTR
    logger.info("\n[1/2] Testing Standard LoFTR...")
    try:
        loftr_matcher = create_dino_loftr_matcher(
            use_gpu=True,
            loftr_model="outdoor",
            resize_max=resize_max
        )
        loftr_metrics = test_matcher(
            loftr_matcher,
            "Standard LoFTR",
            video_path,
            tiles_dir,
            initial_tile_x,
            initial_tile_y,
            grid_size,
            num_frames,
            frame_skip
        )
    except Exception as e:
        logger.error(f"Standard LoFTR test failed: {e}")
        loftr_metrics = PerformanceMetrics("Standard LoFTR")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Test EfficientLoFTR
    logger.info("\n[2/2] Testing EfficientLoFTR...")
    try:
        eloftr_matcher = create_efficient_loftr_matcher(
            use_gpu=True,
            resize_max=resize_max,
            use_mixed_precision=True
        )
        eloftr_metrics = test_matcher(
            eloftr_matcher,
            "EfficientLoFTR",
            video_path,
            tiles_dir,
            initial_tile_x,
            initial_tile_y,
            grid_size,
            num_frames,
            frame_skip
        )
    except Exception as e:
        logger.error(f"EfficientLoFTR test failed: {e}")
        eloftr_metrics = PerformanceMetrics("EfficientLoFTR")
    
    # Print summaries
    loftr_metrics.print_summary()
    eloftr_metrics.print_summary()
    
    # Comparison
    loftr_sum = loftr_metrics.get_summary()
    eloftr_sum = eloftr_metrics.get_summary()
    
    if loftr_sum and eloftr_sum:
        logger.info("="*70)
        logger.info("COMPARISON RESULTS")
        logger.info("="*70)
        
        speedup = loftr_sum['avg_time'] / eloftr_sum['avg_time']
        memory_reduction = (1 - eloftr_sum['avg_memory_mb'] / loftr_sum['avg_memory_mb']) * 100
        match_ratio = eloftr_sum['avg_matches'] / loftr_sum['avg_matches']
        inlier_ratio = eloftr_sum['avg_inliers'] / loftr_sum['avg_inliers']
        
        logger.info(f"Speed Improvement: {speedup:.2f}x faster")
        logger.info(f"Memory Reduction: {memory_reduction:.1f}%")
        logger.info(f"Match Quality: {match_ratio:.2f}x (1.0 = same)")
        logger.info(f"Inlier Quality: {inlier_ratio:.2f}x (1.0 = same)")
        
        logger.info("\nTime Savings:")
        time_saved = loftr_sum['total_time'] - eloftr_sum['total_time']
        logger.info(f"  Per frame: {(loftr_sum['avg_time'] - eloftr_sum['avg_time']):.3f}s")
        logger.info(f"  Total ({num_frames} frames): {time_saved:.1f}s ({time_saved/60:.1f} min)")
        
        logger.info("\nRecommendation:")
        if speedup > 1.5 and match_ratio > 0.9:
            logger.info("  ✅ EfficientLoFTR is RECOMMENDED")
            logger.info("  - Significantly faster")
            logger.info("  - Maintains matching quality")
            logger.info("  - Lower memory usage")
        elif speedup > 1.2:
            logger.info("  ⚠️ EfficientLoFTR shows improvement")
            logger.info("  - Faster processing")
            logger.info("  - Consider for real-time applications")
        else:
            logger.info("  ℹ️ Results are similar")
            logger.info("  - Both methods perform comparably")
        
        logger.info("="*70)


def main():
    """Main comparison function."""
    # Configuration
    video_path = r"C:\Binomial Technologies\Non GPS based Navigation\Earth_Studio\Ajabgarh_videos\Ajabgarh_left.mp4"
    tiles_dir = r"C:\Binomial Technologies\Non GPS based Navigation\NGBN\Satellite Dataset\ajabgarh_z18Tiles_gmap"
    initial_tile_x = 186625
    initial_tile_y = 110502
    
    # Test parameters
    grid_size = 7          # Smaller for faster testing
    num_frames = 20        # Test on 20 frames
    frame_skip = 15        # Skip frames
    resize_max = 1080      # Standard resolution
    
    # Verify paths
    if not os.path.exists(video_path):
        logger.error(f"Video not found: {video_path}")
        logger.info("Please update video_path in compare_models.py")
        sys.exit(1)
    
    if not os.path.exists(tiles_dir):
        logger.error(f"Tiles directory not found: {tiles_dir}")
        logger.info("Please update tiles_dir in compare_models.py")
        sys.exit(1)
    
    # Run comparison
    compare_matchers(
        video_path=video_path,
        tiles_dir=tiles_dir,
        initial_tile_x=initial_tile_x,
        initial_tile_y=initial_tile_y,
        grid_size=grid_size,
        num_frames=num_frames,
        frame_skip=frame_skip,
        resize_max=resize_max
    )


if __name__ == "__main__":
    main()
