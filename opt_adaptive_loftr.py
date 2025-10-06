"""
Jetson Nano Orin 4GB Optimized LoFTR - Perfect Balance of Speed & Accuracy

Optimizations:
1. CUDA detection and verification
2. Original adaptive resizing (target_scale_ratio=2.0)
3. Efficient frame saving (every 3rd matched)
4. No forced downsampling
5. Aggressive tile cache (128) + moderate feature cache (50)
6. CUDA optimizations (TF32, cuDNN)
7. Proper error handling for trajectory map
"""

import os
import sys
import cv2
import numpy as np
import torch
import logging
import time
import gc
from collections import deque
from src.tile_loading_utilis import (
    TileCache,
    load_and_stitch_grid,
    update_center_from_match,
    load_and_stitch_rect,
)
from src.dino_loftr_matcher import create_dino_loftr_matcher
from src.video_utils import get_video_info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_cuda_setup():
    """Verify CUDA is properly configured for Jetson."""
    logger.info("\n" + "=" * 70)
    logger.info("CUDA VERIFICATION")
    logger.info("=" * 70)
    
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU Count: {torch.cuda.device_count()}")
        logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Test GPU
        try:
            test_tensor = torch.randn(100, 100).cuda()
            _ = test_tensor @ test_tensor.T
            logger.info("✓ GPU test successful")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"✗ GPU test failed: {e}")
            cuda_available = False
    else:
        logger.warning("✗ CUDA not available - will use CPU (much slower)")
        logger.warning("  Check: Is PyTorch installed with CUDA support?")
        logger.warning("  Install: pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    logger.info("=" * 70 + "\n")
    return cuda_available


class JetsonOptimizedLoFTRMatcher:
    """
    Jetson Nano Orin optimized wrapper with perfect speed/accuracy balance.
    """
    
    def __init__(self, base_matcher, use_mixed_precision=True):
        self.matcher = base_matcher
        self.cuda_available = torch.cuda.is_available()
        self.use_mixed_precision = use_mixed_precision and self.cuda_available
        
        # Feature cache: 50 items (as requested)
        self.tile_feature_cache = {}
        self.max_cache_size = 50
        
        # Performance tracking
        self.timing_stats = {'total': [], 'matching': [], 'homography': []}
        
        # GPU optimizations if available
        if self.cuda_available:
            torch.cuda.empty_cache()
            # Enable TF32 for faster matmul
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            logger.info("✓ CUDA optimizations enabled (TF32, cuDNN)")
        
        if self.use_mixed_precision:
            logger.info("✓ FP16 mixed precision enabled")
        else:
            logger.info("✓ Using FP32 (CPU mode)")
    
    def clear_gpu_memory(self):
        """Aggressive GPU memory cleanup."""
        if self.cuda_available:
            torch.cuda.empty_cache()
        gc.collect()
    
    def match_with_optimizations(
        self,
        frame,
        stitched,
        min_matches=20,
    ):
        """
        Optimized matching with timing breakdown.
        """
        start_time = time.time()
        
        # Use mixed precision if available
        if self.use_mixed_precision:
            with torch.cuda.amp.autocast():
                match_start = time.time()
                src_pts, dst_pts, confidence, num_matches = self.matcher.match_images(
                    frame, stitched, min_matches=min_matches
                )
                self.timing_stats['matching'].append(time.time() - match_start)
        else:
            match_start = time.time()
            src_pts, dst_pts, confidence, num_matches = self.matcher.match_images(
                frame, stitched, min_matches=min_matches
            )
            self.timing_stats['matching'].append(time.time() - match_start)
        
        total_time = time.time() - start_time
        self.timing_stats['total'].append(total_time)
        
        return src_pts, dst_pts, confidence, num_matches
    
    def get_average_timing(self):
        """Get average timing statistics."""
        if not self.timing_stats['total']:
            return 0.0
        return np.mean(self.timing_stats['total'])
    
    def get_timing_breakdown(self):
        """Get detailed timing breakdown."""
        if not self.timing_stats['total']:
            return {}
        return {
            'total': np.mean(self.timing_stats['total']),
            'matching': np.mean(self.timing_stats['matching']) if self.timing_stats['matching'] else 0,
            'homography': np.mean(self.timing_stats['homography']) if self.timing_stats['homography'] else 0,
        }
    
    def clear_old_cache(self, force=False):
        """Clear cache when needed."""
        if force or len(self.tile_feature_cache) > self.max_cache_size:
            self.tile_feature_cache.clear()
            self.clear_gpu_memory()


def calculate_adaptive_resize(grid_size, tile_size=256, target_scale_ratio=2.0):
    """
    Calculate optimal resize_max based on grid size.
    ORIGINAL VERSION - target_scale_ratio=2.0 for best accuracy.
    """
    estimated_stitched_size = grid_size * tile_size
    resize_max = int(estimated_stitched_size / target_scale_ratio)
    
    # Original bounds
    resize_max = max(640, min(resize_max, 2048))
    resize_max = (resize_max // 8) * 8
    
    return resize_max


def generate_dynamic_tile_matching_jetson(
    tiles_dir,
    initial_tile_x,
    initial_tile_y,
    video_path,
    grid_size,
    frame_skip,
    ext=".png",
    tile_cache_items=256,      # Aggressive: 128
    min_good_matches=20,       # Standard: 20
    save_dir="results",
    adaptive_resize=True,
    cleanup_interval=8,        # Every 8 frames
    save_every_nth=3,
):
    """
    Jetson optimized with perfect balance of speed and accuracy.
    
    Configuration:
    - Adaptive resize: Original algorithm (scale_ratio=2.0)
    - Tile cache: 128 items (aggressive)
    - Feature cache: 50 items (moderate)
    - No forced downsampling
    - Cleanup every 8 frames
    - Save every 3rd matched frame
    """
    centers = []
    traj_frame_paths = []
    min_x = initial_tile_x
    max_x = initial_tile_x
    min_y = initial_tile_y
    max_y = initial_tile_y
    
    # Aggressive tile cache
    cache = TileCache(max_items=tile_cache_items)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file {video_path}")
        return centers, traj_frame_paths

    frame_count = 0
    processed_frames = 0
    matched_frames = 0
    center_x = int(initial_tile_x)
    center_y = int(initial_tile_y)

    # Calculate resize_max with ORIGINAL algorithm
    if adaptive_resize:
        resize_max = calculate_adaptive_resize(grid_size, tile_size=256, target_scale_ratio=2.0)
        logger.info(f"Adaptive resize_max calculated: {resize_max} (original algorithm)")
    else:
        resize_max = 840
    
    # Adjust RANSAC threshold
    if grid_size <= 5:
        ransac_threshold = 3.0
    elif grid_size <= 7:
        ransac_threshold = 4.0
    else:
        ransac_threshold = 5.0
    
    logger.info("=" * 70)
    logger.info("JETSON OPTIMIZED LoFTR - PERFECT BALANCE")
    logger.info("=" * 70)
    logger.info(f"Grid size: {grid_size}×{grid_size}")
    logger.info(f"Resize_max: {resize_max} (adaptive: {adaptive_resize})")
    logger.info(f"Frame preprocessing: No downsampling (original size)")
    logger.info(f"RANSAC threshold: {ransac_threshold}")
    logger.info(f"Tile cache: {tile_cache_items} items (aggressive)")
    logger.info(f"Feature cache: 50 items (moderate)")
    logger.info(f"Min good matches: {min_good_matches}")
    logger.info(f"Memory cleanup: every {cleanup_interval} frames")
    logger.info(f"Save visualizations: every {save_every_nth} matched frames")
    logger.info("=" * 70)
    
    # Verify CUDA
    cuda_ok = verify_cuda_setup()
    
    # Create matcher
    base_matcher = create_dino_loftr_matcher(
        use_gpu=cuda_ok, 
        loftr_model="outdoor", 
        resize_max=resize_max
    )
    matcher = JetsonOptimizedLoFTRMatcher(base_matcher, use_mixed_precision=cuda_ok)
    
    # Performance tracking
    frame_times = deque(maxlen=10)
    match_index = 0
    
    # Initial memory cleanup
    matcher.clear_gpu_memory()
    
    while True:
        frame_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        processed_frames += 1
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Frame {processed_frames} (size: {frame.shape[1]}x{frame.shape[0]}):")
        
        # Load tiles
        tile_start = time.time()
        stitched = load_and_stitch_grid(
            tiles_dir=tiles_dir,
            center_x=center_x,
            center_y=center_y,
            grid_size=grid_size,
            ext=ext,
            cache=cache,
        )
        tile_time = time.time() - tile_start
        
        if stitched is None:
            logger.warning(f"  ✗ Failed to load tiles")
            frame_count += 1
            continue

        # Optimized matching
        src_pts, dst_pts, confidence, num_matches = matcher.match_with_optimizations(
            frame, stitched, min_matches=min_good_matches,
        )

        if num_matches >= min_good_matches:
            # Compute homography
            homo_start = time.time()
            H, mask, num_inliers = matcher.matcher.compute_homography_with_confidence(
                src_pts, dst_pts, confidence, ransac_threshold=ransac_threshold
            )
            homo_time = time.time() - homo_start
            matcher.timing_stats['homography'].append(homo_time)
            
            if H is not None and num_inliers >= min_good_matches:
                h, w = frame.shape[:2]
                corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                transformed = cv2.perspectiveTransform(corners, H)
                cx = float(sum(pt[0][0] for pt in transformed) / 4.0)
                cy = float(sum(pt[0][1] for pt in transformed) / 4.0)

                # Update center
                stitched_h, stitched_w = stitched.shape[:2]
                dx = cx - (stitched_w * 0.5)
                dy = cy - (stitched_h * 0.5)
                tile_px = int(round(stitched_h / grid_size))
                center_x, center_y = update_center_from_match((center_x, center_y), (dx, dy), tile_px)
                
                # Track trajectory
                centers.append((center_x, center_y))
                min_x = min(min_x, center_x)
                max_x = max(max_x, center_x)
                min_y = min(min_y, center_y)
                max_y = max(max_y, center_y)
                
                matched_frames += 1

                # Calculate frame time
                frame_time = time.time() - frame_start
                frame_times.append(frame_time)
                avg_time = np.mean(frame_times)
                
                logger.info(f"  ✓ Matches: {num_matches}, Inliers: {num_inliers}")
                logger.info(f"  → Position: ({center_x}, {center_y})")
                logger.info(f"  ⚡ Timing: tile={tile_time:.2f}s, match={matcher.timing_stats['matching'][-1]:.2f}s, homo={homo_time:.2f}s")
                logger.info(f"  ⏱️  Total: {frame_time:.2f}s (avg: {avg_time:.2f}s, {1.0/avg_time:.2f} FPS)")

                # Save visualization every Nth matched frame
                if matched_frames % save_every_nth == 0:
                    kpts0 = src_pts.reshape(-1, 2)
                    kpts1 = dst_pts.reshape(-1, 2)
                    vis = matcher.matcher.draw_matches_visualization(
                        frame, stitched, kpts0, kpts1, mask, confidence
                    )
                    match_name = os.path.join(save_dir, f"match_{match_index:04d}.png")
                    cv2.imwrite(match_name, vis)
                    logger.info(f"  💾 Saved: match_{match_index:04d}.png")
                    match_index += 1
            else:
                logger.warning(f"  ✗ Homography failed: {num_inliers} inliers")
        else:
            logger.warning(f"  ✗ Insufficient matches: {num_matches}")

        frame_count += 1
        
        # Periodic memory cleanup (every 8 frames)
        if processed_frames % cleanup_interval == 0:
            matcher.clear_old_cache(force=False)
            logger.debug("  🧹 Memory cleanup performed")

    cap.release()
    
    # Final memory cleanup
    matcher.clear_gpu_memory()
    
    # Performance summary
    timing = matcher.get_timing_breakdown()
    logger.info("\n" + "=" * 70)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total frames processed: {processed_frames}")
    logger.info(f"Successful matches: {matched_frames} ({matched_frames/processed_frames*100:.1f}%)")
    logger.info(f"Average time per frame: {matcher.get_average_timing():.2f}s")
    if timing:
        logger.info(f"  - Matching: {timing['matching']:.2f}s")
        logger.info(f"  - Homography: {timing['homography']:.2f}s")
    if frame_times:
        avg_fps = 1.0 / np.mean(frame_times)
        logger.info(f"Recent average: {np.mean(frame_times):.2f}s ({avg_fps:.2f} FPS)")
    logger.info(f"Visualizations saved: {match_index}")
    logger.info("=" * 70)

    # Build trajectory map with proper error handling
    if centers:
        logger.info(f"\nBuilding trajectory map: tiles ({min_x},{min_y}) to ({max_x},{max_y})")
        
        try:
            full = load_and_stitch_rect(
                tiles_dir=tiles_dir,
                min_x=min_x,
                max_x=max_x,
                min_y=min_y,
                max_y=max_y,
                z=19,
                ext=ext,
                cache=cache,
            )
            
            if full is not None:
                tile_px = int(round(full.shape[0] / (max_y - min_y + 1)))
                
                def tile_to_pixel(tx, ty):
                    px = (tx - min_x) * tile_px + tile_px // 2
                    py = (ty - min_y) * tile_px + tile_px // 2
                    return (int(px), int(py))

                temp = full.copy()
                pts = [tile_to_pixel(x, y) for (x, y) in centers]
                
                # Draw trajectory
                for i in range(1, len(pts)):
                    cv2.line(temp, pts[i - 1], pts[i], (255, 0, 0), 3)
                    cv2.circle(temp, pts[i - 1], 5, (0, 255, 0), -1)
                
                cv2.circle(temp, pts[0], 8, (0, 255, 0), -1)
                cv2.putText(temp, "START", (pts[0][0] + 10, pts[0][1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.circle(temp, pts[-1], 8, (0, 0, 255), -1)
                
                traj_path = os.path.join(save_dir, "trajectory_map.png")
                cv2.imwrite(traj_path, temp)
                logger.info(f"✓ Trajectory map saved: {traj_path}")
                logger.info(f"  Total trajectory points: {len(centers)}")
            else:
                logger.warning("✗ Failed to create trajectory map: Could not stitch tiles")
                
        except Exception as e:
            logger.error(f"✗ Error creating trajectory map: {e}")
            logger.info("  Trajectory data saved but visualization failed")
    else:
        logger.warning("✗ No trajectory points to visualize")
    
    # Clean up cache properly
    if hasattr(cache, 'cache'):
        cache.cache.clear()
    
    return centers, traj_frame_paths


def main():
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    input_video_path = r"/home/jetson/Desktop/V5_Data/Videos/France_videos/France_nadir_0d_mainRd.mp4"
    tiles_dir = r"/home/jetson/Desktop/V5_Data/Dataset/france_z19Tiles_osm"
    initial_tile_x = 372408
    initial_tile_y = 221428
    
    # Perfect balance parameters
    grid_size = 5
    
    if not os.path.exists(input_video_path):
        logger.error(f"Video file not found: {input_video_path}")
        sys.exit(1)
    
    if not os.path.exists(tiles_dir):
        logger.error(f"Tiles directory not found: {tiles_dir}")
        sys.exit(1)
    
    # Get video info
    video_info = get_video_info(input_video_path)
    if video_info:
        frame_skip = max(1, int(video_info['fps'] / 2))  # Process ~2 FPS
        logger.info(f"Video: {video_info['resolution']}, {video_info['fps']:.1f} FPS")
        logger.info(f"Frame skip: {frame_skip} (processing ~{video_info['fps']/frame_skip:.1f} FPS)")
    else:
        frame_skip = 15
    
    start_time = time.time()
    
    centers, _ = generate_dynamic_tile_matching_jetson(
        tiles_dir=tiles_dir,
        initial_tile_x=initial_tile_x,
        initial_tile_y=initial_tile_y,
        video_path=input_video_path,
        grid_size=grid_size,
        frame_skip=frame_skip,
        ext=".png",
        tile_cache_items=128,         # Aggressive tile cache
        min_good_matches=20,          # Standard threshold
        save_dir=results_dir,
        adaptive_resize=True,         # Original algorithm
        cleanup_interval=8,           # Every 8 frames
        save_every_nth=3,             # Save every 3rd matched frame
    )
    
    total_time = time.time() - start_time
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ PROCESSING COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    logger.info(f"Trajectory points: {len(centers)}")
    if len(centers) > 0:
        logger.info(f"Average time per point: {total_time/len(centers):.2f}s")
        logger.info(f"Effective processing rate: {len(centers)/total_time:.2f} points/sec")
    logger.info(f"Results directory: {results_dir}/")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
