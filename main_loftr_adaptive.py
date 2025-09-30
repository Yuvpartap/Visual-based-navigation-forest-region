"""
Adaptive LoFTR matching with dynamic resize based on grid size.

This version automatically adjusts resize_max based on grid size to maintain
consistent scale between drone frame and stitched satellite tiles.
"""

import os
import sys
import cv2
import numpy as np
import logging
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


def calculate_adaptive_resize(grid_size, tile_size=512, target_scale_ratio=2.0):
    """
    Calculate optimal resize_max based on grid size to maintain scale consistency.
    
    Args:
        grid_size: N×N grid size
        tile_size: Approximate size of each tile in pixels
        target_scale_ratio: Target ratio between stitched size and resize_max
        
    Returns:
        Optimal resize_max value
    """
    # Estimate stitched image size
    estimated_stitched_size = grid_size * tile_size
    
    # Calculate resize_max to maintain consistent scale
    # We want the resized image to be large enough to preserve features
    resize_max = int(estimated_stitched_size / target_scale_ratio)
    
    # Clamp to reasonable bounds
    resize_max = max(640, min(resize_max, 2048))
    
    # Make divisible by 8 (LoFTR requirement)
    resize_max = (resize_max // 8) * 8
    
    logger.info(f"Grid size: {grid_size}×{grid_size}")
    logger.info(f"Estimated stitched size: {estimated_stitched_size}×{estimated_stitched_size}")
    logger.info(f"Adaptive resize_max: {resize_max}")
    logger.info(f"Scale ratio: {estimated_stitched_size / resize_max:.2f}x")
    
    return resize_max


def generate_dynamic_tile_matching_adaptive(
    tiles_dir,
    initial_tile_x,
    initial_tile_y,
    video_path,
    grid_size=5,
    frame_skip=15,
    ext=".png",
    tile_cache_items=512,
    min_good_matches=20,
    save_dir="results",
    adaptive_resize=True,
):
    """
    Dynamic tile matching with adaptive resize based on grid size.
    
    Args:
        tiles_dir: Directory containing satellite tiles
        initial_tile_x: Initial tile X coordinate
        initial_tile_y: Initial tile Y coordinate
        video_path: Path to drone video
        grid_size: N×N grid size for tile stitching
        frame_skip: Process every Nth frame
        ext: Tile file extension
        tile_cache_items: Maximum cached tiles
        min_good_matches: Minimum matches required
        save_dir: Directory for output visualizations
        adaptive_resize: Use adaptive resize based on grid size
    """
    centers = []
    traj_frame_paths = []
    min_x = initial_tile_x
    max_x = initial_tile_x
    min_y = initial_tile_y
    max_y = initial_tile_y
    cache = TileCache(max_items=tile_cache_items)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file {video_path}")
        return centers, traj_frame_paths

    frame_count = 0
    processed_frames = 0
    center_x = int(initial_tile_x)
    center_y = int(initial_tile_y)

    # Calculate adaptive resize_max based on grid size
    if adaptive_resize:
        resize_max = calculate_adaptive_resize(grid_size, tile_size=512, target_scale_ratio=2.0)
    else:
        resize_max = 840
    
    # Adjust RANSAC threshold based on grid size
    # Larger grids need more lenient thresholds due to scale differences
    if grid_size <= 5:
        ransac_threshold = 3.0
    elif grid_size <= 7:
        ransac_threshold = 4.0
    else:
        ransac_threshold = 5.0
    
    logger.info(f"Using RANSAC threshold: {ransac_threshold}")
    
    logger.info("Using DINOv2 + LoFTR with adaptive resize")
    matcher = create_dino_loftr_matcher(use_gpu=True, loftr_model="outdoor", resize_max=resize_max)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        processed_frames += 1
        logger.info(f"Processed frames: {processed_frames}")
        
        # Prepare stitched satellite image around current center
        stitched = load_and_stitch_grid(
            tiles_dir=tiles_dir,
            center_x=center_x,
            center_y=center_y,
            grid_size=grid_size,
            ext=ext,
            cache=cache,
        )
        if stitched is None:
            logger.warning(f"  Failed to load tiles at ({center_x}, {center_y})")
            frame_count += 1
            continue

        logger.info(f"  Stitched shape: {stitched.shape}, Frame shape: {frame.shape}")
        
        # Calculate actual scale ratio
        scale_ratio = max(stitched.shape[:2]) / max(frame.shape[:2])
        logger.info(f"  Scale ratio (stitched/frame): {scale_ratio:.2f}x")
        
        # Warn if scale ratio is too large
        if scale_ratio > 4.0:
            logger.warning(f"  ⚠ Large scale ratio ({scale_ratio:.2f}x) may reduce matching accuracy")
            logger.warning(f"  Consider: reducing grid_size or increasing resize_max")

        # Match using DINOv2 + LoFTR
        src_pts, dst_pts, confidence, num_matches = matcher.match_images(
            frame,
            stitched,
            cache_key0=None,
            cache_key1=None,
            min_matches=min_good_matches,
        )

        if num_matches >= min_good_matches:
            # Compute homography with adaptive RANSAC threshold
            H, mask, num_inliers = matcher.compute_homography_with_confidence(
                src_pts, dst_pts, confidence, ransac_threshold=ransac_threshold
            )
            
            if H is not None and num_inliers >= min_good_matches:
                h, w = frame.shape[:2]
                corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                transformed = cv2.perspectiveTransform(corners, H)
                cx = float(sum(pt[0][0] for pt in transformed) / 4.0)
                cy = float(sum(pt[0][1] for pt in transformed) / 4.0)

                # Offset relative to stitched center
                stitched_h, stitched_w = stitched.shape[:2]
                dx = cx - (stitched_w * 0.5)
                dy = cy - (stitched_h * 0.5)

                # Estimate tile size and update center
                tile_px = int(round(stitched_h / grid_size))
                center_x, center_y = update_center_from_match((center_x, center_y), (dx, dy), tile_px)
                
                # Track bounds
                min_x = min(min_x, center_x)
                max_x = max(max_x, center_x)
                min_y = min(min_y, center_y)
                max_y = max(max_y, center_y)

                logger.info(f"  ✓ Matched: {num_matches} features, {num_inliers} inliers -> tile ({center_x}, {center_y})")

                # Save match visualization
                kpts0 = src_pts.reshape(-1, 2)
                kpts1 = dst_pts.reshape(-1, 2)
                
                vis = matcher.draw_matches_visualization(
                    frame, stitched, kpts0, kpts1, mask, confidence
                )
                
                match_name = os.path.join(save_dir, f"match_adaptive_{processed_frames:05d}.png")
                cv2.imwrite(match_name, vis)
                logger.info(f"  ✓ Saved: {match_name}")
            else:
                logger.warning(f"  ✗ Homography failed: {num_inliers} inliers (need {min_good_matches})")
        else:
            logger.warning(f"  ✗ Insufficient matches: {num_matches} < {min_good_matches}")

        centers.append((center_x, center_y))
        frame_count += 1

    cap.release()
    
    # Build full-extent mosaic and draw trajectory
    logger.info(f"Building full trajectory mosaic: tiles ({min_x},{min_y}) to ({max_x},{max_y})")
    full = load_and_stitch_rect(
        tiles_dir=tiles_dir,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        z=18,
        ext=ext,
        cache=cache,
    )
    
    traj_frames = []
    if full is not None and centers:
        tile_px = int(round(full.shape[0] / (max_y - min_y + 1)))
        def tile_to_pixel(tx, ty):
            px = (tx - min_x) * tile_px + tile_px // 2
            py = (ty - min_y) * tile_px + tile_px // 2
            return (int(px), int(py))

        temp = full.copy()
        pts = [tile_to_pixel(x, y) for (x, y) in centers]
        for i in range(1, len(pts)):
            cv2.line(temp, pts[i - 1], pts[i], (255, 0, 0), 3)
            cv2.circle(temp, pts[i - 1], 5, (0, 255, 0), -1)
            frame_path = os.path.join(save_dir, f"traj_{i:05d}.png")
            cv2.imwrite(frame_path, temp)
            traj_frames.append(frame_path)
        
        cv2.circle(temp, pts[0], 8, (0, 255, 0), -1)
        cv2.putText(temp, "START", (pts[0][0] + 10, pts[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.circle(temp, pts[-1], 8, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(save_dir, "trajectory_map_adaptive.png"), temp)
        logger.info(f"Trajectory map saved with {len(centers)} points")
    
    return centers, traj_frames


def main():
    # Configuration
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    input_video_path = r"C:\Binomial Technologies\Non GPS based Navigation\Earth_Studio\Ajabgarh_videos\Ajabgarh_right.mp4"
    tiles_dir = r"C:\Binomial Technologies\Non GPS based Navigation\NGBN\Satellite Dataset\ajabgarh_z18Tiles_osm"
    initial_tile_x = 186625
    initial_tile_y = 110502
    
    # Grid size - now works with larger values!
    grid_size = int(os.environ.get("GRID_SIZE", "7"))  # Try 5, 7, or 9
    
    if not os.path.exists(input_video_path):
        logger.error(f"Video file not found: {input_video_path}")
        sys.exit(1)
    
    if not os.path.exists(tiles_dir):
        logger.error(f"Tiles directory not found: {tiles_dir}")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("Adaptive DINOv2 + LoFTR Trajectory Tracking")
    logger.info("=" * 60)
    logger.info(f"Video: {input_video_path}")
    logger.info(f"Tiles: {tiles_dir}")
    logger.info(f"Initial position: ({initial_tile_x}, {initial_tile_y})")
    logger.info(f"Grid size: {grid_size}×{grid_size}")
    logger.info("=" * 60)
    
    # Get video info
    video_info = get_video_info(input_video_path)
    if video_info:
        frame_skip = max(1, int(video_info['fps'] / 2))
        logger.info(f"Video: {video_info['resolution']}, {video_info['fps']:.1f} FPS")
        logger.info(f"Using frame skip: {frame_skip}")
    else:
        frame_skip = 15
        logger.warning(f"Could not get video info, using default frame_skip={frame_skip}")
    
    # Run adaptive LoFTR matching
    centers, _ = generate_dynamic_tile_matching_adaptive(
        tiles_dir=tiles_dir,
        initial_tile_x=initial_tile_x,
        initial_tile_y=initial_tile_y,
        video_path=input_video_path,
        grid_size=grid_size,
        frame_skip=frame_skip,
        ext=".png",
        tile_cache_items=1024,
        min_good_matches=20,
        save_dir=results_dir,
        adaptive_resize=True,  # Enable adaptive resize
    )
    
    logger.info("=" * 60)
    logger.info(f"✓ Processing complete!")
    logger.info(f"  Total trajectory points: {len(centers)}")
    logger.info(f"  Results saved to: {results_dir}/")
    logger.info(f"  - Match visualizations: match_adaptive_*.png")
    logger.info(f"  - Final trajectory map: trajectory_map_adaptive.png")
    logger.info("=" * 60)
    
    # Provide recommendations
    success_rate = len([c for c in centers if c != (initial_tile_x, initial_tile_y)]) / max(1, len(centers)) * 100
    logger.info(f"Success rate: {success_rate:.1f}%")
    
    if success_rate < 50:
        logger.warning("⚠ Low success rate. Try:")
        logger.warning("  - Reduce grid_size (e.g., from 9 to 7 or 5)")
        logger.warning("  - Increase frame_skip to process easier frames")
        logger.warning("  - Check if tiles are loading correctly")


if __name__ == "__main__":
    main()
