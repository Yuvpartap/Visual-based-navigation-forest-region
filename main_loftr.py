"""
Main script using DINOv2 + LoFTR for trajectory tracking in challenging environments (dense forests, low-texture areas).

This version uses:
- DINOv2 for global image retrieval (optional)
- LoFTR for dense local matching (detector-free)
- RANSAC for robust homography estimation

Advantages over SuperPoint+LightGlue:
- Better performance in low-texture areas (forests, fields, water)
- Dense matching without keypoint detection
- More robust to extreme viewpoint changes
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


def generate_dynamic_tile_matching_loftr(
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
):
    """
    Dynamic tile matching using DINOv2 + LoFTR.
    
    Args:
        tiles_dir: Directory containing satellite tiles
        initial_tile_x: Initial tile X coordinate
        initial_tile_y: Initial tile Y coordinate
        video_path: Path to drone video
        grid_size: N×N grid size for tile stitching
        frame_skip: Process every Nth frame
        ext: Tile file extension
        tile_cache_items: Maximum cached tiles
        min_good_matches: Minimum matches required (LoFTR produces many matches)
        save_dir: Directory for output visualizations
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

    logger.info("Using DINOv2 + LoFTR for dynamic tile matching")
    matcher = create_dino_loftr_matcher(use_gpu=True, loftr_model="outdoor", resize_max=840)

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
            frame_count += 1
            continue

        # Match using DINOv2 + LoFTR
        src_pts, dst_pts, confidence, num_matches = matcher.match_images(
            frame,
            stitched,
            cache_key0=None,
            cache_key1=None,  # Don't cache stitched tiles (they change)
            min_matches=min_good_matches,
        )

        if num_matches >= min_good_matches:
            # Compute homography with confidence weighting
            # Use lower RANSAC threshold for LoFTR (denser matches)
            H, mask, num_inliers = matcher.compute_homography_with_confidence(
                src_pts, dst_pts, confidence, ransac_threshold=3.0
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

                logger.info(f"  Matched: {num_matches} features, {num_inliers} inliers -> tile ({center_x}, {center_y})")

                # Save match visualization with inlier lines
                kpts0 = src_pts.reshape(-1, 2)
                kpts1 = dst_pts.reshape(-1, 2)
                
                logger.info(f"  Creating LoFTR visualization with {len(kpts0)} keypoints, {int(mask.sum())} inliers")
                logger.info(f"  Frame shape: {frame.shape}, Stitched shape: {stitched.shape}")
                
                # Draw side-by-side visualization with match lines
                vis = matcher.draw_matches_visualization(
                    frame, stitched, kpts0, kpts1, mask, confidence
                )
                
                logger.info(f"  Visualization shape: {vis.shape}")
                
                match_name = os.path.join(save_dir, f"match_loftr_{processed_frames:05d}.png")
                success = cv2.imwrite(match_name, vis)
                
                if success:
                    logger.info(f"  ✓ Saved LoFTR match visualization: {match_name}")
                else:
                    logger.error(f"  ✗ Failed to save match visualization: {match_name}")

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
        cv2.imwrite(os.path.join(save_dir, "trajectory_map_loftr.png"), temp)
        logger.info(f"Trajectory map saved with {len(centers)} points")
    
    return centers, traj_frames


def main():
    # Configuration
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    input_video_path = r"C:\Binomial Technologies\Non GPS based Navigation\Earth_Studio\Munnar_videos\Munnar_curve.mp4"
    tiles_dir = r"C:\Binomial Technologies\Non GPS based Navigation\NGBN\Satellite Dataset\munnar_z19Tiles_gmap"
    initial_tile_x = 374279
    initial_tile_y = 247460
    grid_size = 7  # Smaller grid for LoFTR (it handles larger areas better)
    
    if not os.path.exists(input_video_path):
        logger.error(f"Video file not found: {input_video_path}")
        sys.exit(1)
    
    if not os.path.exists(tiles_dir):
        logger.error(f"Tiles directory not found: {tiles_dir}")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("DINOv2 + LoFTR Trajectory Tracking")
    logger.info("=" * 60)
    logger.info(f"Video: {input_video_path}")
    logger.info(f"Tiles: {tiles_dir}")
    logger.info(f"Initial position: ({initial_tile_x}, {initial_tile_y})")
    logger.info(f"Grid size: {grid_size}x{grid_size}")
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
    
    # Run LoFTR matching
    centers, _ = generate_dynamic_tile_matching_loftr(
        tiles_dir=tiles_dir,
        initial_tile_x=initial_tile_x,
        initial_tile_y=initial_tile_y,
        video_path=input_video_path,
        grid_size=grid_size,
        frame_skip=frame_skip,
        ext=".png",
        tile_cache_items=1024,
        min_good_matches=20,  # LoFTR produces many matches
        save_dir=results_dir,
    )
    
    logger.info("=" * 60)
    logger.info(f"✓ Processing complete!")
    logger.info(f"  Total trajectory points: {len(centers)}")
    logger.info(f"  Results saved to: {results_dir}/")
    logger.info(f"  - Match visualizations: match_loftr_*.png")
    logger.info(f"  - Final trajectory map: trajectory_map_loftr.png")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
