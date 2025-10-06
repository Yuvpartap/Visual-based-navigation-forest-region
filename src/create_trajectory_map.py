import os
import re
import cv2
import numpy as np
import logging
from typing import Optional
from src.tile_loading_utilis import (
    TileCache,
    load_and_stitch_grid,
    update_center_from_match,
    load_and_stitch_rect,
)
from src.superpoint_lightglue_matcher import SuperPointLightGlueMatcher, create_matcher
from src.dino_loftr_matcher import DinoLoFTRMatcher, create_dino_loftr_matcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#numerical sort (1, 2, 3) instead of (1, 10, 2)
def numerical_sort(value):
    return int(re.sub(r'\D', '', value))

#backward compatibility function for crop directory processing
def generate_trajectory_map_from_crops(
    map_path,
    frames_dir,
    output_path,
    use_superpoint: bool = True,
    min_matches: int = 10,
):
    """
    Generate trajectory map from crop directory.
    
    Args:
        map_path: Path to global map image
        frames_dir: Directory containing frame images
        output_path: Output path for trajectory map
        use_superpoint: Use SuperPoint+LightGlue (True) or SIFT (False)
        min_matches: Minimum number of matches required
    """
    #load the global map image
    map_img = cv2.imread(map_path, cv2.IMREAD_COLOR)
    if map_img is None:
        logger.error(f"Failed to load map image: {map_path}")
        return []

    center_coords = []

    if use_superpoint:
        logger.info("Using SuperPoint + LightGlue for feature matching")
        matcher = create_matcher(use_gpu=True, max_keypoints=2048, match_threshold=0.2)
        
        #iterate over all frame images in the directory
        for frame_name in sorted(os.listdir(frames_dir), key=numerical_sort):
            if not frame_name.endswith(".png"):
                continue

            logger.info(f"Processing {frame_name}")

            frame_path = os.path.join(frames_dir, frame_name)
            frame_img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
            if frame_img is None:
                logger.warning(f"Failed to load frame: {frame_path}")
                continue

            # Match using SuperPoint + LightGlue
            src_pts, dst_pts, confidence, num_matches = matcher.match_images(
                frame_img,
                map_img,
                cache_key0=None,
                cache_key1='global_map',
                min_matches=min_matches,
            )

            if num_matches >= min_matches:
                # Compute homography with confidence weighting
                H, mask, num_inliers = matcher.compute_homography_with_confidence(
                    src_pts, dst_pts, confidence, ransac_threshold=5.0
                )
                
                if H is not None and num_inliers >= min_matches:
                    h, w = frame_img.shape[:2]
                    corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                    transformed = cv2.perspectiveTransform(corners, H)

                    #calculate the center point
                    cx = int(sum(pt[0][0] for pt in transformed) / 4)
                    cy = int(sum(pt[0][1] for pt in transformed) / 4)
                    center_coords.append((cx, cy))
                    logger.info(f"  Matched: {num_matches} features, {num_inliers} inliers -> ({cx}, {cy})")
                else:
                    logger.warning(f"  Failed homography: {num_inliers} inliers")
            else:
                logger.warning(f"  Insufficient matches: {num_matches} < {min_matches}")
    
    else:
        # Fallback to SIFT
        logger.info("Using SIFT for feature matching (legacy mode)")
        
        sift = cv2.SIFT_create()
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        kp_map, des_map = sift.detectAndCompute(map_img, None)

        for frame_name in sorted(os.listdir(frames_dir), key=numerical_sort):
            if not frame_name.endswith(".png"):
                continue

            logger.info(f"Processing {frame_name}")
            frame_path = os.path.join(frames_dir, frame_name)
            frame_img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
            frame_gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)

            kp_frame, des_frame = sift.detectAndCompute(frame_gray, None)
            if des_frame is None or len(kp_frame) < 4:
                continue

            matches = bf.knnMatch(des_frame, des_map, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

            if len(good_matches) >= min_matches:
                src_pts = np.float32([kp_frame[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_map[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if H is not None:
                    h, w = frame_gray.shape
                    corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                    transformed = cv2.perspectiveTransform(corners, H)

                    cx = int(sum(pt[0][0] for pt in transformed) / 4)
                    cy = int(sum(pt[0][1] for pt in transformed) / 4)
                    center_coords.append((cx, cy))

    #draw trajectory and save
    trajectory_img = _draw_trajectory_on_map(map_img, center_coords)
    cv2.imwrite(output_path, trajectory_img)
    logger.info(f"Trajectory map saved: {output_path} ({len(center_coords)} points)")
    return center_coords

def _draw_trajectory_on_map(map_img, center_coords):
    """
    Helper function to draw trajectory on map image.
    """
    trajectory_img = map_img.copy()
    
    #draw trajectory lines and points
    for i in range(1, len(center_coords)):
        cv2.line(trajectory_img, center_coords[i - 1], center_coords[i], (255, 0, 0), 3)
        cv2.circle(trajectory_img, center_coords[i - 1], 5, (0, 255, 0), -1)

    #mark the start and end positions
    if center_coords:
        cv2.circle(trajectory_img, center_coords[0], 8, (0, 255, 0), -1)
        cv2.putText(trajectory_img, "START", (center_coords[0][0] + 10, center_coords[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.circle(trajectory_img, center_coords[-1], 8, (0, 0, 255), -1)
        cv2.putText(trajectory_img, "END", (center_coords[-1][0] + 10, center_coords[-1][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    return trajectory_img

#main function to generate the trajectory map from video input
def generate_trajectory_map(
    map_path,
    video_path,
    output_path,
    frame_skip: int = 1,
    use_superpoint: bool = True,
    min_matches: int = 10,
    save_dir: str = "results",
):
    """
    Generate trajectory map from video input.
    
    Args:
        map_path: Path to global map image
        video_path: Path to input video
        output_path: Output path for trajectory map
        frame_skip: Process every Nth frame
        use_superpoint: Use SuperPoint+LightGlue (True) or SIFT (False)
        min_matches: Minimum number of matches required
        save_dir: Directory to save match visualizations
    """
    #load the global map image
    map_img = cv2.imread(map_path, cv2.IMREAD_COLOR)
    if map_img is None:
        logger.error(f"Failed to load map image: {map_path}")
        return []

    center_coords = []
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    #open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file {video_path}")
        return center_coords

    #get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Video info: {total_frames} frames, {fps:.2f} FPS")

    frame_count = 0
    processed_frames = 0

    if use_superpoint:
        logger.info("Using SuperPoint + LightGlue for feature matching")
        matcher = create_matcher(use_gpu=True, max_keypoints=2048, match_threshold=0.2)

        #process video frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            #skip frames based on frame_skip parameter
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            processed_frames += 1
            logger.info(f"Processing frame {frame_count + 1}/{total_frames} (processed: {processed_frames})")

            # Match using SuperPoint + LightGlue
            src_pts, dst_pts, confidence, num_matches = matcher.match_images(
                frame,
                map_img,
                cache_key0=None,
                cache_key1='global_map',
                min_matches=min_matches,
            )

            if num_matches >= min_matches:
                # Compute homography with confidence weighting
                H, mask, num_inliers = matcher.compute_homography_with_confidence(
                    src_pts, dst_pts, confidence, ransac_threshold=5.0
                )
                
                if H is not None and num_inliers >= min_matches:
                    h, w = frame.shape[:2]
                    corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                    transformed = cv2.perspectiveTransform(corners, H)

                    cx = int(sum(pt[0][0] for pt in transformed) / 4)
                    cy = int(sum(pt[0][1] for pt in transformed) / 4)
                    center_coords.append((cx, cy))
                    logger.info(f"  Matched: {num_matches} features, {num_inliers} inliers -> ({cx}, {cy})")
                    
                    # Save match visualization with inlier lines
                    # Extract keypoints as (N, 2) arrays
                    kpts0 = src_pts.reshape(-1, 2)
                    kpts1 = dst_pts.reshape(-1, 2)
                    
                    logger.info(f"  Creating visualization with {len(kpts0)} keypoints, {int(mask.sum())} inliers")
                    logger.info(f"  Frame shape: {frame.shape}, Map shape: {map_img.shape}")
                    
                    # Draw side-by-side visualization with match lines
                    vis = matcher.draw_matches_visualization(
                        frame, map_img, kpts0, kpts1, mask, confidence
                    )
                    
                    logger.info(f"  Visualization shape: {vis.shape}")
                    
                    match_path = os.path.join(save_dir, f"match_{processed_frames:05d}.png")
                    success = cv2.imwrite(match_path, vis)
                    
                    if success:
                        logger.info(f"  ✓ Saved match visualization: {match_path}")
                    else:
                        logger.error(f"  ✗ Failed to save match visualization: {match_path}")
                else:
                    logger.warning(f"  Failed homography: {num_inliers} inliers")
            else:
                logger.warning(f"  Insufficient matches: {num_matches} < {min_matches}")

            frame_count += 1

    else:
        # Fallback to SIFT
        logger.info("Using SIFT for feature matching (legacy mode)")
        sift = cv2.SIFT_create()
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        kp_map, des_map = sift.detectAndCompute(map_img, None)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            processed_frames += 1
            logger.info(f"Processing frame {frame_count + 1}/{total_frames} (processed: {processed_frames})")

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp_frame, des_frame = sift.detectAndCompute(frame_gray, None)
            if des_frame is None or len(kp_frame) < 4:
                frame_count += 1
                continue

            matches = bf.knnMatch(des_frame, des_map, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

            if len(good_matches) >= min_matches:
                src_pts = np.float32([kp_frame[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_map[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if H is not None:
                    h, w = frame_gray.shape
                    corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                    transformed = cv2.perspectiveTransform(corners, H)

                    cx = int(sum(pt[0][0] for pt in transformed) / 4)
                    cy = int(sum(pt[0][1] for pt in transformed) / 4)
                    center_coords.append((cx, cy))

            frame_count += 1

    #release video capture
    cap.release()
    logger.info(f"Processed {processed_frames} frames, found {len(center_coords)} valid positions")

    #draw trajectory and save
    trajectory_img = _draw_trajectory_on_map(map_img, center_coords)
    cv2.imwrite(output_path, trajectory_img)
    return center_coords


def generate_dynamic_tile_matching(
    tiles_dir,
    initial_tile_x,
    initial_tile_y,
    video_path,
    grid_size= 9,
    frame_skip=15,
    ext=".png",
    tile_cache_items=512,
    min_good_matches=10,
    save_dir="results",
    use_superpoint: bool = True,
):
    """
    Perform dynamic satellite tile loading and matching against drone video frames.
    
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
        use_superpoint: Use SuperPoint+LightGlue (True) or SIFT (False)
        
    Returns:
        Tuple of (centers, trajectory_frame_paths)
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

    if use_superpoint:
        logger.info("Using SuperPoint + LightGlue for dynamic tile matching")
        matcher = create_matcher(use_gpu=True, max_keypoints=2048, match_threshold=0.2)

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

            # Match using SuperPoint + LightGlue
            src_pts, dst_pts, confidence, num_matches = matcher.match_images(
                frame,
                stitched,
                cache_key0=None,
                cache_key1=None,  # Don't cache stitched tiles (they change)
                min_matches=min_good_matches,
            )

            if num_matches >= min_good_matches:
                # Compute homography with confidence weighting
                H, mask, num_inliers = matcher.compute_homography_with_confidence(
                    src_pts, dst_pts, confidence, ransac_threshold=5.0
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
                    
                    logger.info(f"  Creating visualization with {len(kpts0)} keypoints, {int(mask.sum())} inliers")
                    logger.info(f"  Frame shape: {frame.shape}, Stitched shape: {stitched.shape}")
                    
                    # Draw side-by-side visualization with match lines
                    vis = matcher.draw_matches_visualization(
                        frame, stitched, kpts0, kpts1, mask, confidence
                    )
                    
                    logger.info(f"  Visualization shape: {vis.shape}")
                    
                    match_name = os.path.join(save_dir, f"match_{processed_frames:05d}.png")
                    success = cv2.imwrite(match_name, vis)
                    
                    if success:
                        logger.info(f"  ✓ Saved match visualization: {match_name}")
                    else:
                        logger.error(f"  ✗ Failed to save match visualization: {match_name}")

            centers.append((center_x, center_y))
            frame_count += 1

    else:
        # Fallback to SIFT
        logger.info("Using SIFT for dynamic tile matching (legacy mode)")
        sift = cv2.SIFT_create()
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            processed_frames += 1
            logger.info(f"Processed frames: {processed_frames}")
            
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

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp_frame, des_frame = sift.detectAndCompute(frame_gray, None)
            kp_sat, des_sat = sift.detectAndCompute(stitched, None)
            if des_frame is None or des_sat is None or len(kp_frame) < 4 or len(kp_sat) < 4:
                frame_count += 1
                continue

            matches = bf.knnMatch(des_frame, des_sat, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]
            if len(good) >= min_good_matches:
                src_pts = np.float32([kp_frame[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_sat[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if H is not None:
                    h, w = frame_gray.shape
                    corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                    transformed = cv2.perspectiveTransform(corners, H)
                    cx = float(sum(pt[0][0] for pt in transformed) / 4.0)
                    cy = float(sum(pt[0][1] for pt in transformed) / 4.0)

                    stitched_h, stitched_w = stitched.shape[:2]
                    dx = cx - (stitched_w * 0.5)
                    dy = cy - (stitched_h * 0.5)

                    tile_px = int(round(stitched_h / grid_size))
                    center_x, center_y = update_center_from_match((center_x, center_y), (dx, dy), tile_px)
                    
                    min_x = min(min_x, center_x)
                    max_x = max(max_x, center_x)
                    min_y = min(min_y, center_y)
                    max_y = max(max_y, center_y)

                    try:
                        inlier_mask = mask.ravel().tolist()
                        inlier_matches = [good[i] for i, v in enumerate(inlier_mask) if v == 1]
                    except Exception:
                        inlier_matches = good
                    vis = cv2.drawMatches(
                        frame_gray, kp_frame,
                        stitched, kp_sat,
                        inlier_matches,
                        None,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                    )
                    match_name = os.path.join(save_dir, f"match_{processed_frames:05d}.png")
                    cv2.imwrite(match_name, vis)

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
        z=19,
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
        cv2.imwrite(os.path.join(save_dir, "trajectory_map.png"), temp)
        logger.info(f"Trajectory map saved with {len(centers)} points")
    
    return centers, traj_frames
