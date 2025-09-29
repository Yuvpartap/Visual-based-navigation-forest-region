import os
import re
import cv2
import numpy as np
from src.tile_loading_utilis import (
    TileCache,
    load_and_stitch_grid,
    update_center_from_match,
    load_and_stitch_rect,
)

#numerical sort (1, 2, 3) instead of (1, 10, 2)
def numerical_sort(value):
    return int(re.sub(r'\D', '', value))

#backward compatibility function for crop directory processing
def generate_trajectory_map_from_crops(map_path, frames_dir, output_path):
    """
    Generate trajectory map from crop directory (original functionality).
    Maintained for backward compatibility.
    """
    #load the global map image
    map_img = cv2.imread(map_path, cv2.IMREAD_COLOR)

    #initialize SIFT feature detector and BFMatcher
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    #detect keypoints and descriptors in the global map
    kp_map, des_map = sift.detectAndCompute(map_img, None)

    center_coords = []

    #iterate over all frame images in the directory
    for frame_name in sorted(os.listdir(frames_dir), key=numerical_sort):
        if not frame_name.endswith(".png"):
            continue

        #show progress
        print(f"Processing {frame_name}")

        frame_path = os.path.join(frames_dir, frame_name)
        frame_img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        frame_gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)

        #detect keypoints and descriptors
        kp_frame, des_frame = sift.detectAndCompute(frame_gray, None)
        if des_frame is None or len(kp_frame) < 4:
            continue

        #use KNN matching to find good matches
        matches = bf.knnMatch(des_frame, des_map, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        #proceed only if enough good matches are found
        if len(good_matches) > 10:
            src_pts = np.float32([kp_frame[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_map[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            #estimate homography matrix
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is not None:
                h, w = frame_gray.shape
                corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                transformed = cv2.perspectiveTransform(corners, H)

                #calculate the center point
                cx = int(sum(pt[0][0] for pt in transformed) / 4)
                cy = int(sum(pt[0][1] for pt in transformed) / 4)
                center_coords.append((cx, cy))

    #draw trajectory and save
    trajectory_img = _draw_trajectory_on_map(map_img, center_coords)
    cv2.imwrite(output_path, trajectory_img)
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
def generate_trajectory_map(map_path, video_path, output_path, frame_skip=1):
    #load the global map image
    map_img = cv2.imread(map_path, cv2.IMREAD_COLOR)

    #initialize SIFT feature detector and BFMatcher
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    #detect keypoints and descriptors in the global map
    kp_map, des_map = sift.detectAndCompute(map_img, None)

    center_coords = []

    #open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return center_coords

    #get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video info: {total_frames} frames, {fps:.2f} FPS")

    frame_count = 0
    processed_frames = 0

    #process video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #skip frames based on frame_skip parameter
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        #show progress
        processed_frames += 1
        print(f"Processing frame {frame_count + 1}/{total_frames} (processed: {processed_frames})")

        #convert frame to grayscale for feature detection
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #detect keypoints and descriptors
        kp_frame, des_frame = sift.detectAndCompute(frame_gray, None)
        if des_frame is None or len(kp_frame) < 4:
            frame_count += 1
            continue

        #use KNN matching to find good matches
        matches = bf.knnMatch(des_frame, des_map, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        #proceed only if enough good matches are found
        if len(good_matches) > 10:
            src_pts = np.float32([kp_frame[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_map[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            #estimate homography matrix
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is not None:
                h, w = frame_gray.shape
                corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                transformed = cv2.perspectiveTransform(corners, H)

                #calculate the center point
                cx = int(sum(pt[0][0] for pt in transformed) / 4)
                cy = int(sum(pt[0][1] for pt in transformed) / 4)
                center_coords.append((cx, cy))

        frame_count += 1

    #release video capture
    cap.release()
    print(f"Processed {processed_frames} frames, found {len(center_coords)} valid positions")

    #draw trajectory and save
    trajectory_img = _draw_trajectory_on_map(map_img, center_coords)
    cv2.imwrite(output_path, trajectory_img)
    return center_coords


def generate_dynamic_tile_matching(
    tiles_dir,
    initial_tile_x,
    initial_tile_y,
    video_path,
    grid_size=3,
    frame_skip=15,
    ext=".png",
    tile_cache_items=512,
    min_good_matches=10,
    save_dir="results",
):
    """
    Perform dynamic satellite tile loading and matching against drone video frames.
    - Loads an N x N grid of tiles centered at the current tile, stitches them,
      and matches each frame to this stitched satellite image using SIFT.
    - Updates the center tile each frame based on the estimated best match offset.
    Returns a list of (tile_x, tile_y) centers over time.
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
        print(f"Error: Could not open video file {video_path}")
        return centers

    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    frame_count = 0
    processed_frames = 0

    center_x = int(initial_tile_x)
    center_y = int(initial_tile_y)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        processed_frames += 1
        print("Processed frames :-",{processed_frames})
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

        # Compute features
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

                # Offset relative to stitched center
                stitched_h, stitched_w = stitched.shape[:2]
                dx = cx - (stitched_w * 0.5)
                dy = cy - (stitched_h * 0.5)

                # Estimate tile size from stitched dimensions and grid size (assumes square tiles)
                tile_px = int(round(stitched_h / grid_size))
                center_x, center_y = update_center_from_match((center_x, center_y), (dx, dy), tile_px)
                # track bounds to render full-extent trajectory later
                if center_x < min_x:
                    min_x = center_x
                if center_x > max_x:
                    max_x = center_x
                if center_y < min_y:
                    min_y = center_y
                if center_y > max_y:
                    max_y = center_y

                # Save match visualization with inlier lines
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
    # Build a full-extent mosaic from covered tile bounds and draw the entire trajectory
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
        # map tile centers to pixel coordinates on the full mosaic
        tile_px = int(round(full.shape[0] / (max_y - min_y + 1)))
        def tile_to_pixel(tx, ty):
            px = (tx - min_x) * tile_px + tile_px // 2
            py = (ty - min_y) * tile_px + tile_px // 2
            return (int(px), int(py))

        # draw progressive trajectory and save frames
        temp = full.copy()
        pts = [tile_to_pixel(x, y) for (x, y) in centers]
        for i in range(1, len(pts)):
            cv2.line(temp, pts[i - 1], pts[i], (255, 0, 0), 3)
            cv2.circle(temp, pts[i - 1], 5, (0, 255, 0), -1)
            frame_path = os.path.join(save_dir, f"traj_{i:05d}.png")
            cv2.imwrite(frame_path, temp)
            traj_frames.append(frame_path)
        # save final map
        cv2.circle(temp, pts[0], 8, (0, 255, 0), -1)
        cv2.putText(temp, "START", (pts[0][0] + 10, pts[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.circle(temp, pts[-1], 8, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(save_dir, "trajectory_map.png"), temp)
    return centers, traj_frames
