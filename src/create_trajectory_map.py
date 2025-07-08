import os
import re
import cv2
import numpy as np

#numerical sort (1, 2, 3) instead of (1, 10, 2)
def numerical_sort(value):
    return int(re.sub(r'\D', '', value))

#main function to generate the trajectory map
def generate_trajectory_map(map_path, frames_dir, output_path):
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

    #draw trajectory
    trajectory_img = map_img.copy()
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

    #save the resulting image
    cv2.imwrite(output_path, trajectory_img)
    return center_coords