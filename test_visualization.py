"""
Test script to verify match visualization is working correctly.
"""
import cv2
import numpy as np
from src.superpoint_lightglue_matcher import create_matcher

# Create test images
print("Creating test images...")
img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
img2 = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)

print(f"Image 1 shape: {img1.shape}")
print(f"Image 2 shape: {img2.shape}")

# Create matcher
print("\nInitializing matcher...")
matcher = create_matcher(use_gpu=True, max_keypoints=100, match_threshold=0.2)

# Match images
print("\nMatching images...")
src_pts, dst_pts, confidence, num_matches = matcher.match_images(
    img1, img2, min_matches=5
)

print(f"Found {num_matches} matches")

if num_matches >= 5:
    # Compute homography
    H, mask, num_inliers = matcher.compute_homography_with_confidence(
        src_pts, dst_pts, confidence
    )
    
    print(f"Inliers: {num_inliers}")
    
    if num_inliers >= 5:
        # Create visualization
        print("\nCreating visualization...")
        kpts0 = src_pts.reshape(-1, 2)
        kpts1 = dst_pts.reshape(-1, 2)
        
        vis = matcher.draw_matches_visualization(
            img1, img2, kpts0, kpts1, mask, confidence
        )
        
        print(f"Visualization shape: {vis.shape}")
        print(f"Expected width: {img1.shape[1] + img2.shape[1]} = {img1.shape[1]} + {img2.shape[1]}")
        print(f"Actual width: {vis.shape[1]}")
        
        # Save visualization
        output_path = "test_match_visualization.png"
        success = cv2.imwrite(output_path, vis)
        
        if success:
            print(f"\n✓ Successfully saved visualization to: {output_path}")
            print(f"  File should show two images side-by-side with green lines connecting matches")
        else:
            print(f"\n✗ Failed to save visualization")
    else:
        print("Not enough inliers for visualization")
else:
    print("Not enough matches found")
