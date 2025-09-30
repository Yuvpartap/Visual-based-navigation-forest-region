"""
Compare SuperPoint+LightGlue vs DINOv2+LoFTR on a single frame pair.

This script helps you quickly evaluate which matcher works better for your data.
"""

import cv2
import numpy as np
import time
import logging
from src.superpoint_lightglue_matcher import create_matcher
from src.dino_loftr_matcher import create_dino_loftr_matcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compare_matchers(frame_path, map_path):
    """
    Compare both matchers on a single frame-map pair.
    
    Args:
        frame_path: Path to drone frame image
        map_path: Path to satellite map/tile image
    """
    # Load images
    logger.info(f"Loading images...")
    frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
    map_img = cv2.imread(map_path, cv2.IMREAD_COLOR)
    
    if frame is None or map_img is None:
        logger.error("Failed to load images!")
        return
    
    logger.info(f"Frame shape: {frame.shape}")
    logger.info(f"Map shape: {map_img.shape}")
    logger.info("=" * 60)
    
    # Test SuperPoint + LightGlue
    logger.info("Testing SuperPoint + LightGlue...")
    logger.info("-" * 60)
    
    sp_matcher = create_matcher(use_gpu=True, max_keypoints=2048, match_threshold=0.2)
    
    start_time = time.time()
    sp_src_pts, sp_dst_pts, sp_confidence, sp_num_matches = sp_matcher.match_images(
        frame, map_img, min_matches=10
    )
    sp_time = time.time() - start_time
    
    if sp_num_matches >= 10:
        sp_H, sp_mask, sp_num_inliers = sp_matcher.compute_homography_with_confidence(
            sp_src_pts, sp_dst_pts, sp_confidence, ransac_threshold=5.0
        )
        logger.info(f"✓ SuperPoint Results:")
        logger.info(f"  Matches: {sp_num_matches}")
        logger.info(f"  Inliers: {sp_num_inliers}")
        logger.info(f"  Time: {sp_time:.2f}s")
        logger.info(f"  Inlier ratio: {sp_num_inliers/sp_num_matches*100:.1f}%")
        
        # Save visualization
        kpts0 = sp_src_pts.reshape(-1, 2)
        kpts1 = sp_dst_pts.reshape(-1, 2)
        vis_sp = sp_matcher.draw_matches_visualization(
            frame, map_img, kpts0, kpts1, sp_mask, sp_confidence
        )
        cv2.imwrite("comparison_superpoint.png", vis_sp)
        logger.info(f"  Saved: comparison_superpoint.png")
    else:
        logger.warning(f"✗ SuperPoint failed: only {sp_num_matches} matches")
        sp_num_inliers = 0
    
    logger.info("=" * 60)
    
    # Test DINOv2 + LoFTR
    logger.info("Testing DINOv2 + LoFTR...")
    logger.info("-" * 60)
    
    loftr_matcher = create_dino_loftr_matcher(use_gpu=True, loftr_model="outdoor", resize_max=840)
    
    start_time = time.time()
    loftr_src_pts, loftr_dst_pts, loftr_confidence, loftr_num_matches = loftr_matcher.match_images(
        frame, map_img, min_matches=20
    )
    loftr_time = time.time() - start_time
    
    if loftr_num_matches >= 20:
        loftr_H, loftr_mask, loftr_num_inliers = loftr_matcher.compute_homography_with_confidence(
            loftr_src_pts, loftr_dst_pts, loftr_confidence, ransac_threshold=3.0
        )
        logger.info(f"✓ LoFTR Results:")
        logger.info(f"  Matches: {loftr_num_matches}")
        logger.info(f"  Inliers: {loftr_num_inliers}")
        logger.info(f"  Time: {loftr_time:.2f}s")
        logger.info(f"  Inlier ratio: {loftr_num_inliers/loftr_num_matches*100:.1f}%")
        
        # Save visualization
        kpts0 = loftr_src_pts.reshape(-1, 2)
        kpts1 = loftr_dst_pts.reshape(-1, 2)
        vis_loftr = loftr_matcher.draw_matches_visualization(
            frame, map_img, kpts0, kpts1, loftr_mask, loftr_confidence
        )
        cv2.imwrite("comparison_loftr.png", vis_loftr)
        logger.info(f"  Saved: comparison_loftr.png")
    else:
        logger.warning(f"✗ LoFTR failed: only {loftr_num_matches} matches")
        loftr_num_inliers = 0
    
    logger.info("=" * 60)
    
    # Summary
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Metric':<20} {'SuperPoint':<15} {'LoFTR':<15} {'Winner'}")
    logger.info("-" * 60)
    logger.info(f"{'Matches':<20} {sp_num_matches:<15} {loftr_num_matches:<15} {'LoFTR' if loftr_num_matches > sp_num_matches else 'SuperPoint'}")
    logger.info(f"{'Inliers':<20} {sp_num_inliers:<15} {loftr_num_inliers:<15} {'LoFTR' if loftr_num_inliers > sp_num_inliers else 'SuperPoint'}")
    logger.info(f"{'Time (s)':<20} {sp_time:<15.2f} {loftr_time:<15.2f} {'SuperPoint' if sp_time < loftr_time else 'LoFTR'}")
    
    if sp_num_matches > 0:
        sp_ratio = sp_num_inliers / sp_num_matches * 100
    else:
        sp_ratio = 0
    
    if loftr_num_matches > 0:
        loftr_ratio = loftr_num_inliers / loftr_num_matches * 100
    else:
        loftr_ratio = 0
    
    logger.info(f"{'Inlier ratio (%)':<20} {sp_ratio:<15.1f} {loftr_ratio:<15.1f} {'LoFTR' if loftr_ratio > sp_ratio else 'SuperPoint'}")
    logger.info("=" * 60)
    
    # Recommendation
    if loftr_num_inliers > sp_num_inliers * 2:
        logger.info("🎯 RECOMMENDATION: Use LoFTR (significantly more inliers)")
    elif sp_num_inliers > loftr_num_inliers * 1.5:
        logger.info("🎯 RECOMMENDATION: Use SuperPoint (more inliers, faster)")
    elif loftr_num_inliers > sp_num_inliers:
        logger.info("🎯 RECOMMENDATION: Use LoFTR (more inliers, worth the extra time)")
    else:
        logger.info("🎯 RECOMMENDATION: Use SuperPoint (faster, similar accuracy)")
    
    logger.info("=" * 60)
    logger.info("Check the saved visualizations:")
    logger.info("  - comparison_superpoint.png")
    logger.info("  - comparison_loftr.png")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python compare_matchers.py <frame_image> <map_image>")
        print("\nExample:")
        print('  python compare_matchers.py "path/to/drone_frame.png" "path/to/satellite_map.png"')
        sys.exit(1)
    
    frame_path = sys.argv[1]
    map_path = sys.argv[2]
    
    compare_matchers(frame_path, map_path)
