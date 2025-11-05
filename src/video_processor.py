"""
Video processor for drone navigation using satellite tile matching.
Handles frame processing, rotation tracking, and position estimation.
"""

import os
import cv2
import numpy as np
import time
import logging
from src.cuda_manager import CUDAManager
from src.matcher_wrapper import JetsonOptimizedLoFTRMatcher, AdaptiveResizer
from src.tracking_utils import KalmanFilterTracker, HybridRotationTracker
from src.rotation_utils import RotationCalibrator, SatelliteRotationExtractor
from src.trajectory_visualizer import TrajectoryVisualizer
from src.tile_loading_utilis import TileCache, load_and_stitch_grid, update_center_from_match
from src.refactored_dino_loftr import create_dino_loftr_matcher

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Processes video frames for tile matching with hybrid tracking."""
    
    def __init__(self, config):
        """
        Initialize video processor.
        
        Args:
            config: Configuration dictionary with processing parameters
        """
        self.config = config
        self.centers = []
        self.stats = {
            'frame_count': 0,
            'processed_frames': 0,
            'matched_frames': 0,
            'match_index': 0
        }
        self.current_center = (config['initial_tile_x'], config['initial_tile_y'])
        self.rotation_angle = None  # Will be calibrated on first frame
        
        # Hybrid tracking components
        self.kalman = KalmanFilterTracker(process_noise=0.01, measurement_noise=1.0)
        self.rotation_tracker = HybridRotationTracker(
            use_satellite_rotation=True,  # Use satellite H for rotation
            f2f_smoothing=1.0,             # No artificial dampening
            min_confidence=0.3             # Minimum confidence threshold
        )
        self.kalman.set_initial_state(self.current_center)
    
    def process(self):
        """
        Main processing loop.
        
        Returns:
            Tuple of (centers, empty_list) for compatibility
        """
        self._log_configuration()
        
        cuda_ok = CUDAManager.verify_and_setup()
        matcher = self._create_matcher(cuda_ok)
        cache = TileCache(max_items=self.config['tile_cache_items'])
        
        cap = cv2.VideoCapture(self.config['video_path'])
        if not cap.isOpened():
            logger.error(f"Could not open video file {self.config['video_path']}")
            return self.centers, []
        
        CUDAManager.clear_memory()
        
        while True:
            frame_start = time.time()
            
            if not self._process_frame(cap, matcher, cache, frame_start):
                break
        
        cap.release()
        CUDAManager.clear_memory()
        
        self._log_summary(matcher)
        TrajectoryVisualizer.create_trajectory_map(
            self.centers, self.config['zoom'], self.config['tiles_dir'],
            self.config['ext'], cache, self.config['save_dir']
        )
        
        # Generate trajectory video with progressive line drawing
        TrajectoryVisualizer.create_trajectory_video(
            self.centers, self.config['zoom'], self.config['tiles_dir'],
            self.config['ext'], cache, self.config['save_dir'], fps=30, trail_len=None
        )
        
        if hasattr(cache, 'cache'):
            cache.cache.clear()
        
        return self.centers, []
    
    def _create_matcher(self, cuda_ok):
        """Create and configure the matcher."""
        resize_max = AdaptiveResizer.calculate_resize_max(
            self.config['grid_size'], target_scale_ratio=2.0
        ) if self.config['adaptive_resize'] else 840
        
        base_matcher = create_dino_loftr_matcher(
            use_gpu=cuda_ok, loftr_model="outdoor", resize_max=resize_max
        )
        return JetsonOptimizedLoFTRMatcher(base_matcher, cuda_ok)
    
    def _process_frame(self, cap, matcher, cache, frame_start):
        """Process frame with iterative rotation refinement."""
        ret, frame = cap.read()
        if not ret:
            return False
        
        if self.stats['frame_count'] % self.config['frame_skip'] != 0:
            self.stats['frame_count'] += 1
            return True
        
        self.stats['processed_frames'] += 1
        
        # Step 1: Calibrate base rotation on first frame
        if self.rotation_angle is None:
            stitched = load_and_stitch_grid(
                self.config['zoom'], tiles_dir=self.config['tiles_dir'],
                center_x=self.current_center[0], center_y=self.current_center[1],
                grid_size=self.config['grid_size'], ext=self.config['ext'], cache=cache
            )
            
            if stitched is None:
                logger.error("Failed to load tiles for rotation calibration")
                self.stats['frame_count'] += 1
                return True
            
            self.rotation_angle = RotationCalibrator.find_best_rotation(
                frame, stitched, matcher, min_matches=self.config['min_good_matches']
            )
            logger.info(f"✓ Base rotation calibrated: {self.rotation_angle}°")
            self.rotation_tracker.cumulative_rotation = 0.0
        
        # Step 2: Apply CURRENT best rotation estimate BEFORE matching
        current_best_rotation = self.rotation_angle + self.rotation_tracker.cumulative_rotation
        current_best_rotation_int = int(round(current_best_rotation))
        rotated_frame = RotationCalibrator.rotate_frame(frame, current_best_rotation_int)
        
        # Step 3: Load satellite tiles
        tile_start = time.time()
        stitched = load_and_stitch_grid(
            self.config['zoom'], tiles_dir=self.config['tiles_dir'],
            center_x=self.current_center[0], center_y=self.current_center[1],
            grid_size=self.config['grid_size'], ext=self.config['ext'], cache=cache
        )
        tile_time = time.time() - tile_start
        
        if stitched is None:
            logger.warning(f"  ✗ Failed to load tiles")
            self.stats['frame_count'] += 1
            return True
        
        # Step 4: Match ROTATED frame to satellite
        src_pts, dst_pts, confidence, num_matches = matcher.match(
            rotated_frame, stitched, min_matches=self.config['min_good_matches']
        )
        
        if num_matches < self.config['min_good_matches']:
            logger.warning(f"  ✗ Insufficient matches: {num_matches}")
            self.stats['frame_count'] += 1
            return True
        
        # Step 5: Compute homography
        H, mask, num_inliers = matcher.compute_homography(
            src_pts, dst_pts, confidence, self.config['ransac_threshold']
        )
        
        if H is None or num_inliers < self.config['min_good_matches']:
            logger.warning(f"  ✗ Homography failed: {num_inliers} inliers")
            self.stats['frame_count'] += 1
            return True
        
        # Step 6: Extract RESIDUAL rotation from homography
        # Since we already applied rotation, H should be mostly translation
        # Any remaining rotation is the error we need to correct
        residual_rotation = SatelliteRotationExtractor.extract_rotation_from_homography(
            H, invert=True
        )
        
        if residual_rotation is not None and abs(residual_rotation) < 45.0:
            # Update cumulative rotation with residual
            self.rotation_tracker.cumulative_rotation += residual_rotation
            self.rotation_tracker.cumulative_rotation = self.rotation_tracker._normalize_angle(
                self.rotation_tracker.cumulative_rotation
            )
        
        # Step 7: Final rotation
        total_rotation = self.rotation_angle + self.rotation_tracker.cumulative_rotation
        total_rotation_int = int(round(total_rotation))
        
        # Step 8: Logging
        logger.info(f"\n{'='*50}")
        logger.info(f"Frame {self.stats['processed_frames']} "
                f"(size: {frame.shape[1]}x{frame.shape[0]})")
        logger.info(f"  Rotation: base={self.rotation_angle}°, "
                f"cumulative={self.rotation_tracker.cumulative_rotation:.2f}°, "
                f"total={total_rotation_int}°")
        
        if residual_rotation is not None:
            logger.info(f"  Residual rotation: {residual_rotation:.2f}°")
        
        # Step 9: Update position (using rotated frame)
        self._update_position(rotated_frame, H, stitched)
        
        # Step 10: Log performance
        frame_time = time.time() - frame_start
        matcher.tracker.record_frame_time(frame_time)
        
        logger.info(f"  ✓ Matches: {num_matches}, Inliers: {num_inliers}")
        logger.info(f"  → Position: {self.current_center}")
        logger.info(f"  ⚡ Timing: tile={tile_time:.2f}s, total={frame_time:.2f}s")
        
        # Step 11: Save visualization
        self._save_visualization(rotated_frame, stitched, src_pts, dst_pts, 
                                mask, confidence, matcher)
        
        self.stats['frame_count'] += 1
        
        if self.stats['processed_frames'] % self.config['cleanup_interval'] == 0:
            matcher.clear_cache()
        
        return True

    def _match_and_update(self, frame, stitched, matcher, tile_time, frame_start):
        """Perform matching and update tracking."""
        src_pts, dst_pts, confidence, num_matches = matcher.match(
            frame, stitched, min_matches=self.config['min_good_matches']
        )
        
        if num_matches < self.config['min_good_matches']:
            logger.warning(f"  ✗ Insufficient matches: {num_matches}")
            return False
        
        H, mask, num_inliers = matcher.compute_homography(
            src_pts, dst_pts, confidence, self.config['ransac_threshold']
        )
        
        if H is None or num_inliers < self.config['min_good_matches']:
            logger.warning(f"  ✗ Homography failed: {num_inliers} inliers")
            return False
        
        self._update_position(frame, H, stitched)
        self._log_frame_performance(num_matches, num_inliers, tile_time, 
                                    matcher, frame_start)
        self._save_visualization(frame, stitched, src_pts, dst_pts, 
                                mask, confidence, matcher)
        
        return True
    
    def _update_position(self, frame, H, stitched):
        """Update current position based on homography."""
        # Predict next state (before measurement)
        self.kalman.predict()
        
        h, w = frame.shape[:2]
        corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(corners, H)
        cx = float(sum(pt[0][0] for pt in transformed) / 4.0)
        cy = float(sum(pt[0][1] for pt in transformed) / 4.0)
        
        stitched_h, stitched_w = stitched.shape[:2]
        dx = cx - (stitched_w * 0.5)
        dy = cy - (stitched_h * 0.5)
        tile_px = int(round(stitched_h / self.config['grid_size']))
        
        new_center = update_center_from_match(self.current_center, (dx, dy), tile_px)
        smoothed_center = self.kalman.update(new_center)

        # self.current_center = tuple(smoothed_center)
        self.current_center = (int(round(new_center[0])), int(round(new_center[1])))
        self.centers.append(self.current_center)
        self.stats['matched_frames'] += 1
    
    def _log_frame_performance(self, num_matches, num_inliers, tile_time, 
                              matcher, frame_start):
        """Log performance metrics for the current frame."""
        frame_time = time.time() - frame_start
        matcher.tracker.record_frame_time(frame_time)
        avg_time = np.mean(matcher.tracker.frame_times)
        
        logger.info(f"  ✓ Matches: {num_matches}, Inliers: {num_inliers}")
        logger.info(f"  → Position: {self.current_center}")
        logger.info(f"  ⚡ Timing: tile={tile_time:.2f}s, "
                   f"match={matcher.tracker.get_average('matching'):.2f}s, "
                   f"homo={matcher.tracker.get_average('homography'):.2f}s")
        logger.info(f"  ⏱️  Total: {frame_time:.2f}s (avg: {avg_time:.2f}s, "
                   f"{1.0/avg_time:.2f} FPS)")
    
    def _save_visualization(self, frame, stitched, src_pts, dst_pts, 
                           mask, confidence, matcher):
        """Save match visualization if needed."""
        if self.stats['matched_frames'] % self.config['save_every_nth'] == 0:
            kpts0 = src_pts.reshape(-1, 2)
            kpts1 = dst_pts.reshape(-1, 2)
            vis = matcher.matcher.draw_matches_visualization(
                frame, stitched, kpts0, kpts1, mask, confidence
            )
            match_name = os.path.join(
                self.config['save_dir'], 
                f"match_{self.stats['match_index']:04d}.png"
            )
            cv2.imwrite(match_name, vis)
            logger.info(f"  💾 Saved: match_{self.stats['match_index']:04d}.png")
            self.stats['match_index'] += 1
    
    def _log_configuration(self):
        """Log processing configuration."""
        logger.info("=" * 70)
        logger.info("JETSON OPTIMIZED LoFTR - ROTATION ROBUST")
        logger.info("=" * 70)
        logger.info(f"Grid size: {self.config['grid_size']}×{self.config['grid_size']}")
        logger.info(f"Resize_max: Adaptive (ratio=2.0)")
        logger.info(f"Frame preprocessing: No downsampling")
        logger.info(f"Rotation: Auto-calibrated on first frame (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)")
        logger.info(f"          - 90° increments: No cropping (pure rotation)")
        logger.info(f"          - Other angles: Smart crop (removes black padding)")
        logger.info(f"RANSAC threshold: {self.config['ransac_threshold']}")
        logger.info(f"Tile cache: {self.config['tile_cache_items']} items")
        logger.info(f"Feature cache: 50 items")
        logger.info(f"Min good matches: {self.config['min_good_matches']}")
        logger.info(f"Memory cleanup: every {self.config['cleanup_interval']} frames")
        logger.info(f"Save visualizations: every {self.config['save_every_nth']} matched frames")
        logger.info("=" * 70)
     
    def _log_summary(self, matcher):
        """Log final performance summary."""
        timing = matcher.tracker.get_breakdown()
        logger.info("\n" + "=" * 70)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Base calibrated rotation: {self.rotation_angle}°")
        
        # Rotation tracking statistics
        rot_stats = self.rotation_tracker.get_statistics()
        if rot_stats:
            logger.info(f"\n--- Rotation Tracking Statistics ---")
            logger.info(f"Final cumulative rotation: {rot_stats['final_cumulative']:.2f}°")
            logger.info(f"Total rotation: {self.rotation_angle + rot_stats['final_cumulative']:.2f}°")
            logger.info(f"Frames tracked: {rot_stats['total_frames']}")
            logger.info(f"  - Satellite-based: {rot_stats['satellite_frames']} "
                    f"({rot_stats['satellite_percentage']:.1f}%)")
            logger.info(f"  - F2F-based: {rot_stats['f2f_frames']}")
            if 'avg_satellite_confidence' in rot_stats:
                logger.info(f"Avg satellite confidence: {rot_stats['avg_satellite_confidence']:.2f}")
        
        logger.info(f"\nTotal frames processed: {self.stats['processed_frames']}")
        logger.info(f"Successful matches: {self.stats['matched_frames']} "
                f"({self.stats['matched_frames']/self.stats['processed_frames']*100:.1f}%)")
