"""
Trajectory visualization utilities for creating maps and videos.
Handles trajectory smoothing, drawing, and video generation.
"""

import os
import cv2
import numpy as np
import logging
from scipy.interpolate import splprep, splev
from src.tile_loading_utilis import load_and_stitch_rect

logger = logging.getLogger(__name__)


class TrajectoryVisualizer:
    """Handles trajectory map creation and visualization."""
    
    @staticmethod
    def create_trajectory_map(centers, zoom, tiles_dir, ext, cache, save_dir):
        """
        Build and save trajectory map.
        
        Args:
            centers: List of (tile_x, tile_y) trajectory points
            zoom: Zoom level for tiles
            tiles_dir: Directory containing tile images
            ext: File extension for tiles
            cache: TileCache instance
            save_dir: Directory to save output
        """
        if not centers:
            logger.warning("✗ No trajectory points to visualize")
            return
        
        min_x = min(x for x, y in centers)
        max_x = max(x for x, y in centers)
        min_y = min(y for x, y in centers)
        max_y = max(y for x, y in centers)
        
        logger.info(f"\nBuilding trajectory map: tiles ({min_x},{min_y}) to ({max_x},{max_y})")
        
        try:
            full = load_and_stitch_rect(
                zoom, tiles_dir=tiles_dir, min_x=min_x, max_x=max_x,
                min_y=min_y, max_y=max_y, ext=ext, cache=cache
            )
            
            if full is None:
                logger.warning("✗ Failed to create trajectory map: Could not stitch tiles")
                return
            
            TrajectoryVisualizer._draw_and_save(full, centers, min_x, min_y, max_x, max_y, save_dir)
            
        except Exception as e:
            logger.error(f"✗ Error creating trajectory map: {e}")
            logger.info("  Trajectory data saved but visualization failed")
    
    @staticmethod
    def _smooth_trajectory(pts, window_size=5):
        """
        Smooth trajectory using Gaussian filtering and dense interpolation.
        Creates a smooth, realistic drone trajectory without zig-zag patterns.
        
        Args:
            pts: List of (x, y) pixel coordinates
            window_size: Size of Gaussian smoothing window (higher = smoother)
        
        Returns:
            List of smoothed (x, y) coordinates
        """
        if len(pts) < 3:
            return pts
        
        try:
            pts_array = np.array(pts, dtype=np.float32)
            
            # Step 1: Apply Gaussian smoothing to reduce noise
            from scipy.ndimage import gaussian_filter1d
            sigma = window_size / 2.0
            
            # Smooth x and y coordinates separately
            x_smooth = gaussian_filter1d(pts_array[:, 0], sigma=sigma, mode='nearest')
            y_smooth = gaussian_filter1d(pts_array[:, 1], sigma=sigma, mode='nearest')
            
            smoothed_pts = np.column_stack([x_smooth, y_smooth])
            
            # Step 2: Use spline interpolation with very high density for smooth curves
            if len(smoothed_pts) >= 4:
                # Use cubic spline with minimal smoothing (s close to 0)
                tck, u = splprep([smoothed_pts[:, 0], smoothed_pts[:, 1]], 
                                s=len(smoothed_pts) * 0.01,  # Very low smoothing
                                k=min(3, len(smoothed_pts)-1))
                
                # Generate very dense interpolation (50x more points)
                u_smooth = np.linspace(0, 1, max(len(pts) * 50, 500))
                smooth_curve = splev(u_smooth, tck)
                
                # Convert to list of tuples
                final_pts = list(zip(smooth_curve[0], smooth_curve[1]))
                return final_pts
            else:
                # Not enough points for spline, return Gaussian smoothed version
                return list(zip(smoothed_pts[:, 0], smoothed_pts[:, 1]))
        
        except Exception as e:
            logger.warning(f"Trajectory smoothing failed: {e}. Using original trajectory.")
            return pts
    
    @staticmethod
    def _draw_and_save(full, centers, min_x, min_y, max_x, max_y, save_dir):
        """Draw smooth trajectory on map and save."""
        tile_px = int(round(full.shape[0] / (max_y - min_y + 1)))
        
        def tile_to_pixel(tx, ty):
            px = (tx - min_x) * tile_px + tile_px // 2
            py = (ty - min_y) * tile_px + tile_px // 2
            return (int(px), int(py))
        
        temp = full.copy()
        pts = [tile_to_pixel(x, y) for x, y in centers]
        
        # Smooth the trajectory for realistic curved paths
        smooth_pts = TrajectoryVisualizer._smooth_trajectory(pts, window_size=5)
        
        # Draw smooth trajectory path with thicker line
        # Ensure we draw from the first smooth point to the last
        for i in range(1, len(smooth_pts)):
            cv2.line(temp, 
                    (int(smooth_pts[i - 1][0]), int(smooth_pts[i - 1][1])),
                    (int(smooth_pts[i][0]), int(smooth_pts[i][1])),
                    (255, 0, 0), 30)  # Increased thickness from 3 to 6
        
        # Draw connecting lines from start/end points to the smooth curve
        # This ensures the curve connects to the markers
        if len(smooth_pts) > 0:
            # Connect start point to first smooth point
            cv2.line(temp, pts[0], (int(smooth_pts[0][0]), int(smooth_pts[0][1])), (255, 0, 0), 35)
            # Connect last smooth point to end point
            cv2.line(temp, (int(smooth_pts[-1][0]), int(smooth_pts[-1][1])), pts[-1], (255, 0, 0), 35)
        
        # Mark original trajectory points
        for pt in pts:
            cv2.circle(temp, pt, 7, (0, 255, 0), -1)
        
        # Mark start and end with larger circles
        cv2.circle(temp, pts[0], 15, (0, 255, 0), -1)
        cv2.putText(temp, "START", (pts[0][0] + 15, pts[0][1] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 5)
        cv2.circle(temp, pts[-1], 12, (0, 0, 255), -1)
        cv2.putText(temp, "END", (pts[-1][0] + 15, pts[-1][1] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 5)
        
        traj_path = os.path.join(save_dir, "trajectory_map.png")
        cv2.imwrite(traj_path, temp)
        logger.info(f"✓ Trajectory map saved: {traj_path}")
        logger.info(f"  Total trajectory points: {len(centers)}")
        logger.info(f"  Smooth trajectory curve generated with {len(smooth_pts)} interpolated points")
    
    @staticmethod
    def create_trajectory_video(centers, zoom, tiles_dir, ext, cache, save_dir, 
                               fps=30, trail_len=None):
        """
        Generate a video that progressively draws the trajectory line by line.
        
        Args:
            centers: List of (tile_x, tile_y) trajectory points
            zoom: Zoom level for tiles
            tiles_dir: Directory containing tile images
            ext: File extension for tiles
            cache: TileCache instance
            save_dir: Directory to save video
            fps: Frames per second for output video (default 30)
            trail_len: Optional limit on how many recent segments to show (None = full history)
        """
        if not centers or len(centers) < 2:
            logger.warning("✗ Insufficient trajectory points for video generation")
            return
        
        min_x = min(x for x, y in centers)
        max_x = max(x for x, y in centers)
        min_y = min(y for x, y in centers)
        max_y = max(y for x, y in centers)
        
        logger.info(f"\nGenerating trajectory video: tiles ({min_x},{min_y}) to ({max_x},{max_y})")
        
        try:
            # Load and stitch the background map once
            full = load_and_stitch_rect(
                zoom, tiles_dir=tiles_dir, min_x=min_x, max_x=max_x,
                min_y=min_y, max_y=max_y, ext=ext, cache=cache
            )
            
            if full is None:
                logger.warning("✗ Failed to create trajectory video: Could not stitch tiles")
                return
            
            TrajectoryVisualizer._render_progressive_frames(
                full, centers, min_x, min_y, max_x, max_y, save_dir, fps, trail_len
            )
            
        except Exception as e:
            logger.error(f"✗ Error creating trajectory video: {e}")
            import traceback
            traceback.print_exc()
    
    @staticmethod
    def _render_progressive_frames(full, centers, min_x, min_y, max_x, max_y, 
                                save_dir, fps, trail_len):
        """
        Render frames with progressively drawn trajectory and encode to MP4.
        OPTIMIZED VERSION: Reduces video size via resolution scaling and better codec.
        
        Args:
            full: Stitched background map image
            centers: List of (tile_x, tile_y) trajectory points
            min_x, min_y, max_x, max_y: Tile bounds
            save_dir: Directory to save video
            fps: Frames per second
            trail_len: Optional limit on recent segments to show
        """
        tile_px = int(round(full.shape[0] / (max_y - min_y + 1)))
        h, w = full.shape[:2]
        
        # KEY OPTIMIZATION 1: Scale down resolution (reduces file size by ~75%)
        # Target: 1920x1080 (Full HD) or maintain aspect ratio
        MAX_DIMENSION = 1280  # You can reduce to 1280 for smaller files
        scale_factor = min(MAX_DIMENSION / w, MAX_DIMENSION / h, 1.0)
        
        if scale_factor < 1.0:
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            # Resize background once
            full = cv2.resize(full, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info(f"  Scaled resolution: {w}x{h} -> {new_w}x{new_h} ({scale_factor:.2%})")
            w, h = new_w, new_h
            # Scale tile_px proportionally
            tile_px = int(tile_px * scale_factor)
        
        def tile_to_pixel(tx, ty):
            px = (tx - min_x) * tile_px + tile_px // 2
            py = (ty - min_y) * tile_px + tile_px // 2
            return (int(px), int(py))
        
        # Convert tile centers to pixel coordinates
        pts = [tile_to_pixel(x, y) for x, y in centers]
        
        # Smooth the trajectory once
        smooth_pts = TrajectoryVisualizer._smooth_trajectory(pts, window_size=5)
        
        # KEY OPTIMIZATION 2: Use H.264 codec with quality settings
        video_path = os.path.join(save_dir, "trajectory_video.mp4")
        
        # Try H.264 codec (best compression, widely supported)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
        out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        
        # Fallback to x264 if avc1 not available
        if not out.isOpened():
            logger.warning("  avc1 codec failed, trying x264...")
            fourcc = cv2.VideoWriter_fourcc(*'x264')
            out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        
        # Final fallback to mp4v
        if not out.isOpened():
            logger.warning("  x264 codec failed, using mp4v...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        
        if not out.isOpened():
            logger.error(f"✗ Failed to open video writer for {video_path}")
            return
        
        logger.info(f"  Rendering {len(smooth_pts)} frames at {w}x{h}, {fps} FPS...")
        
        # KEY OPTIMIZATION 3: Scale line thicknesses proportionally
        line_thickness_main = max(2, int(15 * scale_factor))
        line_thickness_faint = max(1, int(2 * scale_factor))
        circle_radius_head = max(3, int(8 * scale_factor))
        circle_radius_marker = max(5, int(12 * scale_factor))
        font_scale = 0.8 * scale_factor
        font_thickness = max(1, int(2 * scale_factor))
        
        # Render frames with progressive trajectory drawing
        for frame_idx in range(len(smooth_pts)):
            # Start with background
            frame = full.copy()
            
            # Draw faint full trajectory as ghost line for reference
            for i in range(1, len(smooth_pts)):
                cv2.line(frame,
                        (int(smooth_pts[i - 1][0]), int(smooth_pts[i - 1][1])),
                        (int(smooth_pts[i][0]), int(smooth_pts[i][1])),
                        (100, 100, 50), line_thickness_faint)
            
            # Determine trail range to draw
            if trail_len is None:
                start_idx = 0
            else:
                start_idx = max(0, frame_idx - trail_len)
            
            # Draw the progressive trajectory up to current frame
            if frame_idx > 0:
                for i in range(start_idx + 1, frame_idx + 1):
                    cv2.line(frame,
                            (int(smooth_pts[i - 1][0]), int(smooth_pts[i - 1][1])),
                            (int(smooth_pts[i][0]), int(smooth_pts[i][1])),
                            (255, 0, 0), line_thickness_main)
            
            # Draw moving head marker at current position
            if frame_idx < len(smooth_pts):
                head_pos = (int(smooth_pts[frame_idx][0]), int(smooth_pts[frame_idx][1]))
                cv2.circle(frame, head_pos, circle_radius_head, (0, 255, 255), -1)
                cv2.circle(frame, head_pos, circle_radius_head + 2, (0, 255, 255), 2)
            
            # Draw start marker
            cv2.circle(frame, pts[0], circle_radius_marker, (0, 255, 0), -1)
            cv2.putText(frame, "START", (pts[0][0] + 15, pts[0][1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
            
            # Draw end marker
            cv2.circle(frame, pts[-1], circle_radius_marker, (0, 0, 255), -1)
            cv2.putText(frame, "END", (pts[-1][0] + 15, pts[-1][1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)
            
            # Add progress indicator
            progress_pct = int((frame_idx / len(smooth_pts)) * 100)
            cv2.putText(frame, f"Progress: {progress_pct}%", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
            
            out.write(frame)
            
            # Log progress every 10% or at key points
            if frame_idx % max(1, len(smooth_pts) // 10) == 0 or frame_idx == len(smooth_pts) - 1:
                logger.info(f"    Frame {frame_idx + 1}/{len(smooth_pts)} ({progress_pct}%)")
        
        out.release()
        
        # Get actual file size
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        
        logger.info(f"✓ Trajectory video saved: {video_path}")
        logger.info(f"  Resolution: {w}x{h}")
        logger.info(f"  FPS: {fps}")
        logger.info(f"  Total frames: {len(smooth_pts)}")
        logger.info(f"  Duration: {len(smooth_pts) / fps:.2f}s")
        logger.info(f"  File size: {file_size_mb:.1f} MB")
        
        # KEY OPTIMIZATION 4: Optional - Re-encode with ffmpeg for better compression
        # This requires ffmpeg to be installed on the system
        try:
            import subprocess
            compressed_path = os.path.join(save_dir, "trajectory_video_compressed.mp4")
            
            # Use ffmpeg for much better compression with H.264
            cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-c:v', 'libx264',  # H.264 codec
                '-preset', 'medium',  # Encoding speed (slow = better compression)
                '-crf', '23',  # Quality (18-28, lower = better quality, 23 is good)
                '-movflags', '+faststart',  # Web optimization
                compressed_path
            ]
            
            logger.info("  Re-encoding with ffmpeg for better compression...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and os.path.exists(compressed_path):
                compressed_size_mb = os.path.getsize(compressed_path) / (1024 * 1024)
                logger.info(f"  ✓ Compressed: {compressed_size_mb:.1f} MB (saved {file_size_mb - compressed_size_mb:.1f} MB)")
                
                # Replace original with compressed version
                os.remove(video_path)
                os.rename(compressed_path, video_path)
            else:
                logger.warning("  ffmpeg compression failed, keeping original")
        
        except (ImportError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.info(f"  ffmpeg not available, skipping re-encoding: {e}")
