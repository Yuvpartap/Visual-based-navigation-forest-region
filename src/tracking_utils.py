"""
Tracking utilities including Kalman filter and hybrid rotation tracking.
Provides smooth position and rotation tracking for drone navigation.
"""

import numpy as np
import logging
from src.rotation_utils import SatelliteRotationExtractor

logger = logging.getLogger(__name__)


class KalmanFilterTracker:
    """Kalman filter for 2D position tracking with velocity estimation."""
    
    def __init__(self, process_noise=0.01, measurement_noise=1.0):
        """
        Initialize Kalman filter for 2D position tracking.
        
        Args:
            process_noise: Process noise covariance
            measurement_noise: Measurement noise covariance
        """
        # State: [x, y, vx, vy]
        self.state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # State transition matrix
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (measure position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Covariance matrices
        self.P = np.eye(4, dtype=np.float32)
        self.Q = process_noise * np.eye(4, dtype=np.float32)
        self.R = measurement_noise * np.eye(2, dtype=np.float32)
    
    def predict(self):
        """Predict next state."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, measurement):
        """
        Update with new measurement.
        
        Args:
            measurement: Measured position [x, y]
            
        Returns:
            Updated state position [x, y]
        """
        z = np.array(measurement, dtype=np.float32)
        y = z - self.H @ self.state
        
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        return self.state[:2]
    
    def set_initial_state(self, position):
        """
        Set initial position.
        
        Args:
            position: Initial position [x, y]
        """
        self.state[:2] = np.array(position, dtype=np.float32)


class HybridRotationTracker:
    """
    Hybrid tracker using satellite as ground truth + F2F for interpolation.
    
    Strategy:
    - Every frame: Extract rotation from satellite matching (most accurate)
    - Between frames: Use F2F for smooth interpolation
    - Result: Accurate rotation with smooth transitions
    """
    
    def __init__(self, use_satellite_rotation=True, f2f_smoothing=1.0,
                 min_confidence=0.3):
        """
        Initialize hybrid rotation tracker.
        
        Args:
            use_satellite_rotation: Use satellite H for rotation (recommended)
            f2f_smoothing: F2F smoothing factor (1.0 = no smoothing)
            min_confidence: Minimum confidence to accept satellite rotation
        """
        self.use_satellite_rotation = use_satellite_rotation
        self.f2f_smoothing = f2f_smoothing
        self.min_confidence = min_confidence
        
        # State tracking
        self.prev_frame = None
        self.cumulative_rotation = 0.0
        self.last_satellite_rotation = 0.0
        
        # History for analysis
        self.history = []
        self.frame_count = 0
        
        logger.info(f"Hybrid Rotation Tracker initialized:")
        logger.info(f"  Satellite rotation: {'ENABLED' if use_satellite_rotation else 'DISABLED'}")
        logger.info(f"  F2F smoothing: {f2f_smoothing}")
        logger.info(f"  Min confidence: {min_confidence}")
    
    def update_from_satellite(self, H_satellite, mask, base_rotation):
        """
        Update rotation estimate from frame-to-satellite homography.
        This is the PRIMARY and most ACCURATE rotation source.
        
        Args:
            H_satellite: Homography from frame to satellite
            mask: Inlier mask from RANSAC
            base_rotation: Initial calibrated rotation
            
        Returns:
            dict with rotation info
        """
        self.frame_count += 1
        
        # Extract rotation from satellite homography
        satellite_rotation_abs = SatelliteRotationExtractor.extract_rotation_from_homography(H_satellite, invert=True)
        confidence = SatelliteRotationExtractor.compute_rotation_confidence(H_satellite, mask)
        
        if satellite_rotation_abs is None or confidence < self.min_confidence:
            logger.debug(f"  Satellite rotation: SKIP (confidence={confidence:.2f})")
            return {
                'success': False,
                'cumulative_rotation': self.cumulative_rotation,
                'satellite_rotation': None,
                'confidence': confidence,
                'source': 'none'
            }
        
        # Convert absolute rotation to cumulative (relative to base)
        # Satellite gives us: base + cumulative = total
        # So: cumulative = total - base
        satellite_cumulative = satellite_rotation_abs - base_rotation
        satellite_cumulative = self._normalize_angle(satellite_cumulative)
        
        # Smooth transition from previous estimate
        if self.frame_count == 1:
            # First frame - use satellite directly
            self.cumulative_rotation = satellite_cumulative
            smoothed = satellite_cumulative
        else:
            # Smooth transition (prevents jumps)
            delta = satellite_cumulative - self.cumulative_rotation
            delta = self._normalize_angle(delta)
            
            # Apply smoothing based on confidence
            alpha = min(0.5 + confidence * 0.5, 0.9)  # 0.5-0.9 based on confidence
            smoothed = self.cumulative_rotation + alpha * delta
            smoothed = self._normalize_angle(smoothed)
            
            self.cumulative_rotation = smoothed
        
        self.last_satellite_rotation = satellite_cumulative
        
        # Store in history
        self.history.append({
            'frame': self.frame_count,
            'cumulative': self.cumulative_rotation,
            'satellite_absolute': satellite_rotation_abs,
            'satellite_cumulative': satellite_cumulative,
            'confidence': confidence,
            'source': 'satellite'
        })
        
        logger.debug(f"  Satellite rotation: {satellite_rotation_abs:.2f}° "
                    f"(cumulative: {satellite_cumulative:.2f}°, "
                    f"confidence: {confidence:.2f})")
        
        return {
            'success': True,
            'cumulative_rotation': self.cumulative_rotation,
            'satellite_rotation': satellite_rotation_abs,
            'confidence': confidence,
            'source': 'satellite',
            'delta_from_previous': delta if self.frame_count > 1 else 0.0
        }
    
    def update_from_f2f(self, current_frame, matcher, min_matches=15):
        """
        Fallback: Update from frame-to-frame tracking.
        Used when satellite matching fails or as supplementary info.
        
        Args:
            current_frame: Current video frame
            matcher: Matcher instance
            min_matches: Minimum matches required
            
        Returns:
            dict with rotation info
        """
        self.frame_count += 1
        
        if self.prev_frame is None:
            self.prev_frame = current_frame.copy()
            return {
                'success': False,
                'cumulative_rotation': self.cumulative_rotation,
                'f2f_rotation': None,
                'source': 'none'
            }
        
        # Match with previous frame
        src_pts, dst_pts, conf, num_matches = matcher.match(
            self.prev_frame, current_frame, min_matches=min_matches
        )
        
        if num_matches < min_matches:
            logger.debug(f"  F2F: Insufficient matches ({num_matches})")
            self.prev_frame = current_frame.copy()
            return {
                'success': False,
                'cumulative_rotation': self.cumulative_rotation,
                'f2f_rotation': None,
                'source': 'none'
            }
        
        # Compute homography
        H_f2f, mask, inliers = matcher.compute_homography(
            src_pts, dst_pts, conf, ransac_threshold=3.0
        )
        
        if H_f2f is None or inliers < min_matches:
            logger.debug(f"  F2F: Homography failed ({inliers} inliers)")
            self.prev_frame = current_frame.copy()
            return {
                'success': False,
                'cumulative_rotation': self.cumulative_rotation,
                'f2f_rotation': None,
                'source': 'none'
            }
        
        # Extract rotation
        f2f_rotation_delta = SatelliteRotationExtractor.extract_rotation_from_homography(H_f2f)
        
        if f2f_rotation_delta is not None:
            # Apply smoothing
            smoothed_delta = self.f2f_smoothing * f2f_rotation_delta
            
            # Update cumulative
            self.cumulative_rotation += smoothed_delta
            self.cumulative_rotation = self._normalize_angle(self.cumulative_rotation)
            
            # Store in history
            self.history.append({
                'frame': self.frame_count,
                'cumulative': self.cumulative_rotation,
                'f2f_delta': f2f_rotation_delta,
                'smoothed_delta': smoothed_delta,
                'source': 'f2f',
                'inliers': inliers
            })
            
            logger.debug(f"  F2F rotation: Δ={f2f_rotation_delta:.2f}°, "
                        f"cumulative={self.cumulative_rotation:.2f}°")
        
        self.prev_frame = current_frame.copy()
        
        return {
            'success': True,
            'cumulative_rotation': self.cumulative_rotation,
            'f2f_rotation': f2f_rotation_delta,
            'source': 'f2f'
        }
    
    def _normalize_angle(self, angle):
        """Normalize to [-180, 180]."""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
    
    def get_statistics(self):
        """Get tracking statistics."""
        if not self.history:
            return None
        
        satellite_frames = [h for h in self.history if h['source'] == 'satellite']
        f2f_frames = [h for h in self.history if h['source'] == 'f2f']
        
        stats = {
            'total_frames': len(self.history),
            'satellite_frames': len(satellite_frames),
            'f2f_frames': len(f2f_frames),
            'final_cumulative': self.cumulative_rotation,
            'satellite_percentage': (len(satellite_frames) / len(self.history) * 100) if self.history else 0
        }
        
        if satellite_frames:
            confidences = [h['confidence'] for h in satellite_frames]
            stats['avg_satellite_confidence'] = np.mean(confidences)
        
        return stats
    
    def reset(self):
        """Reset tracker state."""
        self.prev_frame = None
        self.cumulative_rotation = 0.0
        self.last_satellite_rotation = 0.0
        self.history.clear()
        self.frame_count = 0
