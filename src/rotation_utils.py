"""
Rotation calibration and extraction utilities for drone video processing.
Handles rotation detection, calibration, and frame transformation.
"""

import cv2
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)


class RotationCalibrator:
    """
    Calibrates and applies rotation transformations to video frames.
    Supports multiple rotation angles with optimized processing.
    """

    ROTATION_ANGLES = [0, 39, 45, 90, 135, 180, 225, 270, 315]
    
    _ROTATION_MATRIX_CACHE = {}
    
    @staticmethod
    def find_best_rotation(frame: np.ndarray, stitched: np.ndarray, 
                          matcher, min_matches: int = 20,
                          calibration_size: int = 640,
                          use_parallel: bool = True) -> int:
        """
        Find the best rotation angle by testing all predefined angles.
        
        Args:
            frame: Input video frame
            stitched: Stitched satellite image
            matcher: Matcher instance for feature matching
            min_matches: Minimum number of matches required
            calibration_size: Maximum dimension for calibration images
            use_parallel: Whether to use parallel processing
            
        Returns:
            Best rotation angle in degrees
        """
        logger.info("\n" + "=" * 70)
        logger.info("ROTATION CALIBRATION - Testing all angles")
        logger.info("=" * 70)
        
        # OPTIMIZATION 1: Downscale both images for faster matching
        frame_small = RotationCalibrator._resize_for_calibration(frame, max_dim=calibration_size)
        stitched_small = RotationCalibrator._resize_for_calibration(stitched, max_dim=calibration_size)
        
        # OPTIMIZATION 2: Try to cache stitched features (if matcher supports it)
        stitched_features = RotationCalibrator._extract_and_cache_features(
            matcher, stitched_small
        )
        
        best_angle = 0
        best_inliers = 0
        results = []
        
        # OPTIMIZATION 3: Test all angles (with optional parallelization)
        if use_parallel:
            results = RotationCalibrator._test_angles_parallel(
                frame_small, stitched_small, RotationCalibrator.ROTATION_ANGLES,
                matcher, min_matches, stitched_features
            )
        else:
            for angle in RotationCalibrator.ROTATION_ANGLES:
                inliers, num_matches = RotationCalibrator._test_single_angle(
                    frame_small, stitched_small, angle, matcher, min_matches, stitched_features
                )
                results.append((angle, num_matches, inliers))
                logger.info(f"  Angle {angle:3d}°: {num_matches:3d} matches, {inliers:3d} inliers")
        
        # Find best angle from all results
        for angle, num_matches, inliers in results:
            if inliers > best_inliers:
                best_inliers = inliers
                best_angle = angle
        
        logger.info("-" * 70)
        logger.info(f"✓ Best rotation: {best_angle}° with {best_inliers} inliers")
        logger.info("=" * 70 + "\n")
        
        return best_angle
    
    @staticmethod
    def _extract_and_cache_features(matcher, stitched):
        """Extract and cache features from stitched image if supported."""
        try:
            # Check if matcher has extract_features method
            if hasattr(matcher, 'matcher') and hasattr(matcher.matcher, 'extract_features'):
                logger.info("  ⚡ Caching stitched image features for reuse...")
                return matcher.matcher.extract_features(stitched)
            elif hasattr(matcher, 'extract_features'):
                logger.info("  ⚡ Caching stitched image features for reuse...")
                return matcher.extract_features(stitched)
        except Exception as e:
            logger.debug(f"  Feature caching not supported: {e}")
        
        return None
    
    @staticmethod
    def _test_single_angle(frame_small, stitched_small, angle, matcher, 
                          min_matches, stitched_features=None):
        """Test a single rotation angle and return match quality."""
        # OPTIMIZATION: Fast rotation for 90° increments (no cropping overhead)
        if angle == 0:
            rotated_frame = frame_small
        elif angle == 90:
            rotated_frame = cv2.rotate(frame_small, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated_frame = cv2.rotate(frame_small, cv2.ROTATE_180)
        elif angle == 270:
            rotated_frame = cv2.rotate(frame_small, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            # For oblique angles, use optimized rotate_and_crop
            rotated_frame = RotationCalibrator._rotate_and_crop_optimized(frame_small, angle)
        
        # Match with rotated frame (reusing cached stitched features if available)
        if stitched_features is not None:
            try:
                # Use cached features if matcher supports it
                src_pts, dst_pts, confidence, num_matches = matcher.match_with_cached_features(
                    rotated_frame, stitched_features, min_matches=min_matches
                )
            except (AttributeError, TypeError):
                # Fallback to normal matching
                src_pts, dst_pts, confidence, num_matches = matcher.match(
                    rotated_frame, stitched_small, min_matches=min_matches
                )
        else:
            src_pts, dst_pts, confidence, num_matches = matcher.match(
                rotated_frame, stitched_small, min_matches=min_matches
            )
        
        inliers = 0
        if num_matches >= min_matches:
            H, mask, inliers = matcher.compute_homography(
                src_pts, dst_pts, confidence, ransac_threshold=5.0
            )
        
        return inliers, num_matches
    
    @staticmethod
    def _test_angles_parallel(frame_small, stitched_small, angles, matcher, 
                             min_matches, stitched_features=None):
        """
        Test multiple angles in parallel.
        NOTE: Only use if the matcher is thread-safe. GPU matchers may need special handling.
        """
        results = []
        results_lock = threading.Lock()
        
        def test_angle(angle):
            try:
                inliers, num_matches = RotationCalibrator._test_single_angle(
                    frame_small, stitched_small, angle, matcher, min_matches, stitched_features
                )
                with results_lock:
                    logger.info(f"  Angle {angle:3d}°: {num_matches:3d} matches, {inliers:3d} inliers")
                return (angle, num_matches, inliers)
            except Exception as e:
                logger.warning(f"  Angle {angle}° failed: {e}")
                return (angle, 0, 0)
        
        # Use 4 workers (good balance for 8 angles on Jetson Orin)
        with ThreadPoolExecutor(max_workers=4) as executor:
            logger.info(f"Parallel processing")
            results = list(executor.map(test_angle, angles))
        
        # Sort results by angle for cleaner logging
        results.sort(key=lambda x: x[0])
        
        return results
    
    @staticmethod
    def _resize_for_calibration(frame, max_dim=640):
        """Resize frame for calibration while maintaining aspect ratio."""
        h, w = frame.shape[:2]
        scale = min(max_dim / max(h, w), 1.0)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            # Ensure dimensions are multiples of 8 for better GPU performance
            new_w = (new_w // 8) * 8
            new_h = (new_h // 8) * 8
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return frame
    
    @staticmethod
    def _get_rotation_matrix(w, h, angle):
        """Get cached rotation matrix for given dimensions and angle."""
        key = (w, h, angle)
        if key not in RotationCalibrator._ROTATION_MATRIX_CACHE:
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Precompute bounding box
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            
            # Adjust for new center
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            
            RotationCalibrator._ROTATION_MATRIX_CACHE[key] = (M, new_w, new_h)
        
        return RotationCalibrator._ROTATION_MATRIX_CACHE[key]
    
    @staticmethod
    def _rotate_and_crop_optimized(image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image and crop to remove black borders."""
        if image is None or image.size == 0:
            return image
        
        h, w = image.shape[:2]
        
        # Get cached rotation matrix
        M, new_w, new_h = RotationCalibrator._get_rotation_matrix(w, h, angle)
        
        # Perform rotation
        rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))
        
        # Fast cropping: find non-black region
        if len(rotated.shape) == 3:
            gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        else:
            gray = rotated
        
        # Threshold and find bounding rect (faster than contours)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        
        if coords is not None:
            x, y, w_crop, h_crop = cv2.boundingRect(coords)
            cropped = rotated[y:y+h_crop, x:x+w_crop]
            
            if cropped.size > 0 and cropped.shape[0] >= 10 and cropped.shape[1] >= 10:
                return cropped
        
        # Fallback
        return rotated
    
    @staticmethod
    def rotate_frame(frame: np.ndarray, angle: int) -> np.ndarray:
        """
        Rotate frame by specified angle with optimization for 90° increments.
        
        Args:
            frame: Input frame
            angle: Rotation angle in degrees
            
        Returns:
            Rotated frame
        """
        if angle == 0:
            return frame
        
        # Use fast rotation for 90° increments (no padding/cropping)
        if angle == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # For other angles, use optimized rotate_and_crop
        return RotationCalibrator._rotate_and_crop_optimized(frame, angle)
    
    @staticmethod
    def rotate_and_crop(image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate and crop image (alias for rotate_frame)."""
        if angle == 0:
            return image
        
        # Use fast cv2.rotate for 90° increments
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Use optimized version for oblique angles
        return RotationCalibrator._rotate_and_crop_optimized(image, angle)


class SatelliteRotationExtractor:
    """
    Extracts accurate rotation from frame-to-satellite matching.
    This is the GROUND TRUTH rotation - most accurate source!
    """
    
    @staticmethod
    def extract_rotation_from_homography(H, invert=False):
        """
        Extract rotation angle from homography matrix.
        
        For a homography H = [R | t], where R is 2x2 rotation+scale:
        We decompose R using SVD to separate pure rotation from scale.
        
        Args:
            H: Homography matrix
            invert: If True, return negative angle (for frame→satellite matching)
        
        Returns:
            angle in degrees (-180 to 180)
        """
        if H is None:
            return None
        
        try:
            # Extract rotation/scale component
            R = H[:2, :2]
            
            # Decompose using SVD to get pure rotation
            U, S, Vt = np.linalg.svd(R)
            R_pure = U @ Vt
            
            # Extract angle
            angle_rad = np.arctan2(R_pure[1, 0], R_pure[0, 0])
            angle_deg = np.degrees(angle_rad)
            
            # CRITICAL FIX: When matching frame→satellite, we get the inverse rotation
            # We need to negate it to get the actual drone rotation
            if invert:
                angle_deg = -angle_deg
            
            return angle_deg
            
        except Exception as e:
            logger.debug(f"Rotation extraction failed: {e}")
            return None
    
    @staticmethod
    def compute_rotation_confidence(H, mask):
        """
        Compute confidence score for rotation estimate.
        Based on homography quality.
        
        Returns:
            confidence score 0-1
        """
        if H is None or mask is None:
            return 0.0
        
        try:
            # Inlier ratio
            inlier_ratio = np.sum(mask) / len(mask) if len(mask) > 0 else 0.0
            
            # Check if rotation component is well-conditioned
            R = H[:2, :2]
            U, S, Vt = np.linalg.svd(R)
            condition = S[0] / S[1] if S[1] > 1e-6 else 1000.0
            
            # Good condition number (close to 1) = pure rotation
            # Bad condition number (far from 1) = includes scale/shear
            condition_score = 1.0 / (1.0 + abs(condition - 1.0))
            
            # Combined confidence
            confidence = 0.7 * inlier_ratio + 0.3 * condition_score
            
            return confidence
            
        except Exception as e:
            logger.debug(f"Confidence computation failed: {e}")
            return 0.0
