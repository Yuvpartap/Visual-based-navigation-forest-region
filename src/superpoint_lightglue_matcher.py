"""
SuperPoint + LightGlue feature matching module for high-accuracy trajectory tracking.

This module provides a modern deep learning-based alternative to classical SIFT matching,
offering significantly improved accuracy, robustness, and GPU acceleration.
"""

import torch
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SuperPointLightGlueMatcher:
    """
    High-accuracy feature matcher using SuperPoint keypoint detector and LightGlue matcher.
    
    SuperPoint: Self-supervised interest point detector and descriptor
    LightGlue: Learned feature matcher with adaptive pruning
    
    Advantages over SIFT:
    - Higher repeatability (~80-90% vs ~60%)
    - Better robustness to lighting, scale, and rotation changes
    - Learned matching reduces false positives
    - GPU acceleration for faster processing
    - No manual threshold tuning required
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        max_num_keypoints: int = 2048,
        detection_threshold: float = 0.005,
        nms_radius: int = 4,
        match_threshold: float = 0.2,
        resize_max: Optional[int] = 1024,
        use_cache: bool = True,
    ):
        """
        Initialize SuperPoint + LightGlue matcher.
        
        Args:
            device: 'cuda', 'cpu', or None (auto-detect)
            max_num_keypoints: Maximum keypoints to extract per image
            detection_threshold: SuperPoint detection confidence threshold (lower = more keypoints)
            nms_radius: Non-maximum suppression radius for keypoints
            match_threshold: LightGlue matching confidence threshold (lower = more matches)
            resize_max: Resize images to this max dimension for efficiency (None = no resize)
            use_cache: Cache features for repeated matching (useful for map images)
        """
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initializing SuperPoint + LightGlue on device: {self.device}")
        
        self.max_num_keypoints = max_num_keypoints
        self.detection_threshold = detection_threshold
        self.nms_radius = nms_radius
        self.match_threshold = match_threshold
        self.resize_max = resize_max
        self.use_cache = use_cache
        
        # Initialize models
        try:
            from lightglue import SuperPoint, LightGlue
            
            # SuperPoint configuration
            self.extractor = SuperPoint(
                max_num_keypoints=max_num_keypoints,
                detection_threshold=detection_threshold,
                nms_radius=nms_radius,
            ).eval().to(self.device)
            
            # LightGlue configuration
            self.matcher = LightGlue(
                features='superpoint',
                depth_confidence=-1,  # Disable depth
                width_confidence=-1,  # Disable width
                filter_threshold=match_threshold,
            ).eval().to(self.device)
            
            logger.info("SuperPoint + LightGlue models loaded successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import LightGlue: {e}")
            logger.error("Please install: pip install git+https://github.com/cvg/LightGlue.git")
            raise
        
        # Feature cache for map images
        self.feature_cache: Dict[str, Dict] = {}
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess image for SuperPoint.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Preprocessed tensor and original shape (H, W)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        original_shape = gray.shape[:2]
        
        # Resize if needed for efficiency
        if self.resize_max is not None:
            h, w = gray.shape[:2]
            max_dim = max(h, w)
            if max_dim > self.resize_max:
                scale = self.resize_max / max_dim
                new_h, new_w = int(h * scale), int(w * scale)
                gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Convert to tensor and normalize
        tensor = torch.from_numpy(gray).float()[None, None] / 255.0
        tensor = tensor.to(self.device)
        
        return tensor, original_shape
    
    def extract_features(self, image: np.ndarray, cache_key: Optional[str] = None) -> Dict:
        """
        Extract SuperPoint features from an image.
        
        Args:
            image: Input image (BGR or grayscale)
            cache_key: Optional key for caching features (e.g., 'global_map')
            
        Returns:
            Dictionary with keypoints, descriptors, and scores
        """
        # Check cache
        if cache_key is not None and self.use_cache and cache_key in self.feature_cache:
            logger.debug(f"Using cached features for {cache_key}")
            return self.feature_cache[cache_key]
        
        # Preprocess
        tensor, original_shape = self.preprocess_image(image)
        
        # Extract features
        with torch.no_grad():
            features = self.extractor({'image': tensor})
        
        # Scale keypoints back to original resolution if resized
        if self.resize_max is not None:
            h, w = original_shape
            h_resized, w_resized = tensor.shape[2:]
            scale_x = w / w_resized
            scale_y = h / h_resized
            features['keypoints'][0][:, 0] *= scale_x
            features['keypoints'][0][:, 1] *= scale_y
        
        # Store image shape
        features['image_size'] = torch.tensor(original_shape).to(self.device)
        
        # Cache if requested
        if cache_key is not None and self.use_cache:
            self.feature_cache[cache_key] = features
            logger.debug(f"Cached features for {cache_key}: {features['keypoints'].shape[1]} keypoints")
        
        return features
    
    def match_features(
        self,
        features0: Dict,
        features1: Dict,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match features using LightGlue.
        
        Args:
            features0: Features from first image (query)
            features1: Features from second image (reference)
            
        Returns:
            Tuple of (keypoints0, keypoints1, confidence_scores)
            - keypoints0: Matched keypoints from image 0 (N, 2)
            - keypoints1: Matched keypoints from image 1 (N, 2)
            - confidence_scores: Match confidence scores (N,)
        """
        with torch.no_grad():
            # Perform matching
            matches = self.matcher({
                'image0': features0,
                'image1': features1,
            })
        
        # Extract matched keypoints
        matches_idx = matches['matches'][0]  # (N, 2)
        
        # Handle different LightGlue versions - try both key names
        if 'matching_scores' in matches:
            match_confidence = matches['matching_scores'][0]  # (N,) - older versions
        elif 'scores' in matches:
            match_confidence = matches['scores'][0]  # (N,) - newer versions
        elif 'mscores0' in matches:
            match_confidence = matches['mscores0'][0]  # (N,) - alternative key
        else:
            # Fallback: create uniform confidence scores
            logger.warning("No confidence scores found in matches, using uniform scores")
            match_confidence = torch.ones(len(matches_idx), device=matches_idx.device)
        
        # Get valid matches (not -1)
        valid = matches_idx[:, 0] != -1
        matches_idx = matches_idx[valid]
        match_confidence = match_confidence[valid]
        
        if len(matches_idx) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Extract keypoint coordinates
        kpts0 = features0['keypoints'][0][matches_idx[:, 0]].cpu().numpy()
        kpts1 = features1['keypoints'][0][matches_idx[:, 1]].cpu().numpy()
        confidence = match_confidence.cpu().numpy()
        
        return kpts0, kpts1, confidence
    
    def match_images(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        cache_key0: Optional[str] = None,
        cache_key1: Optional[str] = None,
        min_matches: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Complete matching pipeline: extract features and match.
        
        Args:
            image0: Query image (e.g., drone frame)
            image1: Reference image (e.g., satellite map)
            cache_key0: Cache key for image0 features
            cache_key1: Cache key for image1 features
            min_matches: Minimum number of matches required
            
        Returns:
            Tuple of (src_pts, dst_pts, confidence, num_matches)
            - src_pts: Matched points in image0 (N, 1, 2) for cv2.findHomography
            - dst_pts: Matched points in image1 (N, 1, 2) for cv2.findHomography
            - confidence: Match confidence scores (N,)
            - num_matches: Total number of matches found
        """
        # Extract features
        features0 = self.extract_features(image0, cache_key=cache_key0)
        features1 = self.extract_features(image1, cache_key=cache_key1)
        
        # Match features
        kpts0, kpts1, confidence = self.match_features(features0, features1)
        
        num_matches = len(kpts0)
        
        if num_matches < min_matches:
            logger.debug(f"Insufficient matches: {num_matches} < {min_matches}")
            return np.array([]), np.array([]), np.array([]), num_matches
        
        # Reshape for cv2.findHomography compatibility
        src_pts = kpts0.reshape(-1, 1, 2).astype(np.float32)
        dst_pts = kpts1.reshape(-1, 1, 2).astype(np.float32)
        
        return src_pts, dst_pts, confidence, num_matches
    
    def compute_homography_with_confidence(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        confidence: np.ndarray,
        ransac_threshold: float = 5.0,
        confidence_weight: float = 0.3,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """
        Compute homography with confidence-weighted RANSAC.
        
        Args:
            src_pts: Source points (N, 1, 2)
            dst_pts: Destination points (N, 1, 2)
            confidence: Match confidence scores (N,)
            ransac_threshold: RANSAC reprojection threshold
            confidence_weight: Weight for confidence in inlier selection (0-1)
            
        Returns:
            Tuple of (homography_matrix, inlier_mask, num_inliers)
        """
        if len(src_pts) < 4:
            return None, None, 0
        
        # Standard RANSAC homography
        H, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            ransac_threshold,
        )
        
        if H is None:
            return None, None, 0
        
        # Optionally weight inliers by confidence
        if confidence_weight > 0 and len(confidence) > 0:
            # Combine geometric inliers with confidence
            mask_float = mask.ravel().astype(float)
            confidence_norm = (confidence - confidence.min()) / (confidence.max() - confidence.min() + 1e-8)
            combined_score = (1 - confidence_weight) * mask_float + confidence_weight * confidence_norm
            
            # Update mask based on combined score
            threshold = 0.5
            mask = (combined_score > threshold).astype(np.uint8).reshape(-1, 1)
        
        num_inliers = int(mask.sum())
        
        return H, mask, num_inliers
    
    def clear_cache(self):
        """Clear the feature cache to free memory."""
        self.feature_cache.clear()
        logger.debug("Feature cache cleared")
    
    def get_cache_info(self) -> Dict[str, int]:
        """Get information about cached features."""
        info = {}
        for key, features in self.feature_cache.items():
            info[key] = features['keypoints'].shape[1]
        return info
    
    def draw_matches_visualization(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        kpts0: np.ndarray,
        kpts1: np.ndarray,
        mask: Optional[np.ndarray] = None,
        confidence: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Draw match visualization similar to cv2.drawMatches.
        
        Args:
            image0: First image (query - drone frame)
            image1: Second image (reference - satellite map)
            kpts0: Keypoints from image0 (N, 2)
            kpts1: Keypoints from image1 (N, 2)
            mask: Inlier mask (N, 1) - optional
            confidence: Match confidence scores (N,) - optional
            
        Returns:
            Visualization image with matches drawn
        """
        logger.debug(f"Drawing matches: image0 shape={image0.shape}, image1 shape={image1.shape}")
        logger.debug(f"Keypoints: kpts0={len(kpts0)}, kpts1={len(kpts1)}")
        
        # Convert to grayscale if needed
        if len(image0.shape) == 3:
            img0_gray = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
        else:
            img0_gray = image0.copy()
            
        if len(image1.shape) == 3:
            img1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        else:
            img1_gray = image1.copy()
        
        # Convert to color for visualization
        img0_color = cv2.cvtColor(img0_gray, cv2.COLOR_GRAY2BGR)
        img1_color = cv2.cvtColor(img1_gray, cv2.COLOR_GRAY2BGR)
        
        # Get dimensions
        h0, w0 = img0_color.shape[:2]
        h1, w1 = img1_color.shape[:2]
        
        logger.debug(f"Image dimensions: img0=({h0}, {w0}), img1=({h1}, {w1})")
        
        # Create side-by-side image
        h_max = max(h0, h1)
        vis = np.zeros((h_max, w0 + w1, 3), dtype=np.uint8)
        vis[:h0, :w0] = img0_color
        vis[:h1, w0:w0+w1] = img1_color
        
        # Filter by mask if provided
        if mask is not None:
            mask_bool = mask.ravel().astype(bool)
            kpts0_draw = kpts0[mask_bool]
            kpts1_draw = kpts1[mask_bool]
            if confidence is not None:
                confidence_draw = confidence[mask_bool]
            else:
                confidence_draw = None
            logger.debug(f"After mask filtering: {len(kpts0_draw)} inliers")
        else:
            kpts0_draw = kpts0
            kpts1_draw = kpts1
            confidence_draw = confidence
            logger.debug(f"No mask provided, drawing all {len(kpts0_draw)} matches")
        
        # Draw matches
        for i in range(len(kpts0_draw)):
            pt0 = tuple(kpts0_draw[i].astype(int))
            pt1 = tuple((kpts1_draw[i] + np.array([w0, 0])).astype(int))
            
            # Color based on confidence if available
            if confidence_draw is not None and len(confidence_draw) > i:
                conf = float(confidence_draw[i])
                # Green (high confidence) to yellow (low confidence)
                color = (0, int(255 * conf), int(255 * (1 - conf)))
            else:
                # Default green for inliers
                color = (0, 255, 0)
            
            # Draw line
            cv2.line(vis, pt0, pt1, color, 1, cv2.LINE_AA)
            
            # Draw keypoints
            cv2.circle(vis, pt0, 3, color, -1, cv2.LINE_AA)
            cv2.circle(vis, pt1, 3, color, -1, cv2.LINE_AA)
        
        # Add text with match statistics
        num_matches = len(kpts0_draw)
        text = f"Inlier Matches: {num_matches}"
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        logger.debug(f"Visualization created: shape={vis.shape}")
        return vis


def create_matcher(
    use_gpu: bool = True,
    max_keypoints: int = 2048,
    match_threshold: float = 0.2,
    resize_max: Optional[int] = 1024,
) -> SuperPointLightGlueMatcher:
    """
    Factory function to create a configured matcher.
    
    Args:
        use_gpu: Use GPU if available
        max_keypoints: Maximum keypoints per image
        match_threshold: Matching confidence threshold
        resize_max: Max image dimension for efficiency
        
    Returns:
        Configured SuperPointLightGlueMatcher instance
    """
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    
    return SuperPointLightGlueMatcher(
        device=device,
        max_num_keypoints=max_keypoints,
        match_threshold=match_threshold,
        resize_max=resize_max,
    )
