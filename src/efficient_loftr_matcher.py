"""
EfficientLoFTR (ELoFTR) feature matching module for high-performance trajectory tracking.

This module uses EfficientLoFTR from Hugging Face (zju-community/efficientloftr):
- Faster inference than standard LoFTR (2-3x speedup)
- Maintains high accuracy for dense matching
- Better memory efficiency
- Optimized architecture for real-time applications

Advantages over standard LoFTR:
- 2-3x faster inference speed
- Lower memory footprint
- Maintains matching quality
- Better suited for real-time drone navigation
"""

import torch
import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import logging
from transformers import AutoModel, AutoImageProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EfficientLoFTRMatcher:
    """
    High-performance feature matcher using EfficientLoFTR from Hugging Face.
    
    EfficientLoFTR: Optimized detector-free local feature matcher
    - Faster than standard LoFTR (2-3x speedup)
    - Lower memory usage
    - Maintains high matching accuracy
    - Ideal for real-time drone navigation
    
    Model: zju-community/efficientloftr from Hugging Face
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        model_name: str = "zju-community/efficientloftr",
        resize_max: Optional[int] = 840,
        use_cache: bool = True,
        use_mixed_precision: bool = True,
    ):
        """
        Initialize EfficientLoFTR matcher.
        
        Args:
            device: 'cuda', 'cpu', or None (auto-detect)
            model_name: Hugging Face model name
            resize_max: Resize images to this max dimension for efficiency (None = no resize)
            use_cache: Cache features for repeated matching (useful for map images)
            use_mixed_precision: Use FP16 for faster inference on GPU
        """
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initializing EfficientLoFTR on device: {self.device}")
        
        self.resize_max = resize_max
        self.use_cache = use_cache
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        self.model_name = model_name
        
        # Initialize EfficientLoFTR from Hugging Face
        try:
            logger.info(f"Loading EfficientLoFTR model: {model_name}")
            
            # Load model and processor
            self.matcher = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True
            ).eval().to(self.device)
            
            # Try to load image processor if available
            try:
                self.processor = AutoImageProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                logger.info("Loaded image processor")
            except Exception as e:
                logger.warning(f"Could not load image processor: {e}")
                self.processor = None
            
            logger.info("✓ EfficientLoFTR model loaded successfully")
            
            if self.use_mixed_precision:
                logger.info("✓ Mixed precision (FP16) enabled for faster inference")
                
        except ImportError as e:
            logger.error(f"Failed to import from transformers: {e}")
            logger.error("Please install: pip install transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load EfficientLoFTR model: {e}")
            logger.error("Make sure you have internet connection for first-time download")
            raise
        
        # Feature cache
        self.feature_cache: Dict[str, Dict] = {}
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess image for EfficientLoFTR.
        
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
                # EfficientLoFTR works best with dimensions divisible by 8
                new_h = (new_h // 8) * 8
                new_w = (new_w // 8) * 8
                gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Use processor if available, otherwise manual normalization
        if self.processor is not None:
            try:
                # Convert to RGB for processor
                if len(gray.shape) == 2:
                    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                else:
                    rgb = gray
                
                # Process with HF processor
                inputs = self.processor(images=rgb, return_tensors="pt")
                tensor = inputs['pixel_values'].to(self.device)
            except Exception as e:
                logger.debug(f"Processor failed, using manual normalization: {e}")
                # Fallback to manual normalization
                tensor = torch.from_numpy(gray).float()[None, None] / 255.0
                tensor = tensor.to(self.device)
        else:
            # Manual normalization
            tensor = torch.from_numpy(gray).float()[None, None] / 255.0
            tensor = tensor.to(self.device)
        
        return tensor, original_shape
    
    def match_images(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        cache_key0: Optional[str] = None,
        cache_key1: Optional[str] = None,
        min_matches: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Complete matching pipeline using EfficientLoFTR.
        
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
        # Preprocess images
        tensor0, original_shape0 = self.preprocess_image(image0)
        tensor1, original_shape1 = self.preprocess_image(image1)
        
        # Match with EfficientLoFTR
        with torch.no_grad():
            # Use mixed precision if enabled
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    # Prepare input batch
                    batch = {
                        'image0': tensor0,
                        'image1': tensor1,
                    }
                    
                    # Run inference
                    correspondences = self.matcher(batch)
            else:
                # Prepare input batch
                batch = {
                    'image0': tensor0,
                    'image1': tensor1,
                }
                
                # Run inference
                correspondences = self.matcher(batch)
        
        # Extract matches - handle different output formats
        try:
            # Try standard LoFTR format
            if 'keypoints0' in correspondences:
                mkpts0 = correspondences['keypoints0'].cpu().numpy()
                mkpts1 = correspondences['keypoints1'].cpu().numpy()
                confidence = correspondences.get('confidence', 
                                                torch.ones(len(mkpts0))).cpu().numpy()
            # Try alternative format
            elif 'mkpts0' in correspondences:
                mkpts0 = correspondences['mkpts0'].cpu().numpy()
                mkpts1 = correspondences['mkpts1'].cpu().numpy()
                confidence = correspondences.get('mconf', 
                                                torch.ones(len(mkpts0))).cpu().numpy()
            else:
                logger.error(f"Unknown output format. Keys: {correspondences.keys()}")
                return np.array([]), np.array([]), np.array([]), 0
                
        except Exception as e:
            logger.error(f"Error extracting matches: {e}")
            return np.array([]), np.array([]), np.array([]), 0
        
        num_matches = len(mkpts0)
        
        if num_matches < min_matches:
            logger.debug(f"Insufficient matches: {num_matches} < {min_matches}")
            return np.array([]), np.array([]), np.array([]), num_matches
        
        # Scale keypoints back to original resolution if resized
        if self.resize_max is not None:
            h0, w0 = original_shape0
            h0_resized, w0_resized = tensor0.shape[2:]
            scale_x0 = w0 / w0_resized
            scale_y0 = h0 / h0_resized
            mkpts0[:, 0] *= scale_x0
            mkpts0[:, 1] *= scale_y0
            
            h1, w1 = original_shape1
            h1_resized, w1_resized = tensor1.shape[2:]
            scale_x1 = w1 / w1_resized
            scale_y1 = h1 / h1_resized
            mkpts1[:, 0] *= scale_x1
            mkpts1[:, 1] *= scale_y1
        
        # Reshape for cv2.findHomography compatibility
        src_pts = mkpts0.reshape(-1, 1, 2).astype(np.float32)
        dst_pts = mkpts1.reshape(-1, 1, 2).astype(np.float32)
        
        return src_pts, dst_pts, confidence, num_matches
    
    def compute_homography_with_confidence(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        confidence: np.ndarray,
        ransac_threshold: float = 3.0,
        confidence_weight: float = 0.2,
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
        logger.debug(f"Drawing EfficientLoFTR matches: image0 shape={image0.shape}, image1 shape={image1.shape}")
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
        
        # Sample matches if too many (for visualization clarity)
        max_draw = 500
        if len(kpts0_draw) > max_draw:
            indices = np.random.choice(len(kpts0_draw), max_draw, replace=False)
            kpts0_draw = kpts0_draw[indices]
            kpts1_draw = kpts1_draw[indices]
            if confidence_draw is not None:
                confidence_draw = confidence_draw[indices]
            logger.debug(f"Sampled {max_draw} matches for visualization")
        
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
            
            # Draw keypoints (smaller for dense matches)
            cv2.circle(vis, pt0, 2, color, -1, cv2.LINE_AA)
            cv2.circle(vis, pt1, 2, color, -1, cv2.LINE_AA)
        
        # Add text with match statistics
        num_matches = len(kpts0_draw)
        text = f"EfficientLoFTR Inlier Matches: {num_matches}"
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        logger.debug(f"Visualization created: shape={vis.shape}")
        return vis
    
    def clear_cache(self):
        """Clear the feature cache to free memory."""
        self.feature_cache.clear()
        logger.debug("Feature cache cleared")
    
    def get_cache_info(self) -> Dict[str, int]:
        """Get information about cached features."""
        info = {
            'feature_cache_size': len(self.feature_cache),
        }
        return info


def create_efficient_loftr_matcher(
    use_gpu: bool = True,
    resize_max: Optional[int] = 840,
    use_mixed_precision: bool = True,
) -> EfficientLoFTRMatcher:
    """
    Factory function to create a configured EfficientLoFTR matcher.
    
    Args:
        use_gpu: Use GPU if available
        resize_max: Max image dimension for efficiency
        use_mixed_precision: Use FP16 for faster inference
        
    Returns:
        Configured EfficientLoFTRMatcher instance
    """
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    
    return EfficientLoFTRMatcher(
        device=device,
        resize_max=resize_max,
        use_mixed_precision=use_mixed_precision,
    )
