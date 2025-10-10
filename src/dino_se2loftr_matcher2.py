"""
DINOv2 + SE2-LoFTR feature matching module for rotation-robust trajectory tracking.

This module uses:
- DINOv2: For global image retrieval and coarse localization
- SE2-LoFTR: Rotation equivariant detector-free local feature matcher

Advantages over standard LoFTR:
- Robust to arbitrary rotations (pan, roll, yaw changes)
- Maintains dense matching capabilities
- Better for drone scenarios with orientation changes
- Works in low-texture areas
"""

import torch
import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DinoSE2LoFTRMatcher:
    """
    Rotation-robust feature matcher using DINOv2 for retrieval and SE2-LoFTR for dense matching.
    
    SE2-LoFTR: Rotation equivariant LoFTR using steerable CNNs
    DINOv2: Self-supervised vision transformer for global image understanding
    
    Advantages:
    - Robust to arbitrary rotations (e.g., drone pan/roll)
    - Dense matching without explicit keypoint detection
    - Works in low-texture environments
    - GPU acceleration for faster processing
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        se2loftr_weights: str = "se2_loftr/weights/8rot.ckpt",
        resize_max: Optional[int] = 840,
        use_cache: bool = True,
        dino_model: str = "dinov2_vitb14",
    ):
        """
        Initialize DINOv2 + SE2-LoFTR matcher.
        
        Args:
            device: 'cuda', 'cpu', or None (auto-detect)
            se2loftr_weights: Path to SE2-LoFTR checkpoint weights
            resize_max: Resize images to this max dimension for efficiency (None = no resize)
            use_cache: Cache features for repeated matching
            dino_model: DINOv2 model variant
        """
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initializing DINOv2 + SE2-LoFTR on device: {self.device}")
        
        self.resize_max = resize_max
        self.use_cache = use_cache
        
        # Add SE2-LoFTR src to path (adjusted based on project structure)
        se2loftr_path = os.path.join(os.getcwd(), 'se2_loftr')
        se2loftr_src = os.path.join(se2loftr_path, 'src')
        if se2loftr_src not in sys.path:
            sys.path.insert(0, se2loftr_src)
        
        # Initialize SE2-LoFTR
        try:
            logger.info(f"Loading SE2-LoFTR from: {se2loftr_weights}")
            
            # Import SE2-LoFTR modules
            from loftr import LoFTR as SE2LoFTR
            from config.default import get_cfg_defaults
            
            # Load configuration
            config = get_cfg_defaults()
            
            # Update config for outdoor SE2-LoFTR with 8 rotations
            config.LOFTR.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'
            config.LOFTR.BACKBONE_TYPE = 'ResNetFPN_8_2'  # 8 rotations
            config.LOFTR.RESOLUTION = (8, 2)
            config.LOFTR.FINE_WINDOW_SIZE = 5
            config.LOFTR.FINE_CONCAT_COARSE_FEAT = True
            
            # Create model
            self.matcher = SE2LoFTR(config=config['LOFTR'])
            
            # Load checkpoint
            if not os.path.exists(se2loftr_weights):
                raise FileNotFoundError(f"SE2-LoFTR weights not found at: {se2loftr_weights}")
            
            checkpoint = torch.load(se2loftr_weights, map_location='cpu')
            
            # Extract state dict (handle different checkpoint formats)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Remove 'matcher.' prefix if present
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('matcher.'):
                    cleaned_state_dict[key[8:]] = value
                else:
                    cleaned_state_dict[key] = value
            
            # Load weights
            missing_keys, unexpected_keys = self.matcher.load_state_dict(cleaned_state_dict, strict=False)
            
            if missing_keys:
                logger.warning(f"Missing keys in checkpoint: {missing_keys[:5]}...")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys[:5]}...")
            
            self.matcher = self.matcher.eval().to(self.device)
            logger.info("✓ SE2-LoFTR model loaded successfully (rotation equivariant)")
            
        except ImportError as e:
            logger.error(f"Failed to import SE2-LoFTR: {e}")
            logger.error("Make sure se2_loftr folder with src/ is in the project root")
            logger.error("Install dependencies: pip install e2cnn pytorch-lightning")
            raise
        except Exception as e:
            logger.error(f"Failed to load SE2-LoFTR model: {e}")
            raise
        
        # Initialize DINOv2 (optional, for global retrieval)
        self.dino_model = None
        try:
            logger.info(f"Loading DINOv2 model: {dino_model}")
            self.dino_model = torch.hub.load('facebookresearch/dinov2', dino_model).eval().to(self.device)
            logger.info("✓ DINOv2 model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load DINOv2: {e}")
            logger.warning("Continuing without global retrieval (SE2-LoFTR only)")
        
        # Feature cache
        self.feature_cache: Dict[str, Dict] = {}
        self.dino_cache: Dict[str, torch.Tensor] = {}
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess image for SE2-LoFTR.
        
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
                # SE2-LoFTR works best with dimensions divisible by 8
                new_h = (new_h // 8) * 8
                new_w = (new_w // 8) * 8
                gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Convert to tensor and normalize
        tensor = torch.from_numpy(gray).float()[None, None] / 255.0
        tensor = tensor.to(self.device)
        
        return tensor, original_shape
    
    def extract_dino_features(self, image: np.ndarray, cache_key: Optional[str] = None) -> Optional[torch.Tensor]:
        """
        Extract DINOv2 global features for retrieval.
        
        Args:
            image: Input image (BGR)
            cache_key: Optional key for caching features
            
        Returns:
            Global feature vector or None if DINOv2 not available
        """
        if self.dino_model is None:
            return None
        
        # Check cache
        if cache_key is not None and self.use_cache and cache_key in self.dino_cache:
            logger.debug(f"Using cached DINO features for {cache_key}")
            return self.dino_cache[cache_key]
        
        # Preprocess for DINOv2 (expects RGB)
        if len(image.shape) == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224 for DINOv2
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (resized / 255.0 - mean) / std
        
        # Convert to tensor
        tensor = torch.from_numpy(normalized).float().permute(2, 0, 1)[None].to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.dino_model(tensor)
        
        # Cache if requested
        if cache_key is not None and self.use_cache:
            self.dino_cache[cache_key] = features
            logger.debug(f"Cached DINO features for {cache_key}")
        
        return features
    
    def match_images(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        cache_key0: Optional[str] = None,
        cache_key1: Optional[str] = None,
        min_matches: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Complete matching pipeline: SE2-LoFTR rotation-robust dense matching.
        
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
        # Optional: Use DINOv2 for coarse verification
        if self.dino_model is not None:
            dino0 = self.extract_dino_features(image0, cache_key=cache_key0)
            dino1 = self.extract_dino_features(image1, cache_key=cache_key1)
            
            if dino0 is not None and dino1 is not None:
                # Compute cosine similarity
                similarity = torch.nn.functional.cosine_similarity(dino0, dino1)
                logger.debug(f"DINO similarity: {similarity.item():.3f}")
                
                # Skip matching if images are too dissimilar
                if similarity.item() < 0.3:
                    logger.debug("Images too dissimilar (DINO), skipping SE2-LoFTR")
                    return np.array([]), np.array([]), np.array([]), 0
        
        # Preprocess images
        tensor0, original_shape0 = self.preprocess_image(image0)
        tensor1, original_shape1 = self.preprocess_image(image1)
        
        # Match with SE2-LoFTR (rotation equivariant)
        with torch.no_grad():
            batch = {
                'image0': tensor0,
                'image1': tensor1,
            }
            
            # Run SE2-LoFTR matcher
            self.matcher(batch)
            
            # Extract matches from batch
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            
            # Get confidence scores
            if 'mconf' in batch:
                confidence = batch['mconf'].cpu().numpy()
            else:
                # Fallback if confidence not available
                confidence = np.ones(len(mkpts0))
        
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
        Draw match visualization for SE2-LoFTR matches.
        
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
        logger.debug(f"Drawing SE2-LoFTR matches: image0 shape={image0.shape}, image1 shape={image1.shape}")
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
        text = f"SE2-LoFTR Inlier Matches: {num_matches} (Rotation Robust)"
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        logger.debug(f"Visualization created: shape={vis.shape}")
        return vis
    
    def clear_cache(self):
        """Clear the feature cache to free memory."""
        self.feature_cache.clear()
        self.dino_cache.clear()
        logger.debug("Feature cache cleared")
    
    def get_cache_info(self) -> Dict[str, int]:
        """Get information about cached features."""
        info = {
            'dino_cache_size': len(self.dino_cache),
        }
        return info


def create_dino_se2loftr_matcher(
    use_gpu: bool = True,
    se2loftr_weights: str = "se2_loftr/weights/8rot.ckpt",
    resize_max: Optional[int] = 840,
) -> DinoSE2LoFTRMatcher:
    """
    Factory function to create a configured DINOv2 + SE2-LoFTR matcher.
    
    Args:
        use_gpu: Use GPU if available
        se2loftr_weights: Path to SE2-LoFTR checkpoint weights
        resize_max: Max image dimension for efficiency
        
    Returns:
        Configured DinoSE2LoFTRMatcher instance
    """
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    
    return DinoSE2LoFTRMatcher(
        device=device,
        se2loftr_weights=se2loftr_weights,
        resize_max=resize_max,
    )