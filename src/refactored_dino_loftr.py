"""
DINOv2 + LoFTR feature matching module for high-accuracy trajectory tracking.

Uses:
- DINOv2: Global image retrieval and coarse localization
- LoFTR: Detector-free local feature matcher for dense matching

Advantages over SuperPoint+LightGlue:
- Better in low-texture areas (forests, fields)
- Dense matching without keypoint detection
- Robust to challenging lighting and repetitive patterns
"""

import torch
import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Handles image preprocessing for LoFTR and DINOv2."""
    
    def __init__(self, resize_max: Optional[int] = 840):
        self.resize_max = resize_max
    
    def preprocess_for_loftr(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess image for LoFTR matching.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Preprocessed tensor and original shape (H, W)
        """
        gray = self._to_grayscale(image)
        original_shape = gray.shape[:2]
        
        if self.resize_max is not None:
            gray = self._resize_for_efficiency(gray)
        
        tensor = torch.from_numpy(gray).float()[None, None] / 255.0
        return tensor, original_shape
    
    def preprocess_for_dino(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for DINOv2 feature extraction.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Preprocessed tensor (1, 3, 224, 224)
        """
        rgb = self._to_rgb(image)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        normalized = self._normalize_dino(resized)
        tensor = torch.from_numpy(normalized).float().permute(2, 0, 1)[None]
        return tensor
    
    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if needed."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    @staticmethod
    def _to_rgb(image: np.ndarray) -> np.ndarray:
        """Convert image to RGB."""
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def _resize_for_efficiency(self, gray: np.ndarray) -> np.ndarray:
        """Resize image if larger than resize_max."""
        h, w = gray.shape[:2]
        max_dim = max(h, w)
        if max_dim > self.resize_max:
            scale = self.resize_max / max_dim
            new_h = (int(h * scale) // 8) * 8  # Divisible by 8 for LoFTR
            new_w = (int(w * scale) // 8) * 8
            return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return gray
    
    @staticmethod
    def _normalize_dino(rgb: np.ndarray) -> np.ndarray:
        """Apply ImageNet normalization for DINOv2."""
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        return (rgb / 255.0 - mean) / std


class FeatureCache:
    """Manages feature caching for repeated matching."""
    
    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self.dino_cache: Dict[str, torch.Tensor] = {}
    
    def get_dino(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve cached DINOv2 features."""
        if self.use_cache and key in self.dino_cache:
            logger.debug(f"Using cached DINO features for {key}")
            return self.dino_cache[key]
        return None
    
    def store_dino(self, key: str, features: torch.Tensor):
        """Store DINOv2 features in cache."""
        if self.use_cache:
            self.dino_cache[key] = features
            logger.debug(f"Cached DINO features for {key}")
    
    def clear(self):
        """Clear all cached features."""
        self.dino_cache.clear()
        logger.debug("Feature cache cleared")
    
    def get_info(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {'dino_cache_size': len(self.dino_cache)}


class DinoRetriever:
    """Handles DINOv2-based global image retrieval."""
    
    def __init__(self, device: torch.device, model_name: str = "dinov2_vitb14"):
        self.device = device
        self.model = self._load_model(model_name)
        self.preprocessor = ImagePreprocessor()
    
    def _load_model(self, model_name: str) -> Optional[torch.nn.Module]:
        """Load DINOv2 model."""
        try:
            logger.info(f"Loading DINOv2 model: {model_name}")
            model = torch.hub.load('facebookresearch/dinov2', model_name)
            model = model.eval().to(self.device)
            logger.info("DINOv2 model loaded successfully")
            return model
        except Exception as e:
            logger.warning(f"Failed to load DINOv2: {e}")
            logger.warning("Continuing without global retrieval (LoFTR only)")
            return None
    
    def extract_features(self, image: np.ndarray) -> Optional[torch.Tensor]:
        """Extract global DINOv2 features."""
        if self.model is None:
            return None
        
        tensor = self.preprocessor.preprocess_for_dino(image).to(self.device)
        
        with torch.no_grad():
            features = self.model(tensor)
        
        return features
    
    def compute_similarity(self, features1: torch.Tensor, features2: torch.Tensor) -> float:
        """Compute cosine similarity between feature vectors."""
        similarity = torch.nn.functional.cosine_similarity(features1, features2)
        return similarity.item()
    
    @property
    def is_available(self) -> bool:
        """Check if DINOv2 is available."""
        return self.model is not None


class LoFTRMatcher:
    """Handles LoFTR-based dense feature matching."""
    
    def __init__(self, device: torch.device, model_type: str = "outdoor", 
                 resize_max: Optional[int] = 840):
        self.device = device
        self.matcher = self._load_model(model_type)
        self.preprocessor = ImagePreprocessor(resize_max)
    
    def _load_model(self, model_type: str) -> torch.nn.Module:
        """Load LoFTR model."""
        try:
            from kornia.feature import LoFTR as LoFTR_kornia
            
            model = LoFTR_kornia(pretrained=model_type).eval().to(self.device)
            logger.info(f"Loaded LoFTR {model_type} model")
            return model
        except ImportError as e:
            logger.error(f"Failed to import LoFTR from kornia: {e}")
            logger.error("Please install: pip install kornia kornia-rs")
            raise
    
    def match(self, image0: np.ndarray, image1: np.ndarray, 
              min_matches: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Match features between two images.
        
        Args:
            image0: Query image (e.g., drone frame)
            image1: Reference image (e.g., satellite map)
            min_matches: Minimum number of matches required
            
        Returns:
            Tuple of (src_pts, dst_pts, confidence, num_matches)
        """
        tensor0, original_shape0 = self.preprocessor.preprocess_for_loftr(image0)
        tensor1, original_shape1 = self.preprocessor.preprocess_for_loftr(image1)
        
        tensor0 = tensor0.to(self.device)
        tensor1 = tensor1.to(self.device)
        
        with torch.no_grad():
            correspondences = self.matcher({'image0': tensor0, 'image1': tensor1})
        
        mkpts0 = correspondences['keypoints0'].cpu().numpy()
        mkpts1 = correspondences['keypoints1'].cpu().numpy()
        confidence = correspondences['confidence'].cpu().numpy()
        
        num_matches = len(mkpts0)
        
        if num_matches < min_matches:
            logger.debug(f"Insufficient matches: {num_matches} < {min_matches}")
            return np.array([]), np.array([]), np.array([]), num_matches
        
        # Scale keypoints back to original resolution
        mkpts0 = self._scale_keypoints(mkpts0, original_shape0, tensor0.shape[2:])
        mkpts1 = self._scale_keypoints(mkpts1, original_shape1, tensor1.shape[2:])
        
        # Reshape for cv2.findHomography
        src_pts = mkpts0.reshape(-1, 1, 2).astype(np.float32)
        dst_pts = mkpts1.reshape(-1, 1, 2).astype(np.float32)
        
        return src_pts, dst_pts, confidence, num_matches
    
    @staticmethod
    def _scale_keypoints(keypoints: np.ndarray, original_shape: Tuple[int, int], 
                        resized_shape: Tuple[int, int]) -> np.ndarray:
        """Scale keypoints from resized to original image coordinates."""
        h_orig, w_orig = original_shape
        h_resized, w_resized = resized_shape
        
        if h_orig == h_resized and w_orig == w_resized:
            return keypoints
        
        scale_x = w_orig / w_resized
        scale_y = h_orig / h_resized
        
        scaled = keypoints.copy()
        scaled[:, 0] *= scale_x
        scaled[:, 1] *= scale_y
        
        return scaled


class HomographyEstimator:
    """Computes homography with confidence-weighted RANSAC."""
    
    @staticmethod
    def compute(src_pts: np.ndarray, dst_pts: np.ndarray, confidence: np.ndarray,
                ransac_threshold: float = 3.0, 
                confidence_weight: float = 0.2) -> Tuple[Optional[np.ndarray], 
                                                         Optional[np.ndarray], int]:
        """
        Compute homography with confidence weighting.
        
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
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
        
        if H is None:
            return None, None, 0
        
        # Apply confidence weighting if requested
        if confidence_weight > 0 and len(confidence) > 0:
            mask = HomographyEstimator._apply_confidence_weighting(
                mask, confidence, confidence_weight
            )
        
        num_inliers = int(mask.sum())
        return H, mask, num_inliers
    
    @staticmethod
    def _apply_confidence_weighting(mask: np.ndarray, confidence: np.ndarray, 
                                    weight: float) -> np.ndarray:
        """Apply confidence weighting to inlier mask."""
        mask_float = mask.ravel().astype(float)
        confidence_norm = (confidence - confidence.min()) / (
            confidence.max() - confidence.min() + 1e-8
        )
        combined_score = (1 - weight) * mask_float + weight * confidence_norm
        return (combined_score > 0.5).astype(np.uint8).reshape(-1, 1)


class MatchVisualizer:
    """Creates match visualizations."""
    
    @staticmethod
    def draw_matches(image0: np.ndarray, image1: np.ndarray, 
                    kpts0: np.ndarray, kpts1: np.ndarray,
                    mask: Optional[np.ndarray] = None,
                    confidence: Optional[np.ndarray] = None,
                    max_draw: int = 500) -> np.ndarray:
        """
        Draw match visualization.
        
        Args:
            image0: Query image
            image1: Reference image
            kpts0: Keypoints from image0 (N, 2)
            kpts1: Keypoints from image1 (N, 2)
            mask: Inlier mask (N, 1)
            confidence: Match confidence scores (N,)
            max_draw: Maximum matches to draw
            
        Returns:
            Visualization image
        """
        logger.debug(f"Drawing matches: img0={image0.shape}, img1={image1.shape}")
        logger.debug(f"Keypoints: {len(kpts0)}, {len(kpts1)}")
        
        img0_color = MatchVisualizer._to_color(image0)
        img1_color = MatchVisualizer._to_color(image1)
        
        vis = MatchVisualizer._create_canvas(img0_color, img1_color)
        
        # Filter by mask
        kpts0_draw, kpts1_draw, confidence_draw = MatchVisualizer._filter_by_mask(
            kpts0, kpts1, confidence, mask
        )
        
        # Sample if too many matches
        if len(kpts0_draw) > max_draw:
            kpts0_draw, kpts1_draw, confidence_draw = MatchVisualizer._sample_matches(
                kpts0_draw, kpts1_draw, confidence_draw, max_draw
            )
        
        # Draw matches
        w0 = img0_color.shape[1]
        for i in range(len(kpts0_draw)):
            pt0 = tuple(kpts0_draw[i].astype(int))
            pt1 = tuple((kpts1_draw[i] + np.array([w0, 0])).astype(int))
            color = MatchVisualizer._get_color(confidence_draw, i)
            
            cv2.line(vis, pt0, pt1, color, 1, cv2.LINE_AA)
            cv2.circle(vis, pt0, 2, color, -1, cv2.LINE_AA)
            cv2.circle(vis, pt1, 2, color, -1, cv2.LINE_AA)
        
        # Add statistics
        text = f"Matches: {len(kpts0_draw)}"
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2, cv2.LINE_AA)
        
        logger.debug(f"Visualization created: {vis.shape}")
        return vis
    
    @staticmethod
    def _to_color(image: np.ndarray) -> np.ndarray:
        """Convert image to BGR color."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    @staticmethod
    def _create_canvas(img0: np.ndarray, img1: np.ndarray) -> np.ndarray:
        """Create side-by-side canvas."""
        h0, w0 = img0.shape[:2]
        h1, w1 = img1.shape[:2]
        h_max = max(h0, h1)
        vis = np.zeros((h_max, w0 + w1, 3), dtype=np.uint8)
        vis[:h0, :w0] = img0
        vis[:h1, w0:w0+w1] = img1
        return vis
    
    @staticmethod
    def _filter_by_mask(kpts0: np.ndarray, kpts1: np.ndarray, 
                       confidence: Optional[np.ndarray],
                       mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, 
                                                            Optional[np.ndarray]]:
        """Filter keypoints by mask."""
        if mask is not None:
            mask_bool = mask.ravel().astype(bool)
            kpts0_filtered = kpts0[mask_bool]
            kpts1_filtered = kpts1[mask_bool]
            conf_filtered = confidence[mask_bool] if confidence is not None else None
            logger.debug(f"After mask filtering: {len(kpts0_filtered)} inliers")
        else:
            kpts0_filtered = kpts0
            kpts1_filtered = kpts1
            conf_filtered = confidence
            logger.debug(f"No mask, drawing all {len(kpts0_filtered)} matches")
        
        return kpts0_filtered, kpts1_filtered, conf_filtered
    
    @staticmethod
    def _sample_matches(kpts0: np.ndarray, kpts1: np.ndarray, 
                       confidence: Optional[np.ndarray],
                       max_draw: int) -> Tuple[np.ndarray, np.ndarray, 
                                               Optional[np.ndarray]]:
        """Sample matches for visualization."""
        indices = np.random.choice(len(kpts0), max_draw, replace=False)
        logger.debug(f"Sampled {max_draw} matches for visualization")
        return (kpts0[indices], kpts1[indices], 
                confidence[indices] if confidence is not None else None)
    
    @staticmethod
    def _get_color(confidence: Optional[np.ndarray], index: int) -> Tuple[int, int, int]:
        """Get color based on confidence."""
        if confidence is not None and len(confidence) > index:
            conf = float(confidence[index])
            return (0, int(255 * conf), int(255 * (1 - conf)))
        return (0, 255, 0)


class DinoLoFTRMatcher:
    """
    High-accuracy feature matcher using DINOv2 for retrieval and LoFTR for matching.
    
    Advantages:
    - Works in low-texture environments
    - Dense matching without keypoint detection
    - Robust to viewpoint and scale changes
    - GPU accelerated
    """
    
    def __init__(self, device: Optional[str] = None, loftr_model: str = "outdoor",
                 resize_max: Optional[int] = 840, use_cache: bool = True,
                 dino_model: str = "dinov2_vitb14"):
        """
        Initialize DINOv2 + LoFTR matcher.
        
        Args:
            device: 'cuda', 'cpu', or None (auto-detect)
            loftr_model: 'outdoor' or 'indoor'
            resize_max: Max image dimension for efficiency
            use_cache: Cache features for repeated matching
            dino_model: DINOv2 model variant
        """
        self.device = self._get_device(device)
        logger.info(f"Initializing DINOv2 + LoFTR on device: {self.device}")
        
        self.loftr = LoFTRMatcher(self.device, loftr_model, resize_max)
        self.dino = DinoRetriever(self.device, dino_model)
        self.cache = FeatureCache(use_cache)
        self.homography_estimator = HomographyEstimator()
        self.visualizer = MatchVisualizer()
    
    @staticmethod
    def _get_device(device: Optional[str]) -> torch.device:
        """Determine the device to use."""
        if device is None:
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def match_images(self, image0: np.ndarray, image1: np.ndarray,
                    cache_key0: Optional[str] = None, cache_key1: Optional[str] = None,
                    min_matches: int = 10) -> Tuple[np.ndarray, np.ndarray, 
                                                    np.ndarray, int]:
        """
        Complete matching pipeline.
        
        Args:
            image0: Query image (drone frame)
            image1: Reference image (satellite map)
            cache_key0: Cache key for image0
            cache_key1: Cache key for image1
            min_matches: Minimum matches required
            
        Returns:
            Tuple of (src_pts, dst_pts, confidence, num_matches)
        """
        # Optional DINOv2 coarse verification
        if self.dino.is_available:
            if not self._verify_with_dino(image0, image1, cache_key0, cache_key1):
                return np.array([]), np.array([]), np.array([]), 0
        
        # LoFTR matching
        return self.loftr.match(image0, image1, min_matches)
    
    def _verify_with_dino(self, image0: np.ndarray, image1: np.ndarray,
                         cache_key0: Optional[str], 
                         cache_key1: Optional[str]) -> bool:
        """Verify image similarity using DINOv2."""
        dino0 = self._get_or_extract_dino(image0, cache_key0)
        dino1 = self._get_or_extract_dino(image1, cache_key1)
        
        if dino0 is None or dino1 is None:
            return True
        
        similarity = self.dino.compute_similarity(dino0, dino1)
        logger.debug(f"DINO similarity: {similarity:.3f}")
        
        if similarity < 0.3:
            logger.debug("Images too dissimilar (DINO), skipping LoFTR")
            return False
        
        return True
    
    def _get_or_extract_dino(self, image: np.ndarray, 
                            cache_key: Optional[str]) -> Optional[torch.Tensor]:
        """Get cached or extract new DINOv2 features."""
        if cache_key is not None:
            cached = self.cache.get_dino(cache_key)
            if cached is not None:
                return cached
        
        features = self.dino.extract_features(image)
        
        if features is not None and cache_key is not None:
            self.cache.store_dino(cache_key, features)
        
        return features
    
    def compute_homography_with_confidence(
        self, src_pts: np.ndarray, dst_pts: np.ndarray, confidence: np.ndarray,
        ransac_threshold: float = 3.0, confidence_weight: float = 0.2
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """
        Compute homography with confidence-weighted RANSAC.
        
        Args:
            src_pts: Source points (N, 1, 2)
            dst_pts: Destination points (N, 1, 2)
            confidence: Match confidence scores (N,)
            ransac_threshold: RANSAC threshold
            confidence_weight: Confidence weight (0-1)
            
        Returns:
            Tuple of (homography_matrix, inlier_mask, num_inliers)
        """
        return self.homography_estimator.compute(
            src_pts, dst_pts, confidence, ransac_threshold, confidence_weight
        )
    
    def draw_matches_visualization(
        self, image0: np.ndarray, image1: np.ndarray,
        kpts0: np.ndarray, kpts1: np.ndarray,
        mask: Optional[np.ndarray] = None,
        confidence: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Draw match visualization.
        
        Args:
            image0: Query image
            image1: Reference image
            kpts0: Keypoints from image0 (N, 2)
            kpts1: Keypoints from image1 (N, 2)
            mask: Inlier mask (N, 1)
            confidence: Match confidence scores (N,)
            
        Returns:
            Visualization image
        """
        return self.visualizer.draw_matches(
            image0, image1, kpts0, kpts1, mask, confidence
        )
    
    def clear_cache(self):
        """Clear the feature cache."""
        self.cache.clear()
    
    def get_cache_info(self) -> Dict[str, int]:
        """Get cache statistics."""
        return self.cache.get_info()


def create_dino_loftr_matcher(use_gpu: bool = True, loftr_model: str = "outdoor",
                              resize_max: Optional[int] = 840) -> DinoLoFTRMatcher:
    """
    Factory function to create a configured matcher.
    
    Args:
        use_gpu: Use GPU if available
        loftr_model: 'outdoor' or 'indoor'
        resize_max: Max image dimension
        
    Returns:
        Configured DinoLoFTRMatcher instance
    """
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    return DinoLoFTRMatcher(device=device, loftr_model=loftr_model, resize_max=resize_max)
