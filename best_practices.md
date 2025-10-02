# 📘 Project Best Practices

## 1. Project Purpose
Reconstruct and visualize a UAV/drone trajectory by matching drone frames (from a video or image crops) against a global satellite map or a dynamically loaded tile mosaic. The pipeline computes per-frame camera-to-map alignment via feature matching and homography estimation, then produces:
- A static trajectory map with the flight path drawn on top.
- A trajectory video showing the evolving path over time.

Domain highlights: computer vision, deep learning-based feature matching (SuperPoint + LightGlue), homography estimation, map/tile stitching, GPU acceleration, and OpenCV-based visualization/video encoding.

## 2. Project Structure --need to update as latest approach uses Optimised and Adaptive Dino-LoFTR model
- main.py
  - Orchestrates the pipeline. Chooses between three modes:
    1) Dynamic tile-based matching from a video
    2) Global map matching from a video
    3) Global map matching from a directory of frame crops
  - Handles results directory creation and invokes video generation.
- src/
  - **superpoint_lightglue_matcher.py** (NEW)
    - SuperPointLightGlueMatcher: High-accuracy learned feature matcher
    - Replaces classical SIFT with modern deep learning approach
    - Supports GPU acceleration, feature caching, confidence-weighted matching
    - create_matcher: Factory function for easy instantiation
  - create_trajectory_map.py
    - Core trajectory estimation logic with SuperPoint + LightGlue integration
    - generate_trajectory_map_from_crops: Process directory of frame images
    - generate_trajectory_map: Process video against global map
    - generate_dynamic_tile_matching: Dynamic N×N tile grid loading with moving center
    - All functions support use_superpoint parameter (default=True) for method selection
  - tile_loading_utilis.py
    - TileCache: simple LRU cache to bound memory usage
    - load_tile: loads tiles from disk using naming convention
    - load_and_stitch_grid / load_and_stitch_rect: assemble tiles via NumPy concatenation
    - update_center_from_match: translates pixel offset to tile index movement
  - create_video.py
    - create_trajectory_video: renders the path over the global map into a video
  - video_utils.py
    - extract_frames_from_video, get_video_info, create_video_from_frames: video utilities
- data/
  - Input assets. Global map (global_map.png) and optional crops/ directory for frame images.
- results/
  - Output artifacts: match debug images, progressive trajectory frames, final trajectory_map.png, and trajectory_video.avi.
- requirements.txt
  - Dependencies: numpy, opencv-python, opencv-contrib-python, torch, torchvision, kornia, lightglue, h5py

Conventions and configuration:
- Paths: uses relative paths for data/ and results/, but some absolute Windows paths exist in main.py for video_path and tiles_dir; prefer environment/config-driven values.
- Environment: GRID_SIZE is read via env var in main.py. Extend this pattern for other tunables.
- Tile filenames: expected as tile_z{z}_x{x}_y{y}{ext}. Keep zoom level (z) consistent wherever tiles are read.

## 3. Test Strategy
Current repo has no tests. Adopt Pytest with a tests/ directory and aim for fast, deterministic tests.

Recommended structure and scope:
- Unit tests (fast, deterministic):
  - tests/test_tile_cache.py: TileCache eviction/LRU order; get/set correctness
  - tests/test_stitching.py: stitch_grid with synthetic same-sized tiles; load_and_stitch_grid/rect using a mocked load_tile
  - tests/test_math.py: update_center_from_match conversions from pixel offsets to tile steps (edge rounding cases)
  - tests/test_sorting.py: numerical_sort behavior
  - tests/test_matcher.py: SuperPointLightGlueMatcher initialization, feature extraction, caching
- Component tests (use small images):
  - Minimal 2×2 or 3×3 tile set stitched into a mosaic, verify pixel dimensions and indexing
  - Small synthetic frame and mosaic where a known homography exists; confirm center point computation
  - Compare SuperPoint vs SIFT on same test images
- Integration tests (optional, slower):
  - End-to-end generate_trajectory_map on a tiny video (5–10 frames). Mock cv2.VideoCapture to return in-memory frames
  - generate_dynamic_tile_matching using a tiny tile set and high frame_skip
  - Test both use_superpoint=True and use_superpoint=False paths

Guidelines:
- Use fixtures to create temporary directories for results and synthetic images
- Mock I/O-heavy OpenCV calls when possible (e.g., VideoCapture, imread) or use tiny images to keep tests fast
- For GPU tests, check torch.cuda.is_available() and skip if unavailable
- Establish thresholds instead of exact pixel equality for geometric outputs; verify monotonic path growth and reasonable bounds
- Track coverage and focus on src/; target ≥80% where practical

## 4. Code Style
- Python style
  - Follow PEP 8. Use snake_case for functions and variables; PascalCase for classes
  - Prefer type hints in all public functions. Extend typing to all modules
  - Add docstrings (Google- or NumPy-style) describing args, returns, and side effects
  - Prefer pathlib.Path for filesystem paths; avoid hardcoded absolute OS-specific paths
  - Keep functions small and single-responsibility; separate I/O from computation when feasible
- OpenCV/Numpy/PyTorch specifics
  - OpenCV images are BGR. Be explicit when converting to/from grayscale and when drawing
  - Release resources: always cap.release() and writer.release()
  - Prefer cv2.IMREAD_COLOR explicitly and check for None before using images
  - Use integer pixel coordinates for drawing; center points are (x, y) tuples
  - Use torch.no_grad() context for all inference operations
  - Load models in eval() mode for inference
  - Handle both CPU and GPU gracefully with device auto-detection
- Logging & errors
  - Use logging module (INFO for progress, DEBUG for per-frame details, WARNING/ERROR for issues)
  - Validate inputs early (existence of files/dirs; non-empty frames). Fail fast with actionable messages or exceptions where appropriate; continue gracefully when skipping frames
  - Log device selection (CPU/GPU) at initialization
- Configuration
  - Centralize tunables (e.g., frame_skip, match_threshold, min_good_matches, grid_size, tile zoom, ext, output paths) via argparse or env vars loaded into a config object
  - Avoid global constants in function bodies; pass parameters explicitly or from config

## 5. Common Patterns
- Feature detection & matching
  - **Primary (Recommended)**: SuperPoint + LightGlue for learned feature detection and matching
    - SuperPoint: Self-supervised keypoint detector with 256-dim descriptors
    - LightGlue: Learned matcher with adaptive pruning and confidence scores
    - Advantages: Higher repeatability (~80-90% vs ~60%), better robustness to lighting/scale/rotation, GPU acceleration, fewer false positives
    - Configuration: max_keypoints=2048, match_threshold=0.2, detection_threshold=0.005, resize_max=1024
    - Feature caching: Cache map features with cache_key='global_map' to avoid recomputation across frames
    - Confidence-weighted homography: Combine geometric RANSAC inliers with match confidence scores for better accuracy
    - Device auto-detection: Automatically uses CUDA if available, falls back to CPU
  - **Fallback (Legacy)**: SIFT with BFMatcher + Lowe's ratio test (0.75) for backward compatibility
  - All trajectory functions support use_superpoint parameter (default=True) to toggle between methods
  - When computing homography: cv2.findHomography with RANSAC, ransacReprojThreshold=5.0; handle None H
  - Frame-to-map location via perspective transform of frame corners and averaging to obtain the center
  - Ensure min_good_matches threshold (>=10 by default) before homography
- Tiled map handling
  - LRU tile cache to bound memory; key by (x, y)
  - Grid stitching via NumPy concatenate over rows and columns
  - Dynamic center update translates pixel offset to tile index delta based on tile_px and grid center
  - Don't cache stitched tiles in SuperPoint matcher (they change per frame)
- Rendering
  - Draw trajectory lines in blue (255, 0, 0), start point green, end point red. Keep consistent thickness and radius
  - Generate progressive frames for video by drawing cumulatively
- GPU acceleration
  - Auto-detect CUDA availability; fallback to CPU if unavailable
  - Use torch.no_grad() context for inference to save memory
  - Models loaded in eval() mode for inference
  - Log device selection for transparency

## 6. Do's and Don'ts
- Do
  - **Use SuperPoint + LightGlue by default** for best accuracy (use_superpoint=True)
  - Keep zoom level (z) and tile filename convention consistent across grid and full-extent stitching
  - Use environment variables or CLI args for paths (input video, tiles, map) and parameters (grid_size, frame_skip, min_good_matches, match_threshold, tile zoom, codec, fps)
  - Ensure results/ exists before writing. Include informative filenames with frame counters
  - Check for None on all imread/VideoCapture reads; skip safely if features are insufficient
  - Document any non-default OpenCV/PyTorch parameters and rationale
  - Add type hints and docstrings for new functions
  - Cache global map features when processing multiple frames against the same map
  - Log match statistics (num_matches, num_inliers) for debugging
- Don't
  - Don't hardcode absolute machine-specific paths; avoid OS-specific separators
  - Don't assume tile availability; handle missing tiles by skipping or reducing grid size where appropriate
  - Don't change coordinate ordering (use (x, y) consistently) or switch to RGB without clear conversion
  - Don't leak resources: always release VideoCapture/VideoWriter
  - Don't silently swallow critical errors; at minimum log with context
  - Don't cache stitched tile features (they change dynamically)
  - Don't assume GPU availability; always provide CPU fallback

## 7. Tools & Dependencies
- Key libraries
  - **torch**: PyTorch for deep learning models (SuperPoint, LightGlue)
  - **torchvision**: Vision utilities and transforms
  - **kornia**: Computer vision library with SuperPoint implementation
  - **lightglue**: Learned feature matcher (install from GitHub)
  - numpy: array operations and concatenation for stitching
  - opencv-python: core computer vision and I/O
  - opencv-contrib-python: SIFT (non-free) for legacy fallback
  - h5py: HDF5 file format support
- Setup
  - Python 3.8+ recommended (3.9+ for best PyTorch compatibility)
  - Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
  - For GPU acceleration: Install CUDA-enabled PyTorch
    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    ```
  - Optional: ffmpeg installed on system for broader codec support when writing videos
- Running
  - Ensure data/global_map.png exists for global map modes
  - For crops mode: place ordered .png frames in data/crops/
  - For video modes: provide a valid input video path
  - Outputs go to results/ by default
  - Check logs for device selection (CPU/GPU) and match statistics
- Configuration suggestions (not all currently wired in code, but recommended for future CLI/env support):
  - MAP_PATH, RESULTS_DIR, INPUT_VIDEO_PATH, CROPS_DIR
  - TILES_DIR, INITIAL_TILE_X, INITIAL_TILE_Y
  - TILE_ZOOM (z), TILE_EXT (default .png), GRID_SIZE (odd), FRAME_SKIP
  - USE_SUPERPOINT (default True), MAX_KEYPOINTS (default 2048), MATCH_THRESHOLD (default 0.2)
  - MIN_GOOD_MATCHES (default 10), RANSAC_THRESH (default 5.0)
  - CODEC (XVID/MJPG/H264), FPS
  - DEVICE (cuda/cpu/auto)

## 8. Other Notes
- SuperPoint + LightGlue advantages
  - **30-50% more correct matches** compared to SIFT in challenging conditions
  - **Better robustness** to lighting changes, scale variations, rotation, and repetitive patterns
  - **GPU acceleration** provides 5-10x speedup over CPU SIFT on supported hardware
  - **Learned matching** reduces false positives significantly
  - **Confidence scores** enable better quality filtering and weighted homography
- Tile zoom consistency
  - load_and_stitch_grid default z=18 while load_and_stitch_rect uses z=19; keep these aligned with your dataset. If your files are named tile_z19_x{...}_y{...}.png, ensure both calls use z=19
- Tile filename pattern
  - Current loader expects: tile_z{z}_x{x}_y{y}{ext}. Unify the dataset and comments accordingly
- Performance
  - Increase frame_skip for long/high-fps videos. Tune grid_size and tile_cache max_items to balance accuracy, coverage, and memory
  - SuperPoint resize_max=1024 provides good speed/accuracy tradeoff; increase for higher accuracy, decrease for speed
  - Use GPU when available for 5-10x speedup
  - Feature caching dramatically improves speed when matching multiple frames against the same map
- Determinism & reproducibility
  - Matching has inherent randomness with RANSAC; results may vary slightly between runs
  - SuperPoint/LightGlue are deterministic given the same input and device
  - Prefer thresholds and qualitative assertions in tests
- Video encoding portability
  - XVID/AVI may not be available on all systems. Make codec configurable and consider MJPG or MP4/H264 where supported
- Color spaces
  - OpenCV uses BGR. SuperPoint works on grayscale (automatically converted)
  - Ensure correct conversion if integrating with libraries expecting RGB
- Legal/data sourcing
  - Ensure you have rights to the satellite tiles used. Respect provider terms when distributing datasets
- Large images & memory
  - Full-extent mosaics can be large. Consider progressive saving and avoiding keeping multiple large arrays in memory
  - GPU memory is limited; resize_max parameter helps manage memory usage
  - Clear feature cache when switching between different map regions
- Migration from SIFT
  - All functions maintain backward compatibility with use_superpoint=False
  - Default is use_superpoint=True for best accuracy
  - SIFT fallback available if PyTorch/LightGlue unavailable (will log warning)

This document is intended for developers and LLMs generating code in this repository. Adhere to the conventions above to maintain consistency, reliability, and portability.
