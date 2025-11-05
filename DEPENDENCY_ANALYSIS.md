# Dependency Analysis - src/ Folder Files

## Complete Dependency Chain

### Starting from: `clean_copy_refactor_loftr.py`

```
clean_copy_refactor_loftr.py (MAIN)
├── src/video_processor.py ✅ USED
│   ├── src/cuda_manager.py ✅ USED
│   ├── src/matcher_wrapper.py ✅ USED
│   │   ├── src/performance_tracker.py ✅ USED
│   │   └── src/cuda_manager.py ✅ USED (already counted)
│   ├── src/tracking_utils.py ✅ USED
│   │   └── src/rotation_utils.py ✅ USED
│   ├── src/rotation_utils.py ✅ USED (already counted)
│   ├── src/trajectory_visualizer.py ✅ USED
│   │   └── src/tile_loading_utilis.py ✅ USED
│   ├── src/tile_loading_utilis.py ✅ USED (already counted)
│   └── src/refactored_dino_loftr.py ✅ USED
├── src/video_utils.py ✅ USED
└── src/google_maps_plotter.py ✅ USED
```

## Files in src/ Folder

### ✅ REQUIRED FILES (11 files)

| File | Purpose | Used By |
|------|---------|---------|
| `cuda_manager.py` | GPU/CUDA management | video_processor, matcher_wrapper |
| `google_maps_plotter.py` | Google Maps HTML/KML generation | clean_copy_refactor_loftr (main) |
| `matcher_wrapper.py` | Optimized matcher wrapper | video_processor |
| `performance_tracker.py` | Performance monitoring | matcher_wrapper |
| `refactored_dino_loftr.py` | DINO+LoFTR matcher implementation | video_processor |
| `rotation_utils.py` | Rotation calibration & extraction | video_processor, tracking_utils |
| `tile_loading_utilis.py` | Tile loading & stitching | video_processor, trajectory_visualizer |
| `tracking_utils.py` | Kalman filter & rotation tracking | video_processor |
| `trajectory_visualizer.py` | Map & video generation | video_processor |
| `video_processor.py` | Main processing pipeline | clean_copy_refactor_loftr (main) |
| `video_utils.py` | Video info utilities | clean_copy_refactor_loftr (main) |

### ❌ UNUSED FILES (2 files)

| File | Purpose | Status |
|------|---------|--------|
| `create_trajectory_map.py` | Standalone trajectory map creator | ❌ NOT IMPORTED - Utility script |
| `create_video.py` | Standalone video creator | ❌ NOT IMPORTED - Utility script |

## Detailed Analysis

### UNUSED FILES EXPLANATION

#### 1. `create_trajectory_map.py`
- **Status**: Standalone utility script
- **Purpose**: Creates trajectory maps independently
- **Why unused**: Functionality is now integrated into `trajectory_visualizer.py`
- **Can be deleted**: YES - functionality duplicated in main pipeline

#### 2. `create_video.py`
- **Status**: Standalone utility script  
- **Purpose**: Creates trajectory videos independently
- **Why unused**: Functionality is now integrated into `trajectory_visualizer.py`
- **Can be deleted**: YES - functionality duplicated in main pipeline

### MISSING FILES FROM ORIGINAL LISTING

The following files were mentioned in earlier searches but are NOT present:
- `dino_loftr_matcher.py` - NOT FOUND (functionality in refactored_dino_loftr.py)
- `dino_se2loftr_matcher.py` - NOT FOUND
- `superpoint_lightglue_matcher.py` - NOT FOUND

These files either:
1. Were never created during modularization
2. Were in a different directory
3. Were removed/renamed

## Summary

### Files Count
- **Total files in src/**: 13 Python files
- **Required files**: 11 files ✅
- **Unused files**: 2 files ❌
- **Usage rate**: 84.6%

### Recommendation

**SAFE TO DELETE:**
1. `src/create_trajectory_map.py` - Standalone utility, functionality in trajectory_visualizer.py
2. `src/create_video.py` - Standalone utility, functionality in trajectory_visualizer.py

**KEEP ALL OTHER FILES** - They are part of the active processing pipeline.

### Dependency Depth

```
Level 0: clean_copy_refactor_loftr.py (main entry)
Level 1: video_processor.py, video_utils.py, google_maps_plotter.py
Level 2: cuda_manager.py, matcher_wrapper.py, tracking_utils.py, rotation_utils.py, 
         trajectory_visualizer.py, tile_loading_utilis.py, refactored_dino_loftr.py
Level 3: performance_tracker.py
```

Maximum dependency depth: **3 levels**

## Verification Commands

To verify no broken imports after deletion:
```bash
python -c "from src.video_processor import VideoProcessor; print('✓ All imports OK')"
```

To check for any references to deleted files:
```bash
# Search for create_trajectory_map references
grep -r "create_trajectory_map" *.py src/*.py

# Search for create_video references  
grep -r "create_video" *.py src/*.py
```
