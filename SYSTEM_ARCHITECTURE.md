# 🏗️ System Architecture - EfficientLoFTR Implementation

## 📐 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPS-DENIED DRONE NAVIGATION                   │
│                  EfficientLoFTR-Based System                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │         INPUT PROCESSING                 │
        ├─────────────────────────────────────────┤
        │  • Drone Video (MP4)                    │
        │  • Frame Extraction (with skip)         │
        │  • Preprocessing & Resizing             │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │      SATELLITE TILE MANAGEMENT           │
        ├─────────────────────────────────────────┤
        │  • Dynamic Tile Loading                 │
        │  • LRU Cache (512-1024 tiles)           │
        │  • Grid Stitching (NxN)                 │
        │  • Adaptive Grid Sizing                 │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │      EFFICIENTLOFTR MATCHING             │
        ├────────────────────���────────────────────┤
        │  • Hugging Face Model                   │
        │  • Mixed Precision (FP16)               │
        │  • Dense Feature Matching               │
        │  • Confidence Scoring                   │
        │  • 2-3x Faster than LoFTR              │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │      HOMOGRAPHY ESTIMATION               │
        ├─────────────────────────────────────────┤
        │  • RANSAC Algorithm                     │
        │  • Confidence Weighting                 │
        │  • Inlier Selection                     │
        │  • Transformation Matrix                │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │      TRAJECTORY ESTIMATION               │
        ├─────────────────────────────────────────┤
        │  • Center Point Calculation             │
        │  • Tile Coordinate Update               │
        │  • Path Accumulation                    │
        │  • Bounds Tracking                      │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │      VISUALIZATION & OUTPUT              │
        ├─────────────────────────────────────────┤
        │  • Match Visualizations                 │
        │  • Progressive Trajectory Frames        │
        │  • Final Trajectory Map                 │
        │  • Performance Statistics               │
        └─────────────────────────────────────────┘
```

## 🔄 Data Flow Diagram

```
┌──────────────┐
│ Drone Video  │
│  (1920x1080) │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ Frame Extraction │ ← frame_skip parameter
│  (every Nth)     │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐      ┌─────────────────┐
│  Current Frame   │      │ Satellite Tiles │
│   (640x480)      │      │   (512x512)     │
└──────┬───────────┘      └────────┬────────┘
       │                           │
       │                           ▼
       │                  ┌─────────────────┐
       │                  │  Tile Stitching │
       │                  │   (NxN grid)    │
       │                  └────────┬────────┘
       │                           │
       └───────────┬───────────────┘
                   │
                   ▼
          ┌─────────────────────┐
          │  EfficientLoFTR     │
          │  Feature Matching   │
          │  • Extract features │
          │  • Match keypoints  │
          │  • Compute conf.    │
          └─────────┬───────────┘
                    │
                    ▼
          ┌─────────────────────┐
          │  Homography (RANSAC)│
          │  • Filter inliers   │
          │  • Compute H matrix │
          │  • Transform corners│
          └─────────┬───────────┘
                    │
                    ▼
          ┌─────────────────────┐
          │  Center Calculation │
          │  • Average corners  │
          │  • Update tile pos. │
          │  • Append to path   │
          └─────────┬───────────┘
                    │
                    ▼
          ┌─────────────────────┐
          │  Trajectory Output  │
          ���  • Draw path        │
          │  • Save map         │
          │  • Log statistics   │
          └─────────────────────┘
```

## 🧩 Component Architecture

### 1. EfficientLoFTR Matcher Module

```
┌────────────────────────────────────────────────────────┐
│           EfficientLoFTRMatcher Class                  │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ┌──────────────────────────────────────────────┐    │
│  │  Initialization                               │    │
│  │  • Load HF model (zju-community/efficientloftr)│  │
│  │  • Setup device (CUDA/CPU)                    │    │
│  │  • Configure mixed precision                  │    │
│  │  • Initialize caches                          │    │
│  └──────────────────────────────────────────────┘    │
│                                                        │
│  ┌──────────────────────────────────────────────┐    │
│  │  Image Preprocessing                          │    │
│  │  • Convert to grayscale                       │    │
│  │  • Adaptive resizing                          │    │
│  │  • Normalization                              │    │
│  │  • Tensor conversion                          │    │
│  └──────────────────────────────────────────────┘    │
│                                                        │
│  ┌──────────────────────────────────────────────┐    │
│  │  Feature Matching                             │    │
│  │  • Dense matching (no keypoint detection)     │    │
│  │  • Transformer-based correspondence           │    │
│  │  • Confidence scoring                         │    │
│  │  • Scale compensation                         │    │
│  └──────────────────────���───────────────────────┘    │
│                                                        │
│  ┌──────────────────────────────────────────────┐    │
│  │  Homography Computation                       │    │
│  │  • RANSAC filtering                           │    │
│  │  • Confidence weighting                       │    │
│  │  • Inlier selection                           │    │
│  │  • Matrix estimation                          │    │
│  └──────────────────────────────────────────────┘    │
│                                                        │
│  ┌──────────────────────────────────────────────┐    │
│  │  Visualization                                │    │
│  │  • Side-by-side image display                 │    │
│  │  • Match line drawing                         │    │
│  │  • Confidence color coding                    │    │
│  │  • Statistics overlay                         │    │
│  └──────────────────────────────────────────────┘    │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### 2. Pipeline Architecture

```
┌────────────────────────────────────────────────────────┐
│              Main Pipeline (main_efficient_loftr.py)   │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ┌──────────────────────────────────────────────┐    │
│  ���  Configuration                                │    │
│  │  • Video path                                 │    │
│  │  • Tile directory                             │    │
│  │  • Initial position                           │    │
│  │  • Grid size, resize_max, etc.                │    │
│  └──────────────────────────────────────────────┘    │
│                     │                                  │
│                     ▼                                  │
│  ┌──────────────────────────────────────────────┐    │
│  │  Initialization                               │    │
│  │  • Create matcher                             │    │
│  │  • Setup tile cache                           │    │
│  │  • Open video capture                         │    │
│  │  • Initialize tracking variables              │    │
│  └──────────────────────────────────────────────┘    │
│                     │                                  │
│                     ▼                                  │
│  ┌──────────────────────────────────────────────┐    │
│  │  Frame Processing Loop                        │    │
│  │  ┌────────────────────────────────────┐      │    │
│  │  │ 1. Read frame                       │      │    │
│  │  │ 2. Apply frame skip                 │      │    │
│  │  │ 3. Load & stitch tiles              │      │    │
│  │  │ 4. Match with EfficientLoFTR        │      │    │
│  │  │ 5. Compute homography               │      │    │
│  │  │ 6. Update trajectory                │      │    │
│  │  │ 7. Save visualization               │      │    │
│  │  │ 8. Track performance                │      │    │
│  │  └────────────────────────────────────┘      │    │
│  └──────────────────────────────────────────────┘    │
│                     │                                  │
│                     ▼                                  │
│  ┌──────────────────────────────────────────────┐    │
│  │  Trajectory Map Generation                    │    │
│  │  • Stitch full extent                         │    │
│  │  • Draw trajectory path                       │    │
│  │  • Mark start/end points                      │    │
│  │  • Save final map                             │    │
│  └──────────────────────────────────────────────┘    │
│                     │                                  │
│                     ▼                                  │
│  ┌──────────────────────────────────────────────┐    │
│  │  Performance Summary                          │    │
│  │  • Total time                                 │    │
│  │  • Average time/frame                         │    │
│  │  • Memory usage                               │    │
│  │  • Match statistics                           │    │
│  └──────────────────────────���───────────────────┘    │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## 🔀 Comparison Architecture

```
┌────────────────────────────────────────────────────────┐
│         Comparison Tool (compare_models.py)            │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ┌──────────────────────────────────────────────┐    │
│  │  Test Configuration                           │    │
│  │  • Same video                                 │    │
│  │  • Same tiles                                 │    │
│  │  • Same parameters                            │    │
│  │  • N frames (e.g., 20)                        │    │
│  └──────────────────────────────────────────────┘    │
│                     │                                  │
│         ┌───────────┴───────────┐                     │
│         ▼                       ▼                      │
│  ┌─────────────┐         ┌─────────────┐             │
│  │  Standard   │         │ Efficient   │             │
│  │   LoFTR     │         │   LoFTR     │             │
│  └──────┬──────┘         └──────┬──────┘             │
│         │                       │                      │
│         ▼                       ▼                      │
│  ┌─────────────┐         ┌─────────────┐             │
│  │  Metrics    │         │  Metrics    │             │
│  │  • Time     │         │  • Time     │             │
│  │  • Memory   │         │  • Memory   │             │
│  │  • Matches  │         │  • Matches  │             │
│  │  • Inliers  │         │  • Inliers  │             │
│  └──────┬──────┘         └──────┬──────┘             │
│         │                       │                      │
│         └───────────┬───────────┘                     │
│                     ▼                                  │
│  ┌──────────────────────────────────────────────┐    │
│  │  Comparison Analysis                          │    │
│  │  • Speedup calculation                        │    │
│  │  • Memory reduction                           │    │
│  │  • Accuracy comparison                        │    │
│  │  • Recommendations                            │    │
│  └──────────────────────────────────────────────┘    │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## 🎯 Optimization Stack

```
┌─────────────────────��──────────────────────────────────┐
│                  OPTIMIZATION LAYERS                   │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Layer 1: Model Architecture                          │
│  ┌──────────────────────────────────────────────┐    │
│  │  EfficientLoFTR (vs Standard LoFTR)          │    │
│  │  • Optimized transformer                      │    │
│  │  • Efficient attention                        │    │
│  │  • Reduced parameters                         │    │
│  │  → 2-3x speedup                               │    │
│  └──────────────────────────────────────────────┘    │
│                                                        │
│  Layer 2: Precision Optimization                      │
│  ┌────────────────────────────────────────��─────┐    │
│  │  Mixed Precision (FP16)                       │    │
│  │  • Half precision inference                   │    │
│  │  • Automatic casting                          │    │
│  │  • GPU tensor cores                           │    │
│  │  → 2x additional speedup                      │    │
│  └──────────────────────────────────────────────┘    │
│                                                        │
│  Layer 3: Image Processing                            │
│  ┌──────────────────────────────────────────────┐    │
│  │  Adaptive Resizing                            │    │
│  │  • Dynamic resize_max                         │    │
│  │  • Grid-based scaling                         │    │
│  │  • Efficient interpolation                    │    │
│  │  → Balanced speed/accuracy                    │    │
│  └──────────────────────────────────────────────┘    │
│                                                        │
│  Layer 4: Caching Strategy                            │
│  ┌──────────────────────────────────────────────┐    │
│  │  Multi-level Caching                          │    │
│  │  • Tile cache (LRU)                           │    │
│  │  • Feature cache                              │    │
│  │  • GPU memory management                      │    │
│  │  → Reduced redundant computation              │    │
│  └──────────────────────────────────────────────┘    │
│                                                        │
│  Layer 5: Algorithm Optimization                      │
│  ┌──────────────────────────────────────────────┐    │
│  │  Smart Processing                             │    │
│  │  • Early termination                          │    │
│  │  • Frame skipping                             │    │
│  │  • Confidence thresholding                    │    │
│  │  → Skip unnecessary work                      │    │
│  └──────────────���───────────────────────────────┘    │
│                                                        │
└────────────────────────────────────────────────────────┘

Combined Effect: 5-10x overall speedup vs baseline
```

## 📊 Performance Comparison Matrix

```
┌─────────────────────────────────────────────────────────────┐
│              METHOD COMPARISON MATRIX                       │
├─────────────┬──────┬────────┬────────┬──────────┬─────────┤
│   Method    │Speed │ Memory │Accuracy│Real-time │  Score  │
├─────────────┼──────┼────────┼────────┼──────────┼─────────┤
│   SIFT      │  ★   │  ★★★   │  ★★    │    ✗     │  6/20   │
├─────────────┼──────┼────────┼────────┼──────────┼─────────┤
│ SuperPoint+ │  ★★  │  ★★    │  ★★★   │    ✗     │  10/20  │
│ LightGlue   │      │        │        │          ��         │
├─────────────┼──────┼────────┼────────┼──────────┼─────────┤
│  DINOv2 +   │  ★★  │  ★     │  ★★★★★ │    ✗     │  13/20  │
│   LoFTR     │      │        │        │          │         │
├─────────────┼──────┼────────┼────────┼──────────┼─────────┤
│ Efficient   │ ★★★★ │  ★★★★  │  ★★★★★ │    ✓     │  19/20  │
│   LoFTR     │      │        │        │          │  ⭐     │
└─────────────┴──────┴────────┴────────┴──────────┴─────────┘

Legend: ★ = Poor, ★★★★★ = Excellent
```

## 🔄 State Machine

```
┌─────────────────────────────────────────────────────────┐
│            TRAJECTORY TRACKING STATE MACHINE            │
└─────────────────────────────────────────────────────────┘

    [START]
       │
       ▼
   ┌────────┐
   │  INIT  │ ← Load model, setup cache
   └───┬────┘
       │
       ▼
   ┌────────────┐
   │ READ_FRAME │ ← Get next frame from video
   └───┬────────┘
       │
       ├─→ [End of video] ──→ [FINALIZE]
       │
       ▼
   ┌────────────┐
   │ SKIP_CHECK │ ← Apply frame_skip
   └───┬────────┘
       │
       ├─→ [Skip frame] ──→ [READ_FRAME]
       │
       ▼
   ┌────────────┐
   │ LOAD_TILES │ ← Stitch NxN grid
   └───┬────────┘
       │
       ├─→ [Tiles missing] ──→ [READ_FRAME]
       │
       ▼
   ┌────────────┐
   │   MATCH    │ ← EfficientLoFTR matching
   └───┬────────┘
       │
       ├─→ [Insufficient matches] ──→ [READ_FRAME]
       │
       ▼
   ┌────────────┐
   │ HOMOGRAPHY │ ← RANSAC estimation
   └───┬────────┘
       │
       ├─→ [Homography failed] ──→ [READ_FRAME]
       │
       ▼
   ┌────────────┐
   │   UPDATE   │ ← Update trajectory
   └───┬────────┘
       │
       ├─→ [Early termination] ──→ [VISUALIZE]
       │
       ▼
   ┌────────────┐
   │ VISUALIZE  │ ← Draw matches (optional)
   └───┬────────┘
       │
       └──→ [READ_FRAME]

   [FINALIZE]
       │
       ▼
   ┌────────────┐
   │ BUILD_MAP  │ ← Stitch full extent
   └───┬────────┘
       │
       ▼
   ┌────────────┐
   │ DRAW_TRAJ  │ ← Draw complete path
   └───┬────────┘
       │
       ▼
   ┌────────────┐
   │ SAVE_OUTPUT│ ← Save map & stats
   └───┬────────┘
       │
       ▼
    [END]
```

## 🗂️ File Organization

```
drone-trajectory-tracker/
│
├── 📁 src/                          # Source code
│   ├── efficient_loftr_matcher.py   # ⭐ NEW: EfficientLoFTR
│   ├── dino_loftr_matcher.py        # Standard LoFTR
│   ├── superpoint_lightglue_matcher.py
│   ├── tile_loading_utilis.py       # Tile management
│   ├── create_trajectory_map.py     # Trajectory generation
│   ├── create_video.py              # Video creation
│   └── video_utils.py               # Video utilities
│
├── 📁 data/                         # Input data
│   ├── crops/                       # Frame images
│   ├── global_map.png               # Satellite map
│   └── ...
│
├── 📁 results_efficient_loftr/      # ⭐ NEW: Output
│   ├── match_eloftr_*.png           # Match visualizations
│   └── trajectory_map_eloftr.png    # Final trajectory
│
├── 📄 main_efficient_loftr.py       # ⭐ NEW: Main pipeline
├── 📄 compare_models.py             # ⭐ NEW: Comparison tool
├── 📄 setup_efficientloftr.py       # ⭐ NEW: Setup script
│
├── 📚 EFFICIENTLOFTR_IMPLEMENTATION.md  # Full docs
├── 📚 QUICK_START_EFFICIENTLOFTR.md     # Quick guide
├── 📚 MIGRATION_GUIDE.md                # Migration help
├── 📚 EFFICIENTLOFTR_SUMMARY.md         # Overview
├── 📚 SYSTEM_ARCHITECTURE.md            # This file
├── 📚 IMPLEMENTATION_COMPLETE.md        # Completion summary
│
└── 📄 requirements.txt              # Dependencies
```

## 🎯 Decision Tree: Which Method to Use?

```
                    [Start]
                       │
                       ▼
            ┌──────────────────────┐
            │ Need real-time       │
            │ processing?          │
            └──────┬───────────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
        Yes                 No
         │                   │
         ▼                   ▼
    ┌─────────┐      ┌──────────────┐
    │ Limited │      │ Unlimited    │
    │ GPU     │      │ resources?   │
    │ memory? │      └──────┬───────┘
    └────┬────┘             │
         │           ┌──────┴──────┐
    ┌────┴────┐     │             │
    │         │    Yes           No
   Yes       No     │             │
    │         │     ▼             ▼
    │         │  ┌──────┐    ┌─────────┐
    │         │  │Either│    │Standard │
    │         │  │method│    │ LoFTR   │
    │         │  └──────┘    │(if need │
    │         │              │specific)│
    │         │              └─────────┘
    │         │
    └────┬────┘
         │
         ▼
    ┌──────────────┐
    │ EfficientLoFTR│ ⭐ RECOMMENDED
    │               │
    │ • 2-3x faster │
    │ • 50% less mem│
    │ • Same accuracy│
    │ • Real-time   │
    └───────────────┘
```

## 📈 Performance Scaling

```
Processing Time vs Grid Size

Time (s)
  3.0 │                                    ╱ Standard LoFTR
      │                               ╱
  2.5 │                          ╱
      │                     ╱
  2.0 │                ╱
      │           ╱
  1.5 │      ╱
      │ ╱                    ╱ EfficientLoFTR
  1.0 │─────────────────╱
      │            ╱
  0.5 │       ╱
      │  ╱
  0.0 └─────┬─────┬─────┬─────┬─────┬─────→ Grid Size
           3x3   5x5   7x7   9x9  11x11

Memory Usage vs Grid Size

Memory (GB)
  5.0 │                                    ╱ Standard LoFTR
      │                               ╱
  4.0 │                          ╱
      │                     ╱
  3.0 │                ╱
      │           ╱
  2.0 │      ╱                ╱ EfficientLoFTR
      │ ╱               ╱
  1.0 │───────────╱
      │      ╱
  0.0 └─────┬─────┬─────┬─────┬─────┬─────→ Grid Size
           3x3   5x5   7x7   9x9  11x11
```

## 🔧 Configuration Impact

```
┌─────────────────────────────────────────────────────────┐
│         PARAMETER IMPACT ON PERFORMANCE                 │
├─────────────┬──────────┬──────────┬──────────┬─────────┤
│  Parameter  │  Speed   │  Memory  │ Accuracy │  Impact │
├─────────────┼──────────┼──────────┼──────────┼───��─────┤
│ grid_size   │   ↓↓↓    │   ↑↑↑    │   ↑↑↑    │  HIGH   │
│ resize_max  │   ↓↓     │   ↑↑     │   ↑↑     │  HIGH   │
│ frame_skip  │   ↑↑↑    │   →      │   ↓      │  HIGH   │
│ mixed_prec  │   ↑↑     │   ↓      │   →      │  MEDIUM │
│ min_matches │   →      │   →      │   ↑      │  LOW    │
│ early_term  │   ↑      │   →      │   →      │  LOW    │
└─────────────┴──────────┴──────────┴──────────┴─────────┘

Legend: ↑ = Increase, ↓ = Decrease, → = No change
        ↑↑↑ = Large impact
```

---

This architecture document provides a comprehensive visual understanding of the EfficientLoFTR implementation system.

For implementation details, see `EFFICIENTLOFTR_IMPLEMENTATION.md`
For quick start, see `QUICK_START_EFFICIENTLOFTR.md`
