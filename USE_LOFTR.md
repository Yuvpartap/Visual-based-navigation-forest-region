# Using DINOv2 + LoFTR for Dense Forest Matching

## Quick Start

To use DINOv2 + LoFTR instead of SuperPoint + LightGlue, modify your `main.py`:

### Option 1: Modify main.py directly

Replace the `generate_dynamic_tile_matching` call with:

```python
from src.dino_loftr_matcher import create_dino_loftr_matcher

# Create LoFTR matcher
matcher = create_dino_loftr_matcher(use_gpu=True, loftr_model="outdoor", resize_max=840)

# Then use it in a loop similar to the existing code
```

### Option 2: Use the provided script

Run:
```bash
python main_loftr.py
```

## Installation

Install additional dependencies:

```bash
pip install kornia>=0.7.0
pip install kornia-rs>=0.1.0
```

For DINOv2 (optional, for global retrieval):
```bash
# DINOv2 will be automatically downloaded from torch.hub
```

## Advantages of LoFTR for Dense Forests

1. **Detector-free**: No keypoint detection needed - works in low-texture areas
2. **Dense matching**: Produces many more matches than sparse detectors
3. **Better for forests**: Handles repetitive patterns and low-texture regions
4. **Robust**: Works well with extreme viewpoint changes

## Configuration

- `loftr_model`: "outdoor" (recommended) or "indoor"
- `resize_max`: 840 (default) - balance between speed and accuracy
- `min_good_matches`: 20-50 (LoFTR produces many matches)
- `ransac_threshold`: 3.0 (lower than SuperPoint due to denser matches)

## Expected Performance

- **Matches**: 200-2000+ per frame (vs 50-200 for SuperPoint)
- **Inliers**: 100-1000+ (vs 20-100 for SuperPoint)
- **Speed**: Slower than SuperPoint but more accurate in challenging scenes
- **Memory**: Higher GPU memory usage

## Troubleshooting

If you get "CUDA out of memory":
- Reduce `resize_max` to 640 or 512
- Reduce `grid_size` to 3 or 5
- Process fewer frames (increase `frame_skip`)
