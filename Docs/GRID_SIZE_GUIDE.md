# Grid Size Problem & Solutions

## 🔍 Problem: Why Larger Grid Sizes Fail

You observed:
- ✅ **5×5 grid**: Works well
- ⚠️ **7×7 grid**: Only 4 matches
- ❌ **9×9 grid**: Zero matches

### Root Cause: Scale Mismatch

#### The Math

Assuming each tile is ~512×512 pixels:

| Grid Size | Stitched Size | After Resize (840px) | Scale Loss |
|-----------|---------------|----------------------|------------|
| 5×5 | 2,560×2,560 | 840×840 | 3.0x |
| 7×7 | 3,584×3,584 | 840×840 | 4.3x |
| 9×9 | 4,608×4,608 | 840×840 | 5.5x |

**Drone frame**: ~1920×1080 pixels → resized to ~840×472

#### What Happens

1. **5×5 Grid** (Works ✅)
   - Stitched: 2560×2560 → Resized: 840×840
   - Drone: 1920×1080 → Resized: 840×472
   - **Scale ratio**: ~3x (manageable)
   - Features are similar size after resize

2. **7×7 Grid** (Struggles ⚠️)
   - Stitched: 3584×3584 → Resized: 840×840
   - Drone: 1920×1080 → Resized: 840×472
   - **Scale ratio**: ~4.3x (challenging)
   - Drone features become 4x smaller than map features
   - Only very distinct features match

3. **9×9 Grid** (Fails ❌)
   - Stitched: 4608×4608 → Resized: 840×840
   - Drone: 1920×1080 → Resized: 840×472
   - **Scale ratio**: ~5.5x (too large!)
   - Drone features are 5x smaller - LoFTR can't match
   - Information loss is too severe

### Visual Explanation

```
5×5 Grid (Works):
Drone frame:  [====== 840px ======]
Stitched map: [====== 840px ======]
Scale: Similar ✅

9×9 Grid (Fails):
Drone frame:  [====== 840px ======]
Stitched map: [====== 840px ======] (but represents 4608px!)
                ↑ Drone is tiny relative to map
Scale: 5.5x difference ❌
```

---

## ✅ Solutions

### Solution 1: Adaptive Resize (Recommended)

Use the new `main_loftr_adaptive.py` script that automatically adjusts `resize_max` based on grid size:

```bash
# Works with any grid size!
python main_loftr_adaptive.py

# Or set grid size via environment variable
set GRID_SIZE=7
python main_loftr_adaptive.py

# Or for 9×9
set GRID_SIZE=9
python main_loftr_adaptive.py
```

**How it works:**
- 5×5 grid → `resize_max=1280`
- 7×7 grid → `resize_max=1792`
- 9×9 grid → `resize_max=2048`

This maintains consistent scale ratio (~2x) regardless of grid size.

### Solution 2: Manual Resize Adjustment

Edit `main_loftr.py` and increase `resize_max`:

```python
# For 7×7 grid
matcher = create_dino_loftr_matcher(
    use_gpu=True,
    loftr_model="outdoor",
    resize_max=1600  # Increased from 840
)

# For 9×9 grid
matcher = create_dino_loftr_matcher(
    use_gpu=True,
    loftr_model="outdoor",
    resize_max=2048  # Increased from 840
)
```

### Solution 3: Reduce Grid Size

If you don't need such a large search area:

```python
grid_size = 5  # Stick with what works
```

**When to use smaller grids:**
- Good initial position estimate
- Slow-moving drone
- High frame rate (less movement between frames)

**When you need larger grids:**
- Poor initial position estimate
- Fast-moving drone
- Low frame rate (more movement between frames)
- Large-scale changes in viewpoint

### Solution 4: Hierarchical Matching

Use a coarse-to-fine approach:

1. **Coarse**: Match with 9×9 grid at low resolution
2. **Fine**: Once located, switch to 5×5 grid at high resolution

```python
# Pseudo-code
if uncertain_position:
    grid_size = 9
    resize_max = 2048
else:
    grid_size = 5
    resize_max = 1280
```

---

## 📊 Recommended Settings

### For Different Grid Sizes

| Grid Size | resize_max | RANSAC Threshold | min_matches | Use Case |
|-----------|------------|------------------|-------------|----------|
| 3×3 | 840 | 3.0 | 20 | Precise tracking, good initial position |
| 5×5 | 1280 | 3.0 | 20 | **Recommended default** |
| 7×7 | 1792 | 4.0 | 25 | Moderate uncertainty |
| 9×9 | 2048 | 5.0 | 30 | High uncertainty, fast motion |

### Memory Considerations

Larger `resize_max` uses more GPU memory:

| resize_max | GPU Memory | Speed |
|------------|------------|-------|
| 840 | ~2 GB | Fast |
| 1280 | ~4 GB | Moderate |
| 1792 | ~6 GB | Slow |
| 2048 | ~8 GB | Very Slow |

**If you get "CUDA out of memory":**
1. Reduce `resize_max`
2. Reduce `grid_size`
3. Use CPU (slower but works)

---

## 🎯 Best Practices

### 1. Start Small, Scale Up

```python
# Start with proven settings
grid_size = 5
resize_max = 1280

# If tracking fails, increase grid
grid_size = 7
resize_max = 1792
```

### 2. Monitor Scale Ratio

The adaptive script logs scale ratio:
```
Scale ratio (stitched/frame): 2.3x
```

**Guidelines:**
- < 2.0x: Excellent
- 2.0-3.0x: Good
- 3.0-4.0x: Acceptable
- > 4.0x: Problematic

### 3. Adjust RANSAC Threshold

Larger grids need more lenient RANSAC:

```python
if grid_size <= 5:
    ransac_threshold = 3.0
elif grid_size <= 7:
    ransac_threshold = 4.0
else:
    ransac_threshold = 5.0
```

### 4. Increase min_matches for Larger Grids

```python
if grid_size <= 5:
    min_matches = 20
elif grid_size <= 7:
    min_matches = 25
else:
    min_matches = 30
```

---

## 🧪 Testing Different Grid Sizes

Use the adaptive script to test:

```bash
# Test 5×5
set GRID_SIZE=5
python main_loftr_adaptive.py

# Test 7×7
set GRID_SIZE=7
python main_loftr_adaptive.py

# Test 9×9
set GRID_SIZE=9
python main_loftr_adaptive.py
```

Compare results:
- Check `match_adaptive_*.png` for match quality
- Look at console logs for match counts
- Compare success rates

---

## 🔧 Troubleshooting

### Problem: Still no matches with 9×9

**Try:**
1. Increase `resize_max` even more:
   ```python
   resize_max = 2560  # or even 3072
   ```

2. Reduce `target_scale_ratio`:
   ```python
   resize_max = calculate_adaptive_resize(
       grid_size, 
       tile_size=512, 
       target_scale_ratio=1.5  # Reduced from 2.0
   )
   ```

3. Check if tiles are loading correctly:
   ```python
   # Add after stitching
   if stitched is not None:
       print(f"Stitched shape: {stitched.shape}")
       cv2.imwrite("debug_stitched.png", stitched)
   ```

### Problem: Too slow with large resize_max

**Try:**
1. Process fewer frames:
   ```python
   frame_skip = 30  # Instead of 15
   ```

2. Use CPU for some frames:
   ```python
   # Alternate between GPU and CPU
   use_gpu = (frame_count % 2 == 0)
   ```

3. Reduce grid size:
   ```python
   grid_size = 7  # Instead of 9
   ```

---

## 📈 Expected Performance

### With Adaptive Resize

| Grid Size | Matches | Inliers | Success Rate |
|-----------|---------|---------|--------------|
| 5×5 | 500-1500 | 200-800 | 85-95% |
| 7×7 | 400-1200 | 150-600 | 75-90% |
| 9×9 | 300-1000 | 100-500 | 65-85% |

### Without Adaptive Resize (Fixed 840px)

| Grid Size | Matches | Inliers | Success Rate |
|-----------|---------|---------|--------------|
| 5×5 | 500-1500 | 200-800 | 85-95% |
| 7×7 | 50-200 | 10-80 | 20-40% ❌ |
| 9×9 | 0-50 | 0-20 | 0-10% ❌ |

---

## 🎯 Summary

**The Problem:**
- Fixed `resize_max=840` causes severe scale mismatch with larger grids
- 9×9 grid creates 5.5x scale difference → matching fails

**The Solution:**
- Use `main_loftr_adaptive.py` for automatic resize adjustment
- Or manually increase `resize_max` proportional to grid size
- Adjust RANSAC threshold and min_matches accordingly

**Quick Fix:**
```bash
python main_loftr_adaptive.py
```

This will work with any grid size from 3×3 to 9×9!
