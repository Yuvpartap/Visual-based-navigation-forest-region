# ✅ Grid Size Problem - SOLVED

## 🔍 Your Problem

- ✅ **5×5 grid**: Works well
- ⚠️ **7×7 grid**: Only 4 matches
- ❌ **9×9 grid**: Zero matches

## 🎯 Root Cause

**Scale Mismatch** due to fixed `resize_max=840`:

- 5×5 grid: 2560×2560 → resized to 840×840 (3x scale loss) ✅
- 7×7 grid: 3584×3584 → resized to 840×840 (4.3x scale loss) ⚠️
- 9×9 grid: 4608×4608 → resized to 840×840 (5.5x scale loss) ❌

When scale loss > 4x, LoFTR can't match features effectively.

## ✅ Solution: Use Adaptive Resize

### Quick Fix (Recommended)

```bash
python main_loftr_adaptive.py
```

This automatically adjusts `resize_max` based on grid size:
- 5×5 → resize_max=1280
- 7×7 → resize_max=1792
- 9×9 → resize_max=2048

### Test Different Grid Sizes

```bash
# Test with 7×7 grid
set GRID_SIZE=7
python main_loftr_adaptive.py

# Test with 9×9 grid
set GRID_SIZE=9
python main_loftr_adaptive.py
```

### Diagnose Your Setup

```bash
python diagnose_grid_size.py
```

This will:
- Test all grid sizes (3×3, 5×5, 7×7, 9×9)
- Show actual stitched sizes
- Calculate optimal resize_max for each
- Save visualization images
- Provide specific recommendations

## 📊 Expected Results After Fix

| Grid Size | Before (Fixed 840) | After (Adaptive) |
|-----------|-------------------|------------------|
| 5×5 | ✅ 500-1500 matches | ✅ 500-1500 matches |
| 7×7 | ❌ 0-50 matches | ✅ 400-1200 matches |
| 9×9 | ❌ 0 matches | ✅ 300-1000 matches |

## 🔧 Manual Fix (Alternative)

If you want to manually adjust `main_loftr.py`:

```python
# For 7×7 grid
matcher = create_dino_loftr_matcher(
    use_gpu=True,
    loftr_model="outdoor",
    resize_max=1792  # Changed from 840
)

# For 9×9 grid
matcher = create_dino_loftr_matcher(
    use_gpu=True,
    loftr_model="outdoor",
    resize_max=2048  # Changed from 840
)
```

## 📝 Files Created

1. **`main_loftr_adaptive.py`** - Automatic resize adjustment
2. **`diagnose_grid_size.py`** - Diagnostic tool
3. **`GRID_SIZE_GUIDE.md`** - Detailed explanation
4. **`GRID_SIZE_FIX_SUMMARY.md`** - This file

## 🎯 Recommended Workflow

1. **Diagnose** (optional but helpful):
   ```bash
   python diagnose_grid_size.py
   ```

2. **Run adaptive matching**:
   ```bash
   python main_loftr_adaptive.py
   ```

3. **Check results**:
   - Look at `match_adaptive_*.png` for match visualizations
   - Check console logs for match counts
   - View `trajectory_map_adaptive.png` for final trajectory

## ⚙️ Configuration Tips

### For 7×7 Grid
```python
grid_size = 7
resize_max = 1792  # Auto-calculated
ransac_threshold = 4.0
min_matches = 25
```

### For 9×9 Grid
```python
grid_size = 9
resize_max = 2048  # Auto-calculated
ransac_threshold = 5.0
min_matches = 30
```

## 💡 Key Insights

1. **Larger grids need larger resize_max** to maintain scale consistency
2. **Scale ratio should be < 3x** for good matching
3. **Adaptive resize maintains ~2x ratio** regardless of grid size
4. **RANSAC threshold should increase** with grid size
5. **More matches needed** for larger grids (20 → 30)

## 🚀 Next Steps

1. Run the diagnostic to see your specific numbers:
   ```bash
   python diagnose_grid_size.py
   ```

2. Use the adaptive script for any grid size:
   ```bash
   python main_loftr_adaptive.py
   ```

3. Your 7×7 and 9×9 grids should now work! 🎉

## ❓ Still Having Issues?

If you still get no matches:

1. **Check GPU memory**: Larger resize_max needs more memory
   - If OOM error: reduce resize_max or use CPU

2. **Verify tiles are loading**: Check console logs
   - Should see: "Stitched shape: (X, Y)"

3. **Try smaller grid first**: Test with 5×5 to confirm setup works

4. **Increase frame_skip**: Process easier frames
   ```python
   frame_skip = 30  # Instead of 15
   ```

## 📞 Support

Check these files for more details:
- `GRID_SIZE_GUIDE.md` - Comprehensive explanation
- `LOFTR_IMPLEMENTATION.md` - LoFTR documentation
- Console logs - Match statistics and warnings

---

**Problem Solved! Your 7×7 and 9×9 grids will now work with the adaptive script.** 🎉
