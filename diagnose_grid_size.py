"""
Diagnostic script to analyze grid size impact on matching.

This script helps you understand why larger grid sizes fail by showing:
- Actual stitched image sizes
- Scale ratios
- Resize impact
- Recommended settings
"""

import cv2
import numpy as np
import os
import logging
from src.tile_loading_utilis import TileCache, load_and_stitch_grid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def diagnose_grid_size(tiles_dir, center_x, center_y, grid_sizes=[3, 5, 7, 9], ext=".png"):
    """
    Analyze different grid sizes and their impact on matching.
    
    Args:
        tiles_dir: Directory containing tiles
        center_x: Tile X coordinate
        center_y: Tile Y coordinate
        grid_sizes: List of grid sizes to test
        ext: Tile file extension
    """
    cache = TileCache(max_items=512)
    
    logger.info("=" * 70)
    logger.info("GRID SIZE DIAGNOSTIC ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"Tiles directory: {tiles_dir}")
    logger.info(f"Center position: ({center_x}, {center_y})")
    logger.info("=" * 70)
    
    results = []
    
    for grid_size in grid_sizes:
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing {grid_size}×{grid_size} Grid")
        logger.info(f"{'='*70}")
        
        # Load and stitch
        stitched = load_and_stitch_grid(
            tiles_dir=tiles_dir,
            center_x=center_x,
            center_y=center_y,
            grid_size=grid_size,
            ext=ext,
            cache=cache,
        )
        
        if stitched is None:
            logger.error(f"✗ Failed to load tiles for {grid_size}×{grid_size} grid")
            continue
        
        h, w = stitched.shape[:2]
        logger.info(f"✓ Stitched image size: {w}×{h} pixels")
        
        # Calculate tile size
        tile_size = h // grid_size
        logger.info(f"  Estimated tile size: {tile_size}×{tile_size} pixels")
        
        # Simulate different resize_max values
        resize_options = [840, 1280, 1792, 2048]
        
        logger.info(f"\n  Resize Analysis:")
        logger.info(f"  {'-'*66}")
        logger.info(f"  {'resize_max':<12} {'Resized To':<15} {'Scale Loss':<12} {'Status'}")
        logger.info(f"  {'-'*66}")
        
        best_resize = None
        
        for resize_max in resize_options:
            max_dim = max(h, w)
            if max_dim > resize_max:
                scale = resize_max / max_dim
                new_h = int(h * scale)
                new_w = int(w * scale)
                scale_loss = max_dim / resize_max
            else:
                new_h, new_w = h, w
                scale_loss = 1.0
            
            # Determine status
            if scale_loss < 2.0:
                status = "✓ Excellent"
                if best_resize is None:
                    best_resize = resize_max
            elif scale_loss < 3.0:
                status = "✓ Good"
                if best_resize is None:
                    best_resize = resize_max
            elif scale_loss < 4.0:
                status = "⚠ Acceptable"
                if best_resize is None:
                    best_resize = resize_max
            else:
                status = "✗ Poor"
            
            logger.info(f"  {resize_max:<12} {new_w}×{new_h:<10} {scale_loss:<12.2f}x {status}")
        
        logger.info(f"  {'-'*66}")
        logger.info(f"  Recommended resize_max: {best_resize}")
        
        # Calculate adaptive resize
        adaptive_resize = int((h / 2.0) // 8) * 8  # Divisible by 8
        adaptive_resize = max(640, min(adaptive_resize, 2048))
        logger.info(f"  Adaptive resize_max: {adaptive_resize}")
        
        # RANSAC threshold recommendation
        if grid_size <= 5:
            ransac = 3.0
        elif grid_size <= 7:
            ransac = 4.0
        else:
            ransac = 5.0
        
        logger.info(f"\n  Recommended Settings:")
        logger.info(f"  {'-'*66}")
        logger.info(f"  resize_max: {adaptive_resize}")
        logger.info(f"  ransac_threshold: {ransac}")
        logger.info(f"  min_matches: {20 + (grid_size - 5) * 5}")
        
        # Save diagnostic image
        output_path = f"diagnostic_grid_{grid_size}x{grid_size}.png"
        # Resize for visualization
        vis_size = 800
        scale = vis_size / max(h, w)
        vis_h, vis_w = int(h * scale), int(w * scale)
        vis = cv2.resize(stitched, (vis_w, vis_h))
        
        # Add text overlay
        cv2.putText(vis, f"{grid_size}x{grid_size} Grid", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis, f"Size: {w}x{h}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"Recommended resize: {adaptive_resize}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imwrite(output_path, vis)
        logger.info(f"  Saved visualization: {output_path}")
        
        results.append({
            'grid_size': grid_size,
            'stitched_size': (w, h),
            'tile_size': tile_size,
            'recommended_resize': adaptive_resize,
            'ransac_threshold': ransac,
        })
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY & RECOMMENDATIONS")
    logger.info(f"{'='*70}")
    
    logger.info(f"\n{'Grid':<8} {'Stitched Size':<18} {'Recommended resize_max':<25} {'RANSAC'}")
    logger.info(f"{'-'*70}")
    
    for r in results:
        logger.info(f"{r['grid_size']}×{r['grid_size']:<6} "
                   f"{r['stitched_size'][0]}×{r['stitched_size'][1]:<13} "
                   f"{r['recommended_resize']:<25} "
                   f"{r['ransac_threshold']}")
    
    logger.info(f"\n{'='*70}")
    logger.info("KEY INSIGHTS:")
    logger.info(f"{'='*70}")
    logger.info("1. Larger grids need proportionally larger resize_max")
    logger.info("2. Fixed resize_max=840 only works well for 3×3 and 5×5 grids")
    logger.info("3. For 7×7 and 9×9 grids, use adaptive resize (see above)")
    logger.info("4. Use main_loftr_adaptive.py for automatic adjustment")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    import sys
    
    # Configuration
    tiles_dir = r"C:\Binomial Technologies\Non GPS based Navigation\NGBN\Satellite Dataset\ajabgarh_z18Tiles_gmap"
    center_x = 186625
    center_y = 110502
    
    # Allow command-line override
    if len(sys.argv) > 1:
        tiles_dir = sys.argv[1]
    if len(sys.argv) > 3:
        center_x = int(sys.argv[2])
        center_y = int(sys.argv[3])
    
    if not os.path.exists(tiles_dir):
        logger.error(f"Tiles directory not found: {tiles_dir}")
        logger.error("Usage: python diagnose_grid_size.py [tiles_dir] [center_x] [center_y]")
        sys.exit(1)
    
    diagnose_grid_size(tiles_dir, center_x, center_y, grid_sizes=[3, 5, 7, 9])
    
    logger.info("\n✓ Diagnostic complete!")
    logger.info("Check the generated diagnostic_grid_*.png files")
    logger.info("\nTo fix your issue, run:")
    logger.info("  python main_loftr_adaptive.py")
