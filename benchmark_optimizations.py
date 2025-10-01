"""
Benchmark script to compare optimization strategies.

Tests different optimization levels and reports speedup.
"""

import cv2
import numpy as np
import time
import torch
import logging
from src.dino_loftr_matcher import create_dino_loftr_matcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def benchmark_single_match(frame, map_img, config_name, **kwargs):
    """
    Benchmark a single matching configuration.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {config_name}")
    logger.info(f"{'='*60}")
    
    # Create matcher with config
    resize_max = kwargs.get('resize_max', 1792)
    use_mixed_precision = kwargs.get('use_mixed_precision', False)
    
    matcher = create_dino_loftr_matcher(
        use_gpu=True,
        loftr_model="outdoor",
        resize_max=resize_max
    )
    
    # Warm-up run
    logger.info("Warming up...")
    _ = matcher.match_images(frame, map_img, min_matches=20)
    
    # Benchmark runs
    num_runs = 5
    times = []
    
    logger.info(f"Running {num_runs} iterations...")
    for i in range(num_runs):
        start = time.time()
        
        if use_mixed_precision and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                src_pts, dst_pts, conf, num_matches = matcher.match_images(
                    frame, map_img, min_matches=20
                )
        else:
            src_pts, dst_pts, conf, num_matches = matcher.match_images(
                frame, map_img, min_matches=20
            )
        
        elapsed = time.time() - start
        times.append(elapsed)
        logger.info(f"  Run {i+1}: {elapsed:.3f}s ({num_matches} matches)")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    logger.info(f"\nResults:")
    logger.info(f"  Average: {avg_time:.3f}s ± {std_time:.3f}s")
    logger.info(f"  Min: {min(times):.3f}s")
    logger.info(f"  Max: {max(times):.3f}s")
    
    return avg_time, num_matches


def main():
    """
    Run benchmark comparing different optimization strategies.
    """
    logger.info("="*60)
    logger.info("LoFTR OPTIMIZATION BENCHMARK")
    logger.info("="*60)
    
    # Check GPU
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA: {torch.version.cuda}")
    else:
        logger.warning("No GPU available - mixed precision disabled")
    
    # Create test images
    logger.info("\nCreating test images...")
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    map_img = np.random.randint(0, 255, (3584, 3584, 3), dtype=np.uint8)  # 7×7 grid
    
    logger.info(f"Frame size: {frame.shape}")
    logger.info(f"Map size: {map_img.shape}")
    
    # Test configurations
    configs = [
        {
            'name': 'Baseline (No Optimization)',
            'resize_max': 1792,
            'use_mixed_precision': False,
        },
        {
            'name': 'Mixed Precision (FP16)',
            'resize_max': 1792,
            'use_mixed_precision': True,
        },
        {
            'name': 'Reduced Resolution',
            'resize_max': 1280,
            'use_mixed_precision': False,
        },
        {
            'name': 'Mixed Precision + Reduced Resolution',
            'resize_max': 1280,
            'use_mixed_precision': True,
        },
    ]
    
    results = []
    baseline_time = None
    
    for config in configs:
        try:
            avg_time, num_matches = benchmark_single_match(
                frame, map_img, config['name'], **config
            )
            
            if baseline_time is None:
                baseline_time = avg_time
                speedup = 1.0
            else:
                speedup = baseline_time / avg_time
            
            results.append({
                'name': config['name'],
                'time': avg_time,
                'speedup': speedup,
                'matches': num_matches
            })
            
        except Exception as e:
            logger.error(f"Failed: {e}")
            continue
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK SUMMARY")
    logger.info("="*80)
    logger.info(f"{'Configuration':<40} {'Time':<12} {'Speedup':<12} {'Matches'}")
    logger.info("-"*80)
    
    for r in results:
        logger.info(f"{r['name']:<40} {r['time']:.3f}s{'':<6} "
                   f"{r['speedup']:.2f}x{'':<7} {r['matches']}")
    
    logger.info("="*80)
    
    # Recommendations
    logger.info("\nRECOMMENDATIONS:")
    logger.info("-"*80)
    
    if torch.cuda.is_available():
        best = max(results, key=lambda x: x['speedup'])
        logger.info(f"✓ Best configuration: {best['name']}")
        logger.info(f"  Speedup: {best['speedup']:.2f}x faster")
        logger.info(f"  Time: {best['time']:.3f}s per frame")
        
        if best['speedup'] >= 2.0:
            logger.info(f"\n🎉 Excellent! {best['speedup']:.1f}x speedup achieved!")
        elif best['speedup'] >= 1.5:
            logger.info(f"\n✓ Good! {best['speedup']:.1f}x speedup achieved")
        else:
            logger.info(f"\n⚠ Moderate speedup: {best['speedup']:.1f}x")
    else:
        logger.info("⚠ No GPU available - limited optimization possible")
        logger.info("  Consider: Reduce resize_max or frame_skip")
    
    logger.info("\nFor production use:")
    logger.info("  python main_loftr_optimized.py")
    logger.info("="*80)


if __name__ == "__main__":
    main()
