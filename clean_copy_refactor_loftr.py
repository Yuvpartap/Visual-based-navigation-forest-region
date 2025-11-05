"""
Main script for drone video processing with satellite tile matching.
Modularized version with utilities organized in src/ folder.

This script processes drone video footage and matches it against satellite tiles
to determine the drone's position and trajectory.
"""

import os
import sys
import time
import logging
from src.video_processor import VideoProcessor
from src.video_utils import get_video_info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for video processing."""
    from src.google_maps_plotter import GoogleMapsTrajectoryPlotter
    
    vtime = time.time()

    # Configuration parameters
    z = 19
    video_path = r"C:\Binomial Technologies\Non GPS based Navigation\Earth_Studio\France_videos\france_multi_rotate.mp4"
    tiles_dir = r"C:\Binomial Technologies\Non GPS based Navigation\NGBN\Satellite Dataset\france_z19Tiles_osm"
    
    results_dir = f"results_france_normal_vid{vtime}"
    os.makedirs(results_dir, exist_ok=True)

    # Validate paths
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)
    if not os.path.exists(tiles_dir):
        logger.error(f"Tiles directory not found: {tiles_dir}")
        sys.exit(1)
    
    # Get video info and calculate frame skip
    video_info = get_video_info(video_path)
    frame_skip = 10
    
    # Calculate RANSAC threshold based on grid size
    grid_size = 7
    if grid_size <= 5:
        ransac_threshold = 3.0
    elif grid_size <= 7:
        ransac_threshold = 4.0
    else:
        ransac_threshold = 5.0
    
    # Configuration dictionary
    config = {
        'video_path': video_path,
        'tiles_dir': tiles_dir,
        'zoom': z,
        'initial_tile_x': 265389,
        'initial_tile_y': 180405,
        'grid_size': grid_size,
        'frame_skip': frame_skip,
        'ext': '.png',
        'tile_cache_items': 128,
        'min_good_matches': 20,
        'save_dir': results_dir,
        'adaptive_resize': True,
        'cleanup_interval': 15,
        'save_every_nth': 1,
        'ransac_threshold': ransac_threshold,
        'track_rotation': True,
        'rotation_validation_interval': 5,
    }
    
    # Process video
    start_time = time.time()
    processor = VideoProcessor(config)
    centers, _ = processor.process()
    
    # Generate Google Maps visualizations
    GoogleMapsTrajectoryPlotter.create_google_maps_html(
        centers, z, results_dir, api_key=None
    )
    GoogleMapsTrajectoryPlotter.create_kml_file(
        centers, z, results_dir
    )
    
    total_time = time.time() - start_time
    
    # Log final summary
    logger.info("\n" + "=" * 70)
    logger.info("✓ PROCESSING COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    logger.info(f"Trajectory points: {len(centers)}")
    if len(centers) > 0:
        logger.info(f"Average time per point: {total_time/len(centers):.2f}s")
        logger.info(f"Effective processing rate: {len(centers)/total_time:.2f} points/sec")
    logger.info(f"Results directory: {results_dir}/")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
