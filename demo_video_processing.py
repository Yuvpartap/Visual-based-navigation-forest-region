#!/usr/bin/env python3
"""
Demo script showing how to use the video-based drone trajectory tracker.
This script demonstrates both video processing and utility functions.
"""

import os
import sys
from src.create_trajectory_map import generate_trajectory_map, generate_trajectory_map_from_crops
from src.create_video import create_trajectory_video
from src.video_utils import get_video_info, extract_frames_from_video

def demo_video_processing():
    """
    Demonstrate video-based trajectory processing.
    """
    print("=== Drone Trajectory Tracker - Video Processing Demo ===\n")
    
    # Configuration
    map_path = "data/global_map.png"
    input_video_path = "data/input_video.mp4"
    results_dir = "results"
    
    # Check if required files exist
    if not os.path.exists(map_path):
        print(f"Error: Global map not found at {map_path}")
        return False
        
    if not os.path.exists(input_video_path):
        print(f"Error: Input video not found at {input_video_path}")
        print("Please place your drone video at data/input_video.mp4")
        return False
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Step 1: Get video information
    print("Step 1: Analyzing input video...")
    video_info = get_video_info(input_video_path)
    if video_info:
        print(f"  Resolution: {video_info['resolution']}")
        print(f"  FPS: {video_info['fps']:.2f}")
        print(f"  Duration: {video_info['duration']:.2f} seconds")
        print(f"  Total frames: {video_info['frame_count']}")
    else:
        print("  Error: Could not read video information")
        return False
    
    # Step 2: Process video for trajectory
    print("\nStep 2: Processing video for trajectory...")
    trajectory_img_path = os.path.join(results_dir, "trajectory_map.png")
    
    # Adjust frame skip based on video properties
    frame_skip = max(1, int(video_info['fps'] / 2))  # Process ~2 frames per second
    print(f"  Using frame skip: {frame_skip} (processing every {frame_skip} frames)")
    
    coords = generate_trajectory_map(map_path, input_video_path, trajectory_img_path, frame_skip=frame_skip)
    
    if coords:
        print(f"  Success! Found {len(coords)} trajectory points")
        print(f"  Trajectory map saved to: {trajectory_img_path}")
    else:
        print("  Error: No trajectory points found")
        return False
    
    # Step 3: Create trajectory video
    print("\nStep 3: Creating trajectory visualization video...")
    output_video_path = os.path.join(results_dir, "trajectory_video.avi")
    create_trajectory_video(map_path, coords, output_video_path, fps=4)
    print(f"  Trajectory video saved to: {output_video_path}")
    
    # Step 4: Optional - Extract sample frames for debugging
    print("\nStep 4: Extracting sample frames for debugging...")
    frames_dir = os.path.join(results_dir, "extracted_frames")
    success = extract_frames_from_video(input_video_path, frames_dir, frame_skip=frame_skip*2, max_frames=10)
    if success:
        print(f"  Sample frames saved to: {frames_dir}")
    
    print("\n=== Demo completed successfully! ===")
    print(f"Check the '{results_dir}' directory for outputs:")
    print(f"  - {trajectory_img_path}")
    print(f"  - {output_video_path}")
    print(f"  - {frames_dir}/")
    
    return True

def demo_backward_compatibility():
    """
    Demonstrate backward compatibility with crop directory processing.
    """
    print("\n=== Testing Backward Compatibility ===")
    
    crops_dir = "data/crops"
    if os.path.exists(crops_dir) and os.listdir(crops_dir):
        print("Found crop directory, testing original functionality...")
        
        map_path = "data/global_map.png"
        output_path = "results/trajectory_map_from_crops.png"
        
        coords = generate_trajectory_map_from_crops(map_path, crops_dir, output_path)
        
        if coords:
            print(f"Success! Processed {len(coords)} crop images")
            print(f"Trajectory map saved to: {output_path}")
        else:
            print("No coordinates found from crop processing")
    else:
        print("No crop directory found, skipping backward compatibility test")

if __name__ == "__main__":
    success = demo_video_processing()
    
    if success:
        demo_backward_compatibility()
    else:
        print("\nDemo failed. Please check your input files and try again.")
        sys.exit(1)
