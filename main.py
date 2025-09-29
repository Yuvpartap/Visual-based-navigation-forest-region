import os
import sys
from src.create_trajectory_map import generate_trajectory_map, generate_trajectory_map_from_crops
from src.create_video import create_trajectory_video
from src.video_utils import get_video_info

def main():
    #path to the global map image
    map_path = "data/global_map.png"

    #directory to save output results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    #output paths for trajectory map image and video
    trajectory_img_path = os.path.join(results_dir, "trajectory_map.png")
    output_video_path = os.path.join(results_dir, "trajectory_video.avi")

    #check if we should use video input or crop directory
    input_video_path = r"C:\Binomial Technologies\Non GPS based Navigation\Earth_Studio\Munnar_videos\Munnar Hills_nadir_1.mp4"
    crops_dir = "data/crops"
    
    coords = None
    
    #prioritize video input if available
    if os.path.exists(input_video_path):
        print("Found video input, processing video...")
        
        #get video info
        video_info = get_video_info(input_video_path)
        if video_info:
            print(f"Video: {video_info['resolution']}, {video_info['fps']:.1f} FPS, {video_info['duration']:.1f}s")
            
            #adjust frame skip based on video length and FPS
            frame_skip = max(1, int(video_info['fps'] / 2))  # Process ~2 frames per second
            print(f"Using frame skip: {frame_skip}")
        else:
            frame_skip = 5
            
        coords = generate_trajectory_map(map_path, input_video_path, trajectory_img_path, frame_skip=frame_skip)
        print(f"Trajectory map saved to {trajectory_img_path}")
        
    elif os.path.exists(crops_dir) and os.listdir(crops_dir):
        print("No video found, using crop directory...")
        coords = generate_trajectory_map_from_crops(map_path, crops_dir, trajectory_img_path)
        print(f"Trajectory map saved to {trajectory_img_path}")
        
    else:
        print("Error: No input found!")
        print(f"Please provide either:")
        print(f"  - Video file at: {input_video_path}")
        print(f"  - Crop images in: {crops_dir}")
        sys.exit(1)

    #create a video based on trajectory coordinates
    if coords:
        print(f"Found {len(coords)} trajectory points")
        print("Creating trajectory visualization video")
        create_trajectory_video(map_path, coords, output_video_path, fps=4)
        print(f"Trajectory video saved to {output_video_path}")
    else:
        print("No coordinates found, trajectory video not created")

#entry point of the script
if __name__ == "__main__":
    main()
