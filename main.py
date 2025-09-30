import os
import sys
from src.create_trajectory_map import generate_trajectory_map, generate_trajectory_map_from_crops, generate_dynamic_tile_matching
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
    input_video_path = r"C:\Binomial Technologies\Non GPS based Navigation\Earth_Studio\Ajabgarh_videos\Ajabgarh_right.mp4"
    crops_dir = "data/crops"
    # optional tile configuration
    tiles_dir = r"C:\Binomial Technologies\Non GPS based Navigation\NGBN\Satellite Dataset\ajabgarh_z18Tiles_gmap" #os.environ.get("TILES_DIR", None)  # directory with <x>_<y>.png tiles at zoom 19
    initial_tile_x = 186625#os.environ.get("INITIAL_TILE_X", None)
    initial_tile_y = 110502#os.environ.get("INITIAL_TILE_Y", None)
    grid_size = 9 #int(os.environ.get("GRID_SIZE", "3"))
    
    coords = None
    
    #prioritize tile-based matching if configured and video available
    if os.path.exists(input_video_path) and tiles_dir and initial_tile_x and initial_tile_y:
        print("Found video and tiles, running dynamic tile-based matching...")
        video_info = get_video_info(input_video_path)
        if video_info:
            frame_skip = max(1, int(video_info['fps'] / 2))
            print(f"Using frame skip: {frame_skip}")
        else:
            frame_skip = 5
        coords = generate_dynamic_tile_matching(
            tiles_dir=tiles_dir,
            initial_tile_x=int(initial_tile_x),
            initial_tile_y=int(initial_tile_y),
            video_path=input_video_path,
            grid_size=grid_size,
            frame_skip=frame_skip,
            ext=".png",
            tile_cache_items=1024,
            min_good_matches=10,
        )
        # note: tile centers are returned; trajectory video will still draw over the global map if provided
        print(f"Computed {len(coords)} dynamic tile centers")
    #fallback: video matching against global map image
    elif os.path.exists(input_video_path):
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
    else:
        print("No coordinates found, trajectory video not created")

#entry point of the script
if __name__ == "__main__":
    main()
