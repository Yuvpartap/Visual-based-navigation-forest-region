import os
from src.create_trajectory_map import generate_trajectory_map
from src.create_video import create_trajectory_video

def main():
    #path to the global map image
    map_path = "data/global_map.png"

    #directory containing cropped input frames
    crops_dir = "data/crops"

    #directory to save output results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    #output paths for trajectory map image and video
    trajectory_img_path = os.path.join(results_dir, "trajectory_map.png")
    video_path = os.path.join(results_dir, "trajectory_video.avi")

    #generate a trajectory map and get coordinates
    print("Generating trajectory map")
    coords = generate_trajectory_map(map_path, crops_dir, trajectory_img_path)
    print(f"Trajectory map saved to {trajectory_img_path}")

    #create a video based on trajectory coordinates
    if coords:
        print("Creating video")
        create_trajectory_video(map_path, coords, video_path, fps=4)
        print(f"Video saved to {video_path}")
    else:
        print("No coordinates found, video not created")

#entry point of the script
if __name__ == "__main__":
    main()