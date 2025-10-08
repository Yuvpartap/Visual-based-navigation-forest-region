# 🚁 Visuals-based-Navigation

## Overview
This project is designed to reconstruct and visualize the trajectory of an Unmanned Aerial Vehicle (UAV) using a panoramic satellite map along with a series of drone-captured photos. The end products are a static trajectory map and a video showcasing the drone's route.

## 📊 Dataset
The dataset comprises:
- **Satellite Map**: A panoramic image of the area.
- **Drone Frames**: 50 consecutive photos captured during the UAV flight.

## 🔄 Workflow

### Input 
The system now supports two input methods:
1. **Video Input** (Recommended): Direct processing of drone video files
2. **Dataset Tiles Input**: Path to the tiles folder
3. **Center_x**: Initial tile center x coordinate
4. **Center_y**: Initial tile center y coordinate
5. **Zoom**: Zoom level of the satellite tiles
6. **Grid size (N)**: Satellite tiles grid size (N x N)

### Visuals based Navigation
- **Video Processing**: Extract frames from drone video automatically
- **Satellite map Loading**: Loads Satellite tiles in form of N by N grid and Stich it for further processing.
- **Keypoint Detection**: Implemented Dino approach to identify unique features in frames and the satellite map at global level
- **Matching**: Establish correspondences between drone frames and the global map through feature matching using LoFTR at Dense Level
- **Homography Calculation**: Compute transformation matrices to position frames accurately on the map
- **Updating the Tiles Grid**: Based on new tile match, it updates the grid by loading new tiles and performing the Drone-to-Satellite image matching again" 
- **Trajectory Visualization**: Illustrate the drone's flight path on the map

## Results
- **Static Map**: Provides a comprehensive visual representation of the flight path, indicating key points and the trajectory.

## 📁 Project Structure
```
├── opt_adaptive_loftr.py   # Main Optimized script for running the complete pipeline
├── Docs/                    # Contain md files for better understanding
| 
├── src/                   # Source code for main functionalities
│   ├── create_trajectory_map.py   # Generates a static trajectory map
│   ├── create_video.py             # Creates a trajectory video
|   ├── dino_loftr_matcher.py # Script where dino and loftr are loaded and image matching is performed
|   ├── tile_loading_utilis.py # Script for loading and stiching tiles in form of N x N grid 
│   ├── video_utilis.py # Script for handling video frames
|
├── data/                  # Input data directory
│   ├── satellite_data/    # Contain Satellite tiles for particular zoom
│   ├── Drone_video.mp4     # Drone video
│
├── results/               # Output data
│   ├── trajectory_map.png         # Final trajectory map
│   ├── match_00 ... match_n       # Visualization of each matched frame
│
├── requirements.txt       # Python dependencies
└── README.md              # Project description and instructions
```

🌟 Embark on a journey to visualize drone paths like never before!
