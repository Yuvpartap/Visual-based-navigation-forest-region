# 🚁 Drone Trajectory Tracker

## Overview
This project is designed to reconstruct and visualize the trajectory of an Unmanned Aerial Vehicle (UAV) using a panoramic satellite map along with a series of drone-captured photos. The end products are a static trajectory map and a video showcasing the drone's route.

## 📊 Dataset
The dataset comprises:
- **Satellite Map**: A panoramic image of the area.
- **Drone Frames**: 50 consecutive photos captured during the UAV flight.

## 🔄 Workflow
### 1. Trajectory Map Generation
- **Keypoint Detection**: Implement the SIFT algorithm to identify unique features in both frames and the satellite map.
- **Matching**: Establish correspondences between drone frames and the global map through feature matching.
- **Homography Calculation**: Compute transformation matrices to position frames accurately on the map.
- **Trajectory Visualization**: Illustrate the drone's flight path on the map.

### 2. Video Generation
- Develop a dynamic video displaying the UAV’s movement in real-time.
- Highlight the drone's current position, the trajectory line, and the start/end points.

## Results
- **Static Map**: Provides a comprehensive visual representation of the flight path, indicating key points and the trajectory.
- **Video**: Delivers a dynamic illustration of the UAV's journey.

## 📁 Project Structure
```
├── main.py                # Main script for running the complete pipeline
│
├── src/                   # Source code for main functionalities
│   ├── create_trajectory_map.py   # Generates a static trajectory map
│   ├── create_video.py             # Creates a trajectory video
│
├── data/                  # Input data directory
│   ├── crops/             # Series of UAV images (frames)
│   ├── global_map.png     # Panoramic satellite map
│
├── results/               # Output data
│   ├── trajectory_map.png         # Final trajectory map
│   ├── trajectory_video.avi       # Visualized flight route video
│
├── requirements.txt       # Python dependencies
└── README.md              # Project description and instructions
```

🌟 Embark on a journey to visualize drone paths like never before!