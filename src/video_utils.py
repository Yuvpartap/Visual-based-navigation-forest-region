import cv2
import os

def extract_frames_from_video(video_path, output_dir, frame_skip=1, max_frames=None):
    """
    Extract frames from video and save them as individual images.
    Useful for debugging or creating crops from video.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        frame_skip: Extract every Nth frame (1 = all frames)
        max_frames: Maximum number of frames to extract (None = all)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video info: {total_frames} frames, {fps:.2f} FPS")
    
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_skip == 0:
            if max_frames and extracted_count >= max_frames:
                break
                
            frame_filename = os.path.join(output_dir, f"frame_{extracted_count:04d}.png")
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1
            print(f"Extracted frame {frame_count} -> {frame_filename}")
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {extracted_count} frames from {total_frames} total frames")
    return True

def get_video_info(video_path):
    """
    Get basic information about a video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        dict: Video information (fps, frame_count, duration, width, height)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        'fps': fps,
        'frame_count': frame_count,
        'duration': duration,
        'width': width,
        'height': height,
        'resolution': f"{width}x{height}"
    }

def create_video_from_frames(frames_dir, output_path, fps=30):
    """
    Create a video from a directory of frame images.
    
    Args:
        frames_dir: Directory containing frame images
        output_path: Output video file path
        fps: Frames per second for output video
    """
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if not frame_files:
        print("No frame files found in directory")
        return False
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    height, width = first_frame.shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)
        print(f"Added frame: {frame_file}")
    
    video_writer.release()
    print(f"Video created: {output_path}")
    return True
