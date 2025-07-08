import cv2

def create_trajectory_video(map_path, coords, output_path, fps=4):
    #load the base map image
    map_img = cv2.imread(map_path, cv2.IMREAD_COLOR)
    height, width = map_img.shape[:2]

    #initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    #iterate through the coordinates
    for i in range(1, len(coords)):
        #show progress
        print(f"Creating video frame {i}/{len(coords)}")

        temp_map = map_img.copy()

        #draw lines and points
        for j in range(1, i + 1):
            cv2.line(temp_map, coords[j - 1], coords[j], (255, 0, 0), 3)  # trajectory line
            cv2.circle(temp_map, coords[j - 1], 5, (0, 255, 0), -1)       # green dots

        #draw the start point
        cv2.circle(temp_map, coords[0], 8, (0, 255, 0), -1)
        cv2.putText(temp_map, "START", (coords[0][0] + 5, coords[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        #draw the current endpoint
        cv2.circle(temp_map, coords[i], 8, (0, 0, 255), -1)

        #write the same frame multiple times to create a pause effect
        for _ in range(fps):
            video_writer.write(temp_map)

    #add pause on the last frame
    for _ in range(fps * 2):
        video_writer.write(temp_map)

    #release the video writer
    video_writer.release()
