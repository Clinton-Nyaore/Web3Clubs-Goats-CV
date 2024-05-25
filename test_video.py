import cv2
import numpy as np
from ultralytics import YOLO

# Load the trained model (replace 'path/to/model.pt' with the actual path)
model = YOLO("cabra_best.pt")


def predict_video(video_path):
    # Perform object detection on the video
    cap = cv2.VideoCapture(video_path)

    frame_count = 0  # Initialize frame count

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model.predict(frame)

            # Iterate over the detections
            for result in results:
                # Set the names attribute
                result.names = {0: 'goat'}

                # Extract bounding box coordinates
                x1, y1, x2, y2 = result.xyxy[0]

                # Crop the frame using the bounding box coordinates
                cropped_frame = frame[int(y1):int(y2), int(x1):int(x2)]

                # Save the cropped frame as a .jpg image
                cv2.imwrite(f"frame_{frame_count}.jpg", cropped_frame)

            frame_count += 1  # Increment frame count

            # Display the original frame
            cv2.imshow("Original Frame", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()


# Load video for detection
video = "./test_data/videos/pen_recording.mp4"

predict_video(video_path=video)
