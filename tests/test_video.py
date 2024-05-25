import cv2
from ultralytics import YOLO

# Load the trained model (replace 'path/to/model.pt' with the actual path)
model = YOLO("cabra_best.pt")


def predict_video(video_path):
    # Perform object detection on the video
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model.predict(frame)

            # Iterate over the detections and set the names attribute
            for result in results:
                result.names = {0: 'goat'}

            print("Results: ", results)
            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            print("Annotated frame: ", annotated_frame)

            # Display the annotated frame
            cv2.imshow("Model Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    return cv2.destroyAllWindows()


# Load video for detection
video = "./test_data/videos/pen_recording.mp4"

print(predict_video(video_path=video))