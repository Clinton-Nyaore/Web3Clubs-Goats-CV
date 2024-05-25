"""
### predict.py

### This module is used to make predictions on new images using the our trained model.

"""

import io
import cv2
from PIL import Image
from ultralytics import YOLO

# Load the trained model (replace 'path/to/model.pt' with the actual path)
model = YOLO("best.pt")

def predict_image(image):
    # Perform object detection on the image
    results = model.predict(image, save=True, imgsz=320, conf=0.5)
    #frame = cv2.imencode('.jpg', image)[1].tobytes()

    #image = Image.open(io.BytesIO(frame))
    #results = model(image, save=True)

    # Visualize the results
    #res_plotted = results[0].plot()

    # Display the annotated image
    
    return results
    

def predict_video(video_path):
    # Perform object detection on the video
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame)

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


# Load the image for detection (replace 'path/to/image.jpg' with the actual path)
image = "2.jpg"

# Load video for detection
video = "./test_data/videos/pen_recording.mp4"

print(predict_image(image=image))