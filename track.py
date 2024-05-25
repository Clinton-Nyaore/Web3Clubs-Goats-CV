import io
import cv2
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import os

# Load the trained model (replace 'path/to/model.pt' with the actual path)
model_yolo = YOLO("best.pt")

# Load the pre-trained image classification model for embedding generation
model_resnet = models.resnet18(pretrained=True)
# Remove the final classification layer
model_resnet = torch.nn.Sequential(*list(model_resnet.children())[:-1])
model_resnet.eval()

# Function to generate embeddings for goat images
def generate_embedding(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None
    
    # Preprocess image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    
    # Generate embedding
    with torch.no_grad():
        try:
            embedding = model_resnet(image)
        except Exception as e:
            print(f"Error generating embedding for image {image_path}: {e}")
            return None
    
    return embedding.squeeze().numpy() if embedding is not None else None

def compare_embeddings(video_embedding, goat_embeddings):
    if video_embedding is None:
        return None
    
    similarities = []
    for goat_embedding in goat_embeddings:
        if goat_embedding is None:
            similarities.append(0.0)
        else:
            similarity = np.dot(video_embedding, goat_embedding) / (np.linalg.norm(video_embedding) * np.linalg.norm(goat_embedding))
            similarities.append(similarity)
    max_similarity_index = np.argmax(similarities)
    return max_similarity_index


def predict_single_image(image_path):
    # Load the single image
    frame = cv2.imread(image_path)

    # Run YOLOv8 inference on the image
    results = model_yolo(frame)

    # Iterate over the detections and set the names attribute
    for result in results:
        result.names = {0: 'goat'}

    # Visualize the results on the image
    annotated_frame = results[0].plot()

    # Extract detected goat regions
    goat_regions = [result.boxes.xyxy[0] for result in results if result.names[0] == 'goat']


    # Get all file names in the folder
    goat_images = ['./unique_goats/layer2/12.jpg','./unique_goats/layer2/16.jpg']

    # Generate embeddings for goat images
    goat_embeddings = [generate_embedding(image_path) for image_path in goat_images]

    # Iterate over detected goat regions and re-identify the goat
    for goat_region in goat_regions:
        # Extract embedding from the detected goat region
        goat_image = frame[int(goat_region[1]):int(goat_region[3]), int(goat_region[0]):int(goat_region[2])]
        goat_image_pil = Image.fromarray(goat_image)
        video_embedding = generate_embedding(goat_image_pil)
        # Compare with goat embeddings
        max_similarity_index = compare_embeddings(video_embedding, goat_embeddings)
        if max_similarity_index is not None:
            tracked_goat = goat_images[max_similarity_index]
            print("Tracked goat:", tracked_goat)
        else:
            print("No goat tracked.")

    # Display the annotated image
    cv2.imshow("Model Inference", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Load a single image for detection
image = "./unique_goats/layer2/12.jpg"

predict_single_image(image_path=image)
