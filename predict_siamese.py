import torch
import os
from torchvision import transforms
from PIL import Image
from models_siamese import SiameseNetwork
import torch.nn.functional as F

def get_embedding(model, image_path, data_transforms, device):
    image = Image.open(image_path).convert("RGB")
    image = data_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.forward_once(image)
    return embedding

def is_same_goat(model, img1_path, img2_path, data_transforms, device, threshold=1.0):
    embedding1 = get_embedding(model, img1_path, data_transforms, device)
    embedding2 = get_embedding(model, img2_path, data_transforms, device)
    euclidean_distance = F.pairwise_distance(embedding1, embedding2)
    return euclidean_distance.item() < threshold


def similar_goat(img1_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transformation
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load the model
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load('./models/siamese_model_gpu.pth', map_location=torch.device('cpu')))
    model.eval()

    # Goat identification example
    goat_folders = ["eth1", "layer2", "zk"]

    found_goat = False
    for goat_name in goat_folders:
        folder_path = f"./goat_images/{goat_name}/"
        known_goat_images = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith('.jpg')]

        for known_image in known_goat_images:
            if is_same_goat(model, img1_path, known_image, data_transforms, device):
                print(f"This is : {goat_name}")
                found_goat = True
                break

        if found_goat:
            break
    else:
        print("This is a different goat")

if __name__ == "__main__":
    img1_path = "./goat_images/zk/1.jpg"
    similar_goat(img1_path)
