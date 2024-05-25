import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os
import random
from torch.utils.data import Dataset, DataLoader
from torchvision.models.resnet import ResNet18_Weights

# Siamese Network definition using a pre-trained ResNet
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )
    
    def forward_once(self, x):
        x = self.resnet(x)
        return x
    
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

# Contrastive loss definition
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

# Custom dataset for loading image pairs
class SiameseNetworkDataset(Dataset):
    def __init__(self, image_folder_dataset, transform=None):
        self.image_folder_dataset = image_folder_dataset
        self.transform = transform
        self.image_list = [os.path.join(dp, f) for dp, dn, fn in os.walk(image_folder_dataset) for f in fn]

    def __getitem__(self, index):
        img0_path = random.choice(self.image_list)
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            img1_path = random.choice([os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.dirname(img0_path)) for f in fn])
        else:
            img1_path = random.choice(self.image_list)
            while os.path.dirname(img0_path) == os.path.dirname(img1_path):
                img1_path = random.choice(self.image_list)
        
        img0 = Image.open(img0_path).convert("RGB")
        img1 = Image.open(img1_path).convert("RGB")
        
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        label = torch.tensor([int(os.path.dirname(img0_path) != os.path.dirname(img1_path))], dtype=torch.float32)
        return img0, img1, label
    
    def __len__(self):
        return len(self.image_list)

# Function to get embedding for an image
def get_embedding(model, image_path, data_transforms, device):
    image = Image.open(image_path).convert("RGB")
    image = data_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.forward_once(image)
    return embedding

# Function to check if two images are of the same goat
def is_same_goat(model, img1_path, img2_path, data_transforms, device, threshold=1.0):
    embedding1 = get_embedding(model, img1_path, data_transforms, device)
    embedding2 = get_embedding(model, img2_path, data_transforms, device)
    euclidean_distance = F.pairwise_distance(embedding1, embedding2)
    return euclidean_distance.item() < threshold

# Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transformation
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset and DataLoader
    train_dataset = SiameseNetworkDataset(image_folder_dataset="./goat_images", transform=data_transforms)
    train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=4, batch_size=32)

    # Model, Loss, Optimizer
    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        for i, data in enumerate(train_dataloader, 0):
            img0, img1, label = data
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)
            
            optimizer.zero_grad()
            output1, output2 = model(img0, img1)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    # Goat identification example
    img1_path = "./goat_images/eth1/1.jpg"
    known_goat_images = ["./goat_images/layer2/1.jpg", "./goat_images/layer2/2.jpg", "./goat_images/layer2/3.jpg"]

    for known_image in known_goat_images:
        if is_same_goat(model, img1_path, known_image, data_transforms, device):
            print("This is the same goat")
            break
    else:
        print("This is a different goat")

if __name__ == "__main__":
    main()
