import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import SiameseNetworkDataset
from models import SiameseNetwork, ContrastiveLoss

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
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 20
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

    # Save the model
    torch.save(model.state_dict(), 'siamese_model.pth')

if __name__ == "__main__":
    main()
