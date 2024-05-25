import os
import random
from torch.utils.data import Dataset
from PIL import Image
import torch

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
