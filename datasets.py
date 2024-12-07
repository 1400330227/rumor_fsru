"""
An Lao
"""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import os

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class MyDataset(Dataset):
    def __init__(self, dataset):
        self.text = torch.from_numpy(dataset['text'])
        self.image = list(dataset['image'])  # torch.from_numpy(dataset['image'])
        self.label = torch.from_numpy(dataset['label'])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        image = self.image[item]
        try:
            if os.path.exists(image):
                image = Image.open(image).convert('RGB')
            else:
                print(image)
                image = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 255)
            image = data_transforms(image)
        except Exception as e:
            print(e)
            print(image)

        return self.text[item], image, self.label[item]
