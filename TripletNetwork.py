from pathlib import Path
import os
import numpy as np
import pandas
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import transforms
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TripletNetwork(nn.Module):
    def __init__(self):
        super(TripletNetwork, self).__init__()
        # Setting up the Sequential of CNN Layers
        self.net = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.net.fc = nn.Sequential(
            nn.Linear(2048, 256)
        )
        ct = 0
        for child in self.net.children():
            ct += 1
        if ct < 5:
            for param in child.parameters():
                param.requires_grad = False

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def forward_once(self, x):
        x = x.to(device)
        # Forward pass
        output = self.net(x)
        return output

    def forward(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        return output1, output2, output3

    def predict(self, input):
        with torch.no_grad():
            output = self.forward_once(input)
        return output

    def generate_embeddings(self, img_dir: str, labels: pd.DataFrame):
        #df = pandas.read_csv(labels)
        embeddings = []
        ids = []
        names = []
        for _, row in tqdm(labels.iterrows(), total=labels.shape[0], desc="Generating database"):
            filename = row["Image"]
            img_path = os.path.join(img_dir, filename)
            img = Image.open(img_path)
            img = img.convert("RGB")
            # Apply image transformations
            img_tensor = self.transform(img)
            img_tensor = img_tensor.unsqueeze(0)
            with torch.no_grad():
                res = self.forward_once(img_tensor).to('cpu')
            #res = self.forward_once(img_tensor).to('cpu')
            embeddings.append(res)
            ids.append(row["Id"])
            names.append(filename)
            #embeddings.append(res.numpy)
        res = [names, ids, embeddings]
        return res
