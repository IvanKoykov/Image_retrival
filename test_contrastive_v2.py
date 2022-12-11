# Load the test dataset
import numpy as np
import os
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import config
from utils import imshow
import torchvision
from torch.autograd import Variable
from PIL import Image
import PIL.ImageOps
import os
from Whael_Dataset import SiameseDataset
from Siamese_Network import SiameseNetwork
from Contrastive_loss import ContrastiveLoss

testing_csv=config.training_csv_contrastive
testing_dir=config.training_dir
path_model=''
query_path=''
# TODO добавить в конфиг
df = pd.read_csv(testing_csv)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseNetwork()
#model.load_state_dict(torch.load(path_model))
#model.eval()

transform=transforms.Compose(
            [transforms.Resize((128, 128)), transforms.ToTensor()])

criterion = ContrastiveLoss()
if __name__ == '__main__':

    list_with_distance=[]
    for filename in os.listdir(testing_dir):
        img_path = os.path.join(testing_dir, filename)
        img0 = Image.open(query_path) # Путь до картинки на которую надо найти похожие
        img1 = Image.open(img_path)
        img0 = img0.convert("L")
        img1 = img1.convert("L")
        # Apply image transformations
        if transform is not None:
            img0 = transform(img0)
            img1 = transform(img1)

        output1, output2 = model(img0.to(device), img1.to(device))

        #loss_contrastive = criterion(output1,output2,label)
        eucledian_distance = F.pairwise_distance(output1, output2)
        list_with_distance.append([eucledian_distance,filename])

    axes = []
    fig = plt.figure(figsize=(8, 8))
    list_with_distance.sort() # отсортированные по расстоянию изображения
    #ToDO добавить сопаставление изображений из list_with_distance и их классами
    for i in range(5):
        score = list_with_distance[i]
        axes.append(fig.add_subplot(5, 6, i + 1))
        subplot_title = str(score[0])
        axes[-1].set_title(subplot_title)
        plt.axis('off')
        plt.imshow(Image.open(score[1]))
    fig.tight_layout()
    plt.show()