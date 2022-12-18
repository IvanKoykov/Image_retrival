import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from utils import best_images
import config
from Contrastive_loss import ContrastiveLoss
from Triplet_loss import TripletLoss
from Siamese_Network import SiameseNetwork
from TripletNetwork import TripletNetwork



images_dir = config.images_dir
testing_csv=config.testing_csv
df_test=pd.read_csv(testing_csv)

transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

flag='contrastive'
if flag=='contrastive':
    criterion= ContrastiveLoss()
    model = SiameseNetwork()
    path_model = 'model/model_contrastive_best.pt'
    with open('models/embeddings_best_contrastive.pkl', 'rb') as f:
        filenames, whale_ids, embeddings = pickle.load(f)
else:
    criterion=TripletLoss()
    model=TripletNetwork()
    path_model = 'model/model_triplet_best.pt'
    with open('models/embeddings_best_triplet.pkl', 'rb') as f:
        filenames, whale_ids, embeddings = pickle.load(f)

model.load_state_dict(torch.load(path_model))
model.to(device)
model.eval()
if __name__ == "__main__":

    for _,row in df_test.iterrows():
        list_with_distance,query_whale_ids,query_filename = best_images(filenames, whale_ids, embeddings, transform, model,
                        images_dir, row[0],row[1], device)
        axes = []
        fig = plt.figure(figsize=(8, 8))
        for i in range(10):
            score = list_with_distance[i]
            axes.append(fig.add_subplot(10, 5, i + 1))
            # subplot_title = str(score[0].numpy()[0])
            subplot_title = score[2]
            axes[-1].set_title(subplot_title)
            plt.axis("off")
            plt.imshow(Image.open(os.path.join(images_dir, score[1])))
        fig.tight_layout()
        plt.show()
