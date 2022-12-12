import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

import config
from Contrastive_loss import ContrastiveLoss
from Siamese_Network import SiameseNetwork

testing_csv = config.training_csv_contrastive
images_dir = config.images_dir
path_model = "model_contrastive.pt"
query_filename = "0a00c7a0f.jpg"

# TODO добавить в конфиг
df = pd.read_csv(testing_csv)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseNetwork()
model.load_state_dict(torch.load(path_model))
model.to(device)
model.eval()

with open('database.pkl', 'rb') as f:
    filenames, whale_ids, embeddings = pickle.load(f)

transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

criterion = ContrastiveLoss()
if __name__ == "__main__":

    list_with_distance = []

    for i, item in enumerate(tqdm(embeddings)):

        img = Image.open(os.path.join(images_dir, query_filename))  # Путь до картинки на которую надо найти похожие
        img = img.convert("L")
        # Apply image transformations
        if transform is not None:
            img = transform(img)
        img = img.unsqueeze(0)
        output = model.predict(img)

        item = item.to(device)
        eucledian_distance = F.pairwise_distance(output, item).to('cpu').numpy()[0]
        list_with_distance.append([eucledian_distance, filenames[i], whale_ids[i]])

    axes = []
    fig = plt.figure(figsize=(8, 8))
    list_with_distance.sort()  # отсортированные по расстоянию изображения
    # ToDO добавить сопаставление изображений из list_with_distance и их классами
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
