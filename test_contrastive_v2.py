import os

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
df = df.drop(df[df["Id"] == "new_whale"].index)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseNetwork()
model.load_state_dict(torch.load(path_model))
model.to(device)
model.eval()

transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

criterion = ContrastiveLoss()
if __name__ == "__main__":

    list_with_distance = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        filename = row["Image"]
        whale_id = row["Id"]
        img_path = os.path.join(images_dir, filename)
        img0 = Image.open(os.path.join(images_dir, query_filename))  # Путь до картинки на которую надо найти похожие
        img1 = Image.open(img_path)
        img0 = img0.convert("L")
        img1 = img1.convert("L")
        # Apply image transformations
        if transform is not None:
            img0 = transform(img0)
            img1 = transform(img1)
        img0 = img0.unsqueeze(0)
        img1 = img1.unsqueeze(0)
        output1, output2 = model.predict(img0, img1)

        # loss_contrastive = criterion(output1,output2,label)
        eucledian_distance = F.pairwise_distance(output1, output2).to('cpu').numpy()[0]
        list_with_distance.append([eucledian_distance, filename, whale_id])

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
