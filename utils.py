import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm

def imshow(img):
    img = img / 2 + 0.5  # денормализуем
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def recall_k(filename,whale_ids,embeddings,transform,model,images_dir,query_filename_id,device):
    list_with_distance = []
    query_filename=query_filename_id.iloc[0][0]
    query_whale_id=query_filename_id.iloc[0][1]
    img = Image.open(os.path.join(images_dir, query_filename))  # Путь до картинки на которую надо найти похожие
    img = img.convert("L")
    # Apply image transformations
    if transform is not None:
        img = transform(img)
    img = img.unsqueeze(0)
    output = model.predict(img)
    for i, item in enumerate(tqdm(embeddings)):
        item = item.to(device)
        eucledian_distance = F.pairwise_distance(output, item).to('cpu').numpy()[0]
        list_with_distance.append([eucledian_distance, filename[i], whale_ids[i]])
    list_with_distance.sort()
    for item in list_with_distance[:5]:
        if query_whale_id in item:
            return 1
        else:
            return 0
