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


def recall_k(filename, whale_ids, embeddings, transform,
             model, images_dir, query_filename_id, device,
             top_n=5):

    recall_per_id = []
    for _, row in query_filename_id.iterrows():
        query_filename = row[0]
        query_whale_id = row[1]
        img = Image.open(os.path.join(images_dir, query_filename))  # Путь до картинки на которую надо найти похожие
        img = img.convert("L")
        # Apply image transformations
        if transform is not None:
            img = transform(img)
        img = img.unsqueeze(0)
        output = model.predict(img)

        list_with_distance = []
        for i, item in enumerate(tqdm(embeddings)):
            item = item.to(device)
            eucledian_distance = F.pairwise_distance(output, item).to('cpu').numpy()[0]
            list_with_distance.append([eucledian_distance, filename[i], whale_ids[i]])
        list_with_distance.sort()

        recall_per_id.append(0)
        for item in list_with_distance[:top_n]:
            if query_whale_id == item[2]:
                recall_per_id[-1] = 1
                break

    recall = sum(recall_per_id) / len(recall_per_id)
    return recall
