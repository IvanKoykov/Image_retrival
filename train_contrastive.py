import os.path
import pickle
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import torchvision.utils
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from Contrastive_loss import ContrastiveLoss
from Siamese_Network import SiameseNetwork
from utils import imshow, recall_k
from Whael_Dataset import SiameseDataset

training_dir = config.images_dir
training_csv = config.training_csv_contrastive
# training_csv='old_train.csv'
testing_csv = config.testing_csv

df = pd.read_csv(training_csv)
df_test = pd.read_csv(testing_csv)
numClasses = np.unique(df["Id"])
idx = {
    numClasses[i]: np.where(df["Id"] == numClasses[i])[0]
    for i in range(len(numClasses))
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter("logs/contrastive", comment="contrastive_loss")
contrastiv_loss = torch.nn.MarginRankingLoss().to(device)
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


# breakpoint()
# train the model
def train(optimizer, criterion, scheduler):
    # loss = []
    # counter = []
    # iteration_number = 0
    recall_5_best = -1
    recall_10_best = -1
    best_mean_train_loss = 1000
    for epoch in range(0, config.epochs):
        siamese_dataset = SiameseDataset(
            training_csv,
            training_dir,
            idx,
            transform=transforms.Compose([
                transforms.RandomAffine(10),
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ]),
        )
        train_dataloader = DataLoader(
            siamese_dataset,
            shuffle=True,
            num_workers=config.workers,
            batch_size=config.batch_size,
        )
        train_losses = []
        for data in tqdm(train_dataloader):
            img0, img1, label = data
            img0 = img0.cuda() if device == "cuda" else img0
            img1 = img1.cuda() if device == "cuda" else img1
            label = label.cuda() if device == "cuda" else label
            optimizer.zero_grad()
            output1, output2 = net(img0, img1)
            # loss_contrastive = criterion(output1, output2, label)
            loss_contrastive = contrastiv_loss(output1, output2, label.to(device))
            train_losses.append(loss_contrastive.item())
            loss_contrastive.backward()
            optimizer.step()
        mean_train_loss = sum(train_losses) / len(train_losses)
        scheduler.step()

        filenames, whale_ids, embeddings = net.generate_embeddings(img_dir=training_dir, labels=df)

        query_filename_id = df_test

        recall_1 = recall_k(filenames, whale_ids, embeddings, transform, net,
                            training_dir, query_filename_id, device, top_n=1)
        recall_5 = recall_k(filenames, whale_ids, embeddings, transform, net,
                            training_dir, query_filename_id, device, top_n=5)
        recall_10 = recall_k(filenames, whale_ids, embeddings, transform, net,
                             training_dir, query_filename_id, device, top_n=10)
        recall_100 = recall_k(filenames, whale_ids, embeddings, transform, net,
                              training_dir, query_filename_id, device, top_n=100)
        recall_1000 = recall_k(filenames, whale_ids, embeddings, transform, net,
                               training_dir, query_filename_id, device, top_n=1000)

        writer.add_scalar("Loss/train", mean_train_loss, epoch)
        writer.add_scalar("Recall/recall@1", recall_1, epoch)
        writer.add_scalar("Recall/recall@5", recall_5, epoch)
        writer.add_scalar("Recall/recall@10", recall_10, epoch)
        writer.add_scalar("Recall_debug/recall@100", recall_100, epoch)
        writer.add_scalar("Recall_debug/recall@1000", recall_1000, epoch)

        if recall_5 > recall_5_best:
            recall_5_best = recall_5
            torch.save(net.state_dict(), f"models/contrastive/model_contrastive_best_recall_5.pt")
            with open(f"models/contrastive/embeddings_best_recall_5.pkl", 'wb') as f:
                pickle.dump([filenames, whale_ids, embeddings], f)

        if recall_10 > recall_10_best:
            recall_10_best = recall_10
            torch.save(net.state_dict(), f"models/contrastive/model_contrastive_best_recall_10.pt")
            with open(f"models/contrastive/embeddings_best_recall_10.pkl", 'wb') as f:
                pickle.dump([filenames, whale_ids, embeddings], f)

        if mean_train_loss < best_mean_train_loss:
            best_mean_train_loss = mean_train_loss
            torch.save(net.state_dict(), f"models/contrastive/model_contrastive_best.pt")
            with open(f"models/contrastive/embeddings_best.pkl", 'wb') as f:
                pickle.dump([filenames, whale_ids, embeddings], f)

        print(f"Epoch {epoch}\n Current loss {mean_train_loss}\n")
        # iteration_number += 10
        # counter.append(iteration_number)
        # loss.append(loss_contrastive.item())
    writer.close()
    return net


if __name__ == "__main__":
    # print("Reading data")
    # siamese_dataset = SiameseDataset(
    #     training_csv,
    #     training_dir,
    #     idx,
    #     transform=transforms.Compose(
    #         transforms.RandomAffine(10),
    #         [transforms.Resize((128, 128)), transforms.ToTensor()]
    #     ),
    # )
    # dataloader = DataLoader(siamese_dataset, shuffle=True, batch_size=config.batch_size)
    # dataiter = iter(dataloader)

    # example_batch = next(dataiter)
    # concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
    # imshow(torchvision.utils.make_grid(concatenated))
    # print(example_batch[2].numpy())
    net = SiameseNetwork()
    if os.path.isfile("models/contrastive/model_contrastive_best.pt"):
        net.load_state_dict(torch.load("models/contrastive/model_contrastive_best.pt"))
    net.to(device)
    # Decalre Loss Function
    criterion = ContrastiveLoss()
    # Declare Optimizer
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, verbose=True, gamma=0.9)
    # set the device to cuda

    model = train(optimizer, criterion, scheduler)

    print("Model Saved Successfully")
