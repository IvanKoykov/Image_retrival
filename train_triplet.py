import pickle
import os.path
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import torchvision.utils
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from Triplet_Dataset import SiameseDataset_Triplet
from Triplet_loss import TripletLoss
from TripletNetwork import TripletNetwork
from utils import imshow, recall_k

training_dir = config.images_dir
training_csv_triplet = config.training_csv_triplet
testing_csv = config.testing_csv

df = pd.read_csv(training_csv_triplet)
df_test = pd.read_csv(testing_csv)
numClasses = np.unique(df["Id"])

idx = {
    numClasses[i]: np.where(df["Id"] == numClasses[i])[0]
    for i in range(len(numClasses))
}

transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
writer = SummaryWriter("logs/triplet", comment="triplet_loss")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# train the model
def train(optimizer, criterion, scheduler):
    # loss = []
    # counter = []
    # iteration_number = 0
    recall_5_best = -1
    recall_10_best = -1
    best_mean_train_loss = 1000
    for epoch in range(0, config.epochs):
        triplet_dataset = SiameseDataset_Triplet(
            training_csv_triplet,
            training_dir,
            idx,
            transform=transforms.Compose([
                # transforms.RandomAffine(10),
                transforms.Resize((128, 128)),
                transforms.ToTensor()
            ]),
        )
        train_dataloader = DataLoader(
            triplet_dataset,
            shuffle=True,
            num_workers=config.workers,
            batch_size=config.batch_size,
        )
        train_losses = []
        for data in tqdm(train_dataloader):
            img0, img1, img2 = data
            img0 = img0.cuda() if device == "cuda" else img0
            img1 = img1.cuda() if device == "cuda" else img1
            img2 = img2.cuda() if device == "cuda" else img2
            optimizer.zero_grad()
            output1, output2, output3 = net(img0, img1, img2)
            loss_triplet = criterion(output1, output2, output3)
            train_losses.append(loss_triplet.item())
            loss_triplet.backward()
            optimizer.step()

        mean_train_loss = sum(train_losses) / len(train_losses)
        scheduler.step(mean_train_loss)
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
            torch.save(net.state_dict(), f"models/triplet/model_triplet_best_recall_5.pt")
            with open(f"models/triplet/embeddings_best_recall_5.pkl", 'wb') as f:
                pickle.dump([filenames, whale_ids, embeddings], f)

        if recall_10 > recall_10_best:
            recall_10_best = recall_10
            torch.save(net.state_dict(), f"models/triplet/model_triplet_best_recall_10.pt")
            with open(f"models/triplet/embeddings_best_recall_10.pkl", 'wb') as f:
                pickle.dump([filenames, whale_ids, embeddings], f)

        if mean_train_loss < best_mean_train_loss:
            best_mean_train_loss = mean_train_loss
            torch.save(net.state_dict(), f"models/triplet/model_triplet_best.pt")
            with open(f"models/triplet/embeddings_best.pkl", 'wb') as f:
                pickle.dump([filenames, whale_ids, embeddings], f)

        print(f"Epoch {epoch}\n Current loss {mean_train_loss}\n")
        # iteration_number += 10
        # counter.append(iteration_number)
        # loss.append(loss_contrastive.item())
    writer.close()
    return net


if __name__ == "__main__":
    print("Reading data")

    # Declare Siamese Network

    net = TripletNetwork()
    if os.path.isfile("models/triplet/model_triplet_best.pt"):
        net.load_state_dict(torch.load("models/triplet/model_triplet_best.pt"))
    net.to(device)
    # Decalre Loss Function
    criterion = TripletLoss()
    # Declare Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=5, eps=1e-5)
    # set the device to cuda

    model = train(optimizer, criterion, scheduler)
    print("Model Saved Successfully")
