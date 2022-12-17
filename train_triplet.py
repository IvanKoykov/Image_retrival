import pickle

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
from utils import imshow

training_dir = config.images_dir
training_csv_triplet = config.training_csv_triplet

df = pd.read_csv(training_csv_triplet)
numClasses = np.unique(df["Id"])
idx = {
    numClasses[i]: np.where(df["Id"] == numClasses[i])[0]
    for i in range(len(numClasses))
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# train the model
def train(optimizer, criterion):
    loss = []
    counter = []
    iteration_number = 0
    for epoch in range(0, config.epochs):
        triplet_dataset = SiameseDataset_Triplet(
            training_csv_triplet,
            training_dir,
            idx,
            transform=transforms.Compose(
                [transforms.Resize((128, 128)), transforms.ToTensor()]
            ),
        )
        train_dataloader = DataLoader(
            triplet_dataset,
            shuffle=True,
            num_workers=config.workers,
            batch_size=config.batch_size,
        )
        for data in tqdm(train_dataloader):
            img0, img1, img2 = data
            img0 = img0.cuda() if device == "cuda" else img0
            img1 = img1.cuda() if device == "cuda" else img1
            img2 = img2.cuda() if device == "cuda" else img2
            optimizer.zero_grad()
            output1, output2, output3 = net(img0, img1, img2)
            loss_triplet = criterion(output1, output2, output3)
            loss_triplet.backward()
            optimizer.step()

        print("Epoch {}\n Current loss {}\n".format(epoch, loss_triplet.item()))
        iteration_number += 10
        counter.append(iteration_number)
        loss.append(loss_triplet.item())
    return net


if __name__ == "__main__":
    print("Reading data")
    triplet_dataset = SiameseDataset_Triplet(
        training_csv_triplet,
        training_dir,
        idx,
        transform=transforms.Compose(
            [transforms.Resize((128, 128)), transforms.ToTensor()]
        ),
    )

    dataloader = DataLoader(triplet_dataset, shuffle=True, batch_size=config.batch_size)
    dataiter = iter(dataloader)

    example_batch = next(dataiter)
    concatenated = torch.cat((example_batch[0], example_batch[1], example_batch[2]), 0)
    imshow(torchvision.utils.make_grid(concatenated))

    # Declare Siamese Network

    net = TripletNetwork()
    net.to(device)
    # Decalre Loss Function
    criterion = TripletLoss()
    # Declare Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0005)
    # set the device to cuda

    model = train(optimizer, criterion)
    df = df.drop(df[df["Id"] == "new_whale"].index)
    embeddings = model.generate_embeddings(img_dir=training_dir, labels=df)
    with open('database_triplet.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
    torch.save(model.to("cpu").state_dict(), "model_triplet.pt")
    print("Model Saved Successfully")
