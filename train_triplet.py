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

training_dir = config.training_dir
testing_dir = config.testing_dir
training_csv_triplet = config.training_csv_triplet
testing_csv = config.testing_csv

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
    for epoch in range(1, config.epochs):
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
            optimizer.zero_grad()
            output1, output2, output3 = net(img0, img1, img2)
            loss_contrastive = criterion(output1, output2, output3)
            loss_contrastive.backward()
            optimizer.step()
        print("Epoch {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
        iteration_number += 10
        counter.append(iteration_number)
        loss.append(loss_contrastive.item())
    return net


if __name__ == "__main__":

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
    torch.save(model.to("cpu").state_dict(), "model_triplet.pt")
    print("Model Saved Successfully")
