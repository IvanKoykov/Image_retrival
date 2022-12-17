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
from utils import imshow
from Whael_Dataset import SiameseDataset

training_dir = config.training_dir
testing_dir = config.testing_dir
training_csv = config.training_csv_contrastive
testing_csv = config.testing_csv
df = pd.read_csv(training_csv)
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
        siamese_dataset = SiameseDataset(
            training_csv,
            training_dir,
            idx,
            transform=transforms.Compose(
                [transforms.Resize((128, 128)), transforms.ToTensor()]
            ),
        )
        train_dataloader = DataLoader(
            siamese_dataset,
            shuffle=True,
            num_workers=config.workers,
            batch_size=config.batch_size,
        )
        for data in tqdm(train_dataloader):
            img0, img1, label = data
            img0 = img0.cuda() if device == "cuda" else img0
            img1 = img1.cuda() if device == "cuda" else img1
            label = label.cuda() if device == "cuda" else label
            optimizer.zero_grad()
            output1, output2 = net(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
        print("Epoch {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
        iteration_number += 10
        counter.append(iteration_number)
        loss.append(loss_contrastive.item())
    return net


if __name__ == "__main__":
    print("Reading data")
    siamese_dataset = SiameseDataset(
        training_csv,
        training_dir,
        idx,
        transform=transforms.Compose(
            [transforms.Resize((128, 128)), transforms.ToTensor()]
        ),
    )
    dataloader = DataLoader(siamese_dataset, shuffle=True, batch_size=config.batch_size)
    dataiter = iter(dataloader)

    example_batch = next(dataiter)
    concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
    imshow(torchvision.utils.make_grid(concatenated))
    # print(example_batch[2].numpy())
    net = SiameseNetwork()
    net.to(device)
    # Decalre Loss Function
    criterion = ContrastiveLoss()
    # Declare Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0005)
    # set the device to cuda

    model = train(optimizer, criterion)
    torch.save(model.to("cpu").state_dict(), "model_contrastive.pt")
    print("Model Saved Successfully")
