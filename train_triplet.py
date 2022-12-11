import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import config
from utils import imshow
import torchvision
from torch.autograd import Variable
from PIL import Image
import PIL.ImageOps
import os
from Whael_Dataset import SiameseDataset
from Triplet_Dataset import SiameseDataset_Triplet
from Siamese_Network import SiameseNetwork
from TripletNetwork import TripletNetwork
from Contrastive_loss import ContrastiveLoss
from Triplet_loss import TripletLoss

training_dir = config.training_dir
testing_dir = config.testing_dir
training_csv_triplet = config.training_csv_triplet
testing_csv = config.testing_csv

df = pd.read_csv(training_csv_triplet)
numClasses = np.unique(df['Id'])
idx = {numClasses[i]: np.where(df['Id'] == numClasses[i])[0] for i in range(len(numClasses))}


#train the model
def train(optimizer,criterion):
    loss=[]
    counter=[]
    iteration_number = 0
    for epoch in range(1,config.epochs):
        triplet_dataset = SiameseDataset_Triplet(
            training_csv_triplet,
            training_dir,
            idx,
            transform=transforms.Compose(
                [transforms.Resize((128, 128)), transforms.ToTensor()]
            ),
        )
        train_dataloader = DataLoader(triplet_dataset,
                                      shuffle=True,
                                      num_workers=2,
                                      batch_size=config.batch_size)
        for i, data in enumerate(train_dataloader,0):
            img0, img1 , img2 = data
            img0, img1 , label = img0.cuda(), img1.cuda() , img2.cuda()
            optimizer.zero_grad()
            output1,output2,output3 = net(img0,img1,img2)
            loss_contrastive = criterion(output1,output2,output3)
            loss_contrastive.backward()
            optimizer.step()
        print("Epoch {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
        iteration_number += 10
        counter.append(iteration_number)
        loss.append(loss_contrastive.item())
    #show_plot(counter, loss)
    return net


if __name__=='__main__':
    #breakpoint()

    triplet_dataset = SiameseDataset_Triplet(
        training_csv_triplet,
        training_dir,
        idx,
        transform=transforms.Compose(
            [transforms.Resize((128, 128)), transforms.ToTensor()]
        ),
    )

    dataloader = DataLoader(triplet_dataset, shuffle=True, batch_size=8)
    dataiter = iter(dataloader)

    example_batch = next(dataiter)
    concatenated = torch.cat((example_batch[0], example_batch[1],example_batch[2]), 0)
    imshow(torchvision.utils.make_grid(concatenated))

    # Declare Siamese Network

    net=TripletNetwork().cuda()
    # Decalre Loss Function
    criterion =TripletLoss()
    # Declare Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0005)
    #set the device to cuda


    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train(optimizer,criterion)
    torch.save(model.state_dict(), "model_triplet.pt")
    print("Model Saved Successfully")