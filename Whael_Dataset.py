import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# preprocessing and loading the dataset
class SiameseDataset(Dataset):
    def __init__(self, training_csv=None, training_dir=None, idx=None, transform=None):
        # used to prepare the labels and images path
        self.train_df = pd.read_csv(training_csv)
        # self.train_df.columns =["Image","ID"]
        self.train_dir = training_dir
        self.transform = transform
        self.idx = idx
        self.pairTrain, self.labelTrain = make_pairs(
            self.train_df["Image"], self.train_df["Id"], self.idx
        )

    # index 8 is out  of  bounds  for axis 0 with size 8
    def __getitem__(self, index):
        # getting the image path
        # breakpoint()
        image1_path = os.path.join(self.train_dir, self.pairTrain[index, 0])
        image2_path = os.path.join(self.train_dir, self.pairTrain[index, 1])

        # breakpoint()
        # Loading the image
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        img0 = img0.convert("RGB")
        img1 = img1.convert("RGB")
        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        # breakpoint()
        return (
            img0,
            img1,
            torch.from_numpy(np.array([self.labelTrain[index, 0]], dtype=np.float32)),
        )

    def __len__(self):
        return len(self.train_df)


def make_pairs(images, labels, idx):
    # initialize two empty lists to hold the (image, image) pairs and
    # labels to indicate if a pair is positive or negative
    pairImages = []
    pairLabels = []
    # calculate the total number of classes present in the dataset
    # and then build a list of indexes for each class label that
    # provides the indexes for all examples with a given label
    # loop over all images
    for idxA in range(len(images)):
        # grab the current image and label belonging to the current
        # iteration
        if labels[idxA] == "new_whale":
            continue
        currentImage = images[idxA]
        label = labels[idxA]
        # randomly pick an image that belongs to the *same* class
        # label
        idxB = np.random.choice(idx[label])
        posImage = images[idxB]
        # prepare a positive pair and update the images and labels
        # lists, respectively
        pairImages.append([currentImage, posImage])
        pairLabels.append([1])
        # grab the indices for each of the class labels *not* equal to
        # the current label and randomly pick an image corresponding
        # to a label *not* equal to the current label
        negIdx = np.where(labels != label)[0]
        negImage = images[np.random.choice(negIdx)]
        # prepare a negative pair of images and update our lists
        pairImages.append([currentImage, negImage])
        #Возможно надо ыернуть на 0
        pairLabels.append([-1])
    # return a 2-tuple of our image pairs and labels
    return np.array(pairImages), np.array(pairLabels)
