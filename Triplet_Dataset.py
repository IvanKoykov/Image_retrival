import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# preprocessing and loading the dataset
class SiameseDataset_Triplet(Dataset):
    def __init__(self, training_csv=None, training_dir=None, idx=None, transform=None):
        # used to prepare the labels and images path
        self.train_df = pd.read_csv(training_csv)
        # self.train_df.columns =["anchor","imagePos","imageNeg"]
        self.train_dir = training_dir
        self.transform = transform
        self.idx = idx
        self.pairTrain = make_pairs(
            self.train_df["Image"], self.train_df["Id"], self.idx
        )

    def __getitem__(self, index):
        # getting the image path
        anchor_path = os.path.join(self.train_dir, self.pairTrain[index, 0])
        imagePos_path = os.path.join(self.train_dir, self.pairTrain[index, 1])
        imageNeg_path = os.path.join(self.train_dir, self.pairTrain[index, 2])
        # breakpoint()
        # Loading the image
        img0 = Image.open(anchor_path)
        img1 = Image.open(imagePos_path)
        img2 = Image.open(imageNeg_path)
        img0 = img0.convert("L")
        img1 = img1.convert("L")
        img2 = img2.convert("L")
        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img0, img1, img2

    def __len__(self):
        return len(self.pairTrain)


def make_pairs(images, labels, idx):
    # initialize two empty lists to hold the (image, image) pairs and
    # labels to indicate if a pair is positive or negative
    tripletImages = []
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

        negIdx = np.where(labels != label)[0]
        negImage = images[np.random.choice(negIdx)]
        # prepare a positive pair and update the images and labels
        # lists, respectively
        tripletImages.append([currentImage, posImage, negImage])
        # grab the indices for each of the class labels *not* equal to
        # the current label and randomly pick an image corresponding
        # to a label *not* equal to the current label
        # prepare a negative pair of images and update our lists
        # pairImages.append([currentImage, negImage])
        # pairLabels.append([0])
    # return a 2-tuple of our image pairs and labels
    return np.array(tripletImages)
