from tensorflow.keras.datasets import mnist
import numpy as np
import pandas as pd
import csv

def save_csv(imgs,labels,path='contrastive_train.csv'):
    with open(path, mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
        for i in range(len(imgs)):
            file_writer.writerow([imgs[i][0], imgs[i][1], labels[i][0]])

def make_pairs(images, labels):
    # initialize two empty lists to hold the (image, image) pairs and
    # labels to indicate if a pair is positive or negative
    pairImages = []
    pairLabels = []
    # calculate the total number of classes present in the dataset
    # and then build a list of indexes for each class label that
    # provides the indexes for all examples with a given label
    numClasses = np.unique(labels)
    idx = {numClasses[i]:np.where(labels == numClasses[i])[0] for i in range(len(numClasses))}
    # loop over all images
    for idxA in range(len(images)):
        # grab the current image and label belonging to the current
        # iteration
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
        pairLabels.append([0])
    # return a 2-tuple of our image pairs and labels
    return (np.array(pairImages), np.array(pairLabels))

df=pd.read_csv('train.csv')
#breakpoint()


(pairTrain, labelTrain) = make_pairs(df['Image'], df['Id'])
save_csv(pairTrain,labelTrain)
