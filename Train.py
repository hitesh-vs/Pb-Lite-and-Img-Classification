#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code

Colab file can be found at:
    https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)


import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW
from torchvision.datasets import CIFAR10
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import time
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm.notebook import tqdm
#import Misc.ImageUtils as iu
from Network.Network import CIFAR10Model
from Misc.MiscUtils import *
from Misc.DataUtils import *
#from sklearn.metrics import confusion_matrix
import Test 
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import DataLoader


# Don't generate pyc codes
sys.dont_write_bytecode = True

    
def GenerateBatch(TrainSet, TrainLabels, ImageSize, MiniBatchSize):
    """
    Inputs: 
    TrainSet - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainLabels - Labels corresponding to Train
    NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize is the Size of the Image
    MiniBatchSize is the size of the MiniBatch
   
    Outputs:
    I1Batch - Batch of images
    LabelBatch - Batch of one-hot encoded labels 
    """
    I1Batch = []
    LabelBatch = []
    
    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(TrainSet)-1)
        
        ImageNum += 1
        
          ##########################################################
          # Add any standardization or data augmentation here!
          ##########################################################
        

        I1, Label = TrainSet[RandIdx]

        # Append All Images and Mask
        I1Batch.append(I1)
        LabelBatch.append(torch.tensor(Label))
        
    return torch.stack(I1Batch), torch.stack(LabelBatch)


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Factor of reduction in training data is ' + str(DivTrain))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile)   


def evaluate_model(model, test_dataset, transform=ToTensor()):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for img, label in test_dataset:
            # Load and preprocess the image
            #img = Image.open(img_path).convert('RGB')
            img = img.unsqueeze(0)  # Add batch dimension

            # Forward pass
            output = model(img)
            _, predicted = torch.max(output, 1)

            # Compare prediction to ground truth
            total += 1
            correct += (predicted.item() == label)

    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def TrainOperation(TrainLabels, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, TrainSet, LogsPath, TestSet):
    model = CIFAR10Model(InputSize=3*32*32, OutputSize=10)
    Optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + '.ckpt')
        StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
        model.load_state_dict(CheckPoint['model_state_dict'])
        print('Loaded latest checkpoint with the name ' + LatestFile + '....')
    else:
        StartEpoch = 0
        print('New model initialized....')

    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        print(f"Starting epoch {Epochs + 1} of {NumEpochs}")
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)

        epoch_loss = 0.0
        epoch_acc = 0.0

        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            Batch = GenerateBatch(TrainSet, TrainLabels, ImageSize, MiniBatchSize)
            LossThisBatch = model.training_step(Batch)
            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()

            result = model.validation_step(Batch)
            epoch_loss += result["loss"]
            epoch_acc += result["acc"]
            #Writer.add_scalar('LossEveryIter', result["loss"], Epochs * NumIterationsPerEpoch + PerEpochCounter)
            #Writer.add_scalar('Accuracy', result["acc"], Epochs * NumIterationsPerEpoch + PerEpochCounter)
            print(f'Iter:{PerEpochCounter}')


        epoch_loss /= NumIterationsPerEpoch
        epoch_acc /= NumIterationsPerEpoch
        Writer.add_scalar('Loss/Epoch', epoch_loss, Epochs)
        Writer.add_scalar('Accuracy/Epoch', epoch_acc, Epochs)
        Writer.flush()
        print(f'Train Accuracy:{epoch_acc}')

        # Evaluate on test data
        test_accuracy = evaluate_model(model, TestSet)
        Writer.add_scalar('Test Accuracy/Epoch', test_accuracy, Epochs)
        Writer.flush()


        SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
        torch.save({'epoch': Epochs, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': Optimizer.state_dict(), 'loss': LossThisBatch}, SaveName)
        print('\n' + SaveName + ' Model Saved...')

class ImageDataset(Dataset):
    def __init__(self, img_folder, label_file, transform=None,is_train=True):
        self.img_folder = img_folder
        self.transform = transform
        self.is_train = is_train
        
        # Load labels from the text file
        with open(label_file, 'r') as file:
            lines = file.readlines()
        
        # Assuming the lines contain only class numbers, not filenames
        self.labels = [int(line.strip()) for line in lines]
        # print(self.labels) # list of all labels
        
        # Create image paths by appending labels to the img_folder
        self.img_paths = [os.path.join(img_folder, f"{idx + 1}.png") for idx in range(len(self.labels))]  # Adjust the image naming convention if necessary
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        img_path = os.path.normpath(img_path)


        # Open image and apply transformations if any
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # if self.is_train:
        #     image = self.train_transform(image)
        # else:
        #     image = self.test_transform(image)


        return image, label

        

def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--CheckPointPath', default='../Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
    Parser.add_argument('--NumEpochs', type=int, default=15, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=500, help='Size of the MiniBatch to use, Default:1')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='Logs/', help='Path to save Logs for Tensorboard, Default=Logs/')
    #TrainSet = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=ToTensor())
    
    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath

    basePath = 'Phase2/Code'

    train_img_folder = "./CIFAR10/Train"
    train_label_file = "./TxtFiles/LabelsTrain.txt"
    test_img_folder = "./CIFAR10/Test"
    test_label_file = "./TxtFiles/LabelsTest.txt"

    transform = transforms.Compose([
    #transforms.Resize((128, 128)),  
    transforms.ToTensor(),         # Convert to PyTorch tensor
    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])

    # Create training and test datasets
    TrainSet = ImageDataset(img_folder=train_img_folder, label_file=train_label_file, transform=transform, is_train=True)
    test_dataset = ImageDataset(img_folder=test_img_folder, label_file=test_label_file, transform=transform, is_train=False)

    # Create DataLoaders for training and testing
    #TrainSet = DataLoader(train_dataset, batch_size=500, shuffle=True)
    #est_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Setup all needed parameters including file reading
    DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(basePath, CheckPointPath)



    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None
    
    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    TrainOperation(TrainLabels, NumTrainSamples, ImageSize,
                NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                DivTrain, LatestFile, TrainSet, LogsPath,test_dataset)

    
if __name__ == '__main__':
    main()
 
