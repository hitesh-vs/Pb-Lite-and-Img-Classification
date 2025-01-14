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

import cv2
import os
import sys
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm.notebook import tqdm
import torch
from Network.Network import CIFAR10Model
from Misc.MiscUtils import *
from Misc.DataUtils import *
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image



# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupAll():
    """
    Outputs:
    ImageSize - Size of the Image
    """   
    # Image Input Shape
    ImageSize = [32, 32, 3]

    return ImageSize

def StandardizeInputs(Img):
    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    ##########################################################################
    return Img
    
def ReadImages(Img):
    """
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """    
    I1 = Img
    
    if(I1 is None):
        # OpenCV returns empty list if image is not read! 
        print('ERROR: Image I1 cannot be read')
        sys.exit()
        
    I1S = StandardizeInputs(np.float32(I1))

    I1Combined = np.expand_dims(I1S, axis=0)

    return I1Combined, I1
                

def Accuracy(Pred, GT):
    """
    Inputs: 
    Pred are the predicted labels
    GT are the ground truth labels
    Outputs:
    Accuracy in percentage
    """
    return (np.sum(np.array(Pred)==np.array(GT))*100.0/len(Pred))

def ReadLabels(LabelsPathTest, LabelsPathPred):
    if(not (os.path.isfile(LabelsPathTest))):
        print('ERROR: Test Labels do not exist in '+LabelsPathTest)
        sys.exit()
    else:
        LabelTest = open(LabelsPathTest, 'r')
        LabelTest = LabelTest.read()
        LabelTest = map(float, LabelTest.split())

    if(not (os.path.isfile(LabelsPathPred))):
        print('ERROR: Pred Labels do not exist in '+LabelsPathPred)
        sys.exit()
    else:
        LabelPred = open(LabelsPathPred, 'r')
        LabelPred = LabelPred.read()
        LabelPred = map(float, LabelPred.split())
        
    return LabelTest, LabelPred

def ConfusionMatrix(LabelsTrue, LabelsPred):
    """
    LabelsTrue - True labels
    LabelsPred - Predicted labels
    """
    # Convert inputs to lists if needed
    LabelsTrue, LabelsPred = list(LabelsTrue), list(LabelsPred)

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=LabelsTrue, y_pred=LabelsPred)

    # Print the confusion matrix as text.
    for i in range(len(cm)):
        print(str(cm[i, :]) + ' ({0})'.format(i))

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(len(cm))]
    print("".join(class_numbers))

    print('Accuracy: ' + str(Accuracy(LabelsPred, LabelsTrue)), '%')

    # Plot the confusion matrix as a figure
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix", fontsize=16)
    #plt.colorbar()

    # Add labels to axes
    tick_marks = np.arange(len(cm))
    plt.xticks(tick_marks, tick_marks, rotation=45, fontsize=12)
    plt.yticks(tick_marks, tick_marks, fontsize=12)

    # Add values to the confusion matrix cells
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=12)

    # Add axis labels
    plt.ylabel('True Labels', fontsize=14)
    plt.xlabel('Predicted Labels', fontsize=14)
    plt.tight_layout()

    # Save and show the figure
    plt.savefig('confusion_matrix.png')  # Optional: Save the figure to a file
    plt.show()


def TestOperation(ImageSize, ModelPath, TestSet, LabelsPathPred):
    """
    Inputs: 
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    TestSet - The test dataset
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to /content/data/TxtFiles/PredOut.txt
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(LabelsPathPred)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # Predict output with forward pass, MiniBatchSize for Test is 1
    model = CIFAR10Model(InputSize=3*32*32,OutputSize=10) 

    input_names = ["Images from CIFAR-10"]
    output_names = ["Predicted Class"]

    dummy_input = torch.randn(1, 3, 32, 32)

    torch.onnx.export(model, dummy_input, "model.onnx", input_names=input_names, output_names=output_names)
    
    CheckPoint = torch.load(ModelPath)
    model.load_state_dict(CheckPoint['model_state_dict'])
    print('Number of parameters in this model are %d ' % len(model.state_dict().items()))
    model.eval()
    
    OutSaveT = open(LabelsPathPred, 'w')

    for count in tqdm(range(len(TestSet))): 
        Img, Label = TestSet[count]
        Img, ImgOrg = ReadImages(Img)
         # Convert numpy array to PyTorch Tensor
        ImgTensor = torch.from_numpy(Img)

        # Predict the label
        with torch.no_grad():  # Disable gradient computation for inference
            output = model(ImgTensor)
            _, PredT = torch.max(output, 1)  # Use torch.max instead of torch.argmax
            PredT = PredT.item()  # Convert Tensor to Python int for saving
        
        OutSaveT.write(str(PredT)+'\n')

    OutSaveT.close()

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
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelPath', dest='ModelPath', default='./Checkpoints/49model.ckpt', help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--LabelsPath', dest='LabelsPath', default='./Code/TxtFiles/LabelsTest.txt', help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    LabelsPath = Args.LabelsPath 
    #LabelsPath = './Code/TxtFiles/LabelsTrain.txt' # For train data
    #TestSet = torchvision.datasets.CIFAR10(root='data/', train=False)

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    test_transforms = transforms.Compose([
    #transforms.Resize((32, 32)),
    transforms.ToTensor(),
    #normalize  # Use the same normalization as training
    ])


    TestSet = ImageDataset(img_folder="./CIFAR10/Test", label_file=LabelsPath, transform=test_transforms, is_train=False)

    # Override the labels with dummy ones
    #TestSet.samples = [(path, 5) for path, _ in TestSet.samples]  # Set dummy labels (all 0)


    # Setup all needed parameters including file reading
    ImageSize = SetupAll()

    # Define PlaceHolder variables for Predicted output
    LabelsPathPred = './TxtFiles/PredOut.txt' # Path to save predicted labels

    TestOperation(ImageSize, ModelPath, TestSet, LabelsPathPred)

    # Plot Confusion Matrix
    LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)
    ConfusionMatrix(LabelsTrue, LabelsPred)

    
     
if __name__ == '__main__':
    main()
 
