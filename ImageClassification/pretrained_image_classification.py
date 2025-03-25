#Python3 Shebang
#########################################################################################################################################################################################

#importing dataset
#CODE import opendatasets as od
#CODE od.download("https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification", "Pytorch Projects\\ImageClassification") #Binary type dataset. Either says it is Jasmin or Gonan rice

#import torch packages: Main Framework, NN for layers, Adam Optimizer, Dataset Class and Dataloader for creating objects
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

#import torch vision packages: Transform to process images
import torchvision.transforms as transforms
from torchvision import models # import pretrained models in PyTorch library

#import Sci-Kit Learn Packages: Label Encoder
from sklearn.preprocessing import LabelEncoder

#import Matplotlib packages: Plotting training progress
import matplotlib.pyplot as plt

#import Pandas packages: Data reading and preprocessing
import pandas as pd

#import Numpy
import numpy as np

#import PIL Package: Image to view image
from  PIL import Image

#other imports
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu' # detect the GPU if any, if not use CPU, change cuda to mps if you have a mac

PRINTSTEPS = True #Note Set True to print midsteps

def Print(*args):
    if PRINTSTEPS:
        print(args)

#########################################################################################################################################################################################
#$ Reading Data Paths

train_ds = pd.read_csv("Pytorch Projects\\ImageClassification\\bean-leaf-lesions-classification\\train.csv")
val_ds = pd.read_csv("Pytorch Projects\\ImageClassification\\bean-leaf-lesions-classification\\val.csv")

dataset = pd.concat([train_ds, val_ds], ignore_index=True)

dataset["image:FILE"] = "Pytorch Projects\\ImageClassification\\bean-leaf-lesions-classification\\" + dataset["image:FILE"]

Print("Data Shape is: ", dataset.head())

#########################################################################################################################################################################################
#$ Data Inspection

Print("Classes are: ")
Print(dataset["category"].unique())
Print("Classes ditrubution are: ")
Print(dataset["category"].value_counts())

#########################################################################################################################################################################################
#$ Data Splitting

train_data = dataset.sample(frac=0.7, random_state=7)
test_data = dataset.drop(train_data.index)

#########################################################################################################################################################################################
#$ Data Preprocessing

label_encoder = LabelEncoder()

_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])

#########################################################################################################################################################################################
#$ DataSet Class

class BeanWeightDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.labels = torch.tensor(label_encoder.fit_transform(df['category'])).to(device)
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        img_path = self.df.iloc[index,0]
        label = self.labels[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = (self.transform(image)/255).to(device)
        
        return image, label
    
#########################################################################################################################################################################################
#$ Dataset Objects

train_dataset = BeanWeightDataset(df= train_data, transform=_transform)
test_dataset = BeanWeightDataset(df= test_data, transform=_transform)

#########################################################################################################################################################################################
#$ Data Preprocessing

if PRINTSTEPS:
    n_rows = 3
    n_cols = 3
    f, axarr = plt.subplots(n_rows, n_cols)
    for row in range(n_rows):
        for col in range(n_cols):
            image = train_dataset[np.random.randint(0,train_dataset.__len__())][0].cpu()
            axarr[row, col].imshow((image*255).squeeze().permute(1,2,0))
            axarr[row, col].axis('off')

    plt.tight_layout()
    plt.show()
    
#########################################################################################################################################################################################
#$ HyperParameters

LEARNING_RATE = 1e-3
BATCH_SIZE = 16
EPOCHS = 15

#########################################################################################################################################################################################
#$ Dataloaders

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

#########################################################################################################################################################################################
#$ Pretrained Generic Models

googlenet_model = models.googlenet(weights='DEFAULT')
for param in googlenet_model.parameters():
    param.requires_grad = True

Print(googlenet_model.fc)

num_classes = len(dataset["category"].unique())
googlenet_model.fc = nn.Linear(googlenet_model.fc.in_features, num_classes)
googlenet_model.to(device)

#########################################################################################################################################################################################
#$ Loss and Optimization

criterion = nn.CrossEntropyLoss()
optimizer = Adam(googlenet_model.parameters(), lr = LEARNING_RATE)

#########################################################################################################################################################################################
#$ Model training

total_loss_train_plot = []
total_acc_train_plot = []

for epoch in range(EPOCHS):
    total_acc_train = 0
    total_loss_train = 0
    
    for inputs, labels in train_loader:
        #NOTE Reset the gradients for the next batch
        optimizer.zero_grad()
        
        #NOTE Pass the inputs through the model to get the predictions
        #NOTE We use the squeeze function to modify the shape of the output and make it returns a tensor with all specified dimensions of input of size 1 removed. 
        outputs = googlenet_model(inputs)
        
        #NOTE Calculate the loss between the predictions and labels using the binary cross-entropy loss function
        batch_loss = criterion(outputs, labels)
        
        #NOTE Accumulate the total loss for the epoch
        total_loss_train += batch_loss.item()
        
        #NOTE Backpropagate the loss to update the model parameters
        batch_loss.backward()
        
        #NOTE Calculate the accuracy of the predictions by comparing them to the labels.
        #NOTE DIFFERENT FROM TABULAR because more then 1 class which means we need to cross compare the class data
        acc = (torch.argmax(outputs, axis = 1) == labels).sum().item()
        
        #NOTE Accumulate the total accuracy for the epoch
        total_acc_train += acc
        
        #NOTE Update the model parameters using the Adam optimizer
        optimizer.step() 
            
    total_loss_train_plot.append(round(total_loss_train/1000,4))
    total_acc_train_plot.append(round(total_acc_train/(train_dataset.__len__())*100,4))
    
    Print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {round(total_loss_train/100, 4)} Train Accuracy {round((total_acc_train)/train_dataset.__len__() * 100, 4)}%')

#########################################################################################################################################################################################
#$ Testing
    
with torch.no_grad():
    total_loss_test = 0
    total_acc_test = 0
    
    for index, (input, labels) in enumerate(test_loader):
        predictions = googlenet_model(input)
        
        acc = (torch.argmax(predictions, axis = 1) == labels).sum().item()
        
        total_acc_test += acc
        
Print(f"Accuracy Score is: {round((total_acc_test/test_dataset.__len__())*100, 2)}%")

#########################################################################################################################################################################################
#$ Plotting

if PRINTSTEPS:
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    axs[0].plot(total_loss_train_plot, label='Training Loss')
    axs[0].set_title('Training Loss over Epochs')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[1].set_ylim([0, 2])
    axs[0].legend()

    axs[1].plot(total_acc_train_plot, label='Training Accuracy')
    axs[1].set_title('Training Accuracy over Epochs')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_ylim([0, 100])
    axs[1].legend()

    plt.tight_layout()

    plt.show()

#########################################################################################################################################################################################
#$ Using the Model

def predict_image(image_path):
    image = _transform(Image.open(image_path).convert('RGB')).to(device)
    
    output = torch.argmax(googlenet_model(image.unsqueeze(0)), axis = 1).item()
    return label_encoder.inverse_transform([output])

image = Image.open("Pytorch Projects\\ImageClassification\\predictiontest\\leaf1.jpeg")
plt.imshow(image)
plt.show()

Print("Prediction for the image is: ", predict_image("Pytorch Projects\\ImageClassification\\predictiontest\\leaf1.jpeg"))