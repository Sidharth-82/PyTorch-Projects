#Python3 Shebang
#########################################################################################################################################################################################

#importing dataset
#CODE import opendatasets as od
#CODE od.download("https://www.kaggle.com/datasets/andrewmvd/animal-faces", "Pytorch Projects\\ImageClassification")

#import torch packages: Main Framework, NN for layers, Adam Optimizer, Dataset Class and Dataloader for creating objects
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

#import torch vision packages: Transform to process images
import torchvision.transforms as transforms

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

image_path = []
labels = []

IMAGE_PATHS = "image_paths"
LABELS = "labels"

for i in os.listdir("Pytorch Projects\\ImageClassification\\animal-faces\\afhq"):
    for label in os.listdir(f"Pytorch Projects\\ImageClassification\\animal-faces\\afhq\\{i}"):
        for image in os.listdir(f"Pytorch Projects\\ImageClassification\\animal-faces\\afhq\\{i}\\{label}"):
            labels.append(label)
            image_path.append(f"Pytorch Projects\\ImageClassification\\animal-faces\\afhq\\{i}\\{label}\\{image}")
            
dataset = pd.DataFrame(zip(image_path, labels), columns=[IMAGE_PATHS, LABELS])
Print(f"initial dataframe: {dataset.head()}")

#########################################################################################################################################################################################
#$ Data Splitting

TRAIN_SPLIT = 0.7
VALIDATION_SPLIT_TEST_SPLIT = 0.5


train = dataset.sample(frac=TRAIN_SPLIT, random_state=7)
test = dataset.drop(train.index).sample(frac=VALIDATION_SPLIT_TEST_SPLIT, random_state=7)
validation = dataset.drop(train.index).drop(test.index)

Print("Training set is: ", train.shape[0], " rows which is ", round(train.shape[0]/dataset.shape[0],4)*100, "%") # Print training shape
Print("Validation set is: ",validation.shape[0], " rows which is ", round(validation.shape[0]/dataset.shape[0],4)*100, "%") # Print validation shape
Print("Testing set is: ",test.shape[0], " rows which is ", round(test.shape[0]/dataset.shape[0],4)*100, "%") # Print testing shape

#########################################################################################################################################################################################
#$ Data Preprocessing

label_encoder = LabelEncoder()
label_encoder.fit(dataset['labels'])

_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])

#########################################################################################################################################################################################
#$ Custom Dataset Class

class AnimalFacesDataset(Dataset):
    def __init__(self, data_frame, transform=None):
        self.dataframe =  data_frame
        self.transform = transform
        self.labels = torch.tensor(label_encoder.transform(data_frame[LABELS])).to(device)
        
    def __len__(self):
        return self.dataframe.shape[0]
    
    def __getitem__(self, index):
        img_path = self.dataframe.iloc[index, 0]
        label = self.labels[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image).to(device)
            
        return image, label
    
#########################################################################################################################################################################################
#$ Dataset Objects

train_dataset = AnimalFacesDataset(data_frame=train, transform=_transform)
validation_dataset = AnimalFacesDataset(data_frame=validation, transform=_transform)
test_dataset = AnimalFacesDataset(data_frame=test, transform=_transform)

#########################################################################################################################################################################################
#$ Visualize Images


        
if PRINTSTEPS:
    n_rows = 3
    n_cols = 3

    f, axarr = plt.subplots(n_rows, n_cols)

    for row in range(n_rows):
        for col in range(n_cols):
            image = Image.open(dataset.sample(n=1)[IMAGE_PATHS].iloc[0]).convert("RGB")
            axarr[row,col].imshow(image)
            axarr[row,col].axis('off')
    
    plt.show()
    
#########################################################################################################################################################################################
#$ HyperParameters

LEARNING_RATE = 1e-4
BATCH_SIZE = 32
EPOCHS = 10

#########################################################################################################################################################################################
#$ Data Loaders

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

#########################################################################################################################################################################################
#$ Model Class

class AnimalFacesModel(nn.Module):
    def __init__(self, data_df):
        super(AnimalFacesModel, self).__init__()
        
        self.df = data_df
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3,padding=1)
        self.pooling = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear((128*16*16), 128)
        self.output = nn.Linear(128, len(self.df[LABELS].unique()))
        
    def forward(self, x): #NOTE must be named forward otherwise override wont recognize (self, x)
        #CODE Print(f"Forward Propagation Input Shape: {x.shape}, First few lines: {x}")
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.relu(x)
        #CODE Print(f"Layer 1 Shape: {x.shape}, First few lines: {x}")
        x = self.conv2(x)
        x = self.pooling(x)
        x = self.relu(x)
        #CODE Print(f"Layer 2 Shape: {x.shape}, First few lines: {x}")
        x = self.conv3(x)
        x = self.pooling(x)
        x = self.relu(x)
        #CODE Print(f"Layer 3 Shape: {x.shape}, First few lines: {x}")
        x = self.flatten(x)
        x = self.linear(x)
        x = self.output(x)
        #CODE Print(f"Output Shape: {x.shape}, First few lines: {x}")
        return x
        
#########################################################################################################################################################################################
#$ Model Object

model = AnimalFacesModel(dataset)
if PRINTSTEPS:
    summary(model, input_size= (3, 128, 128))


#########################################################################################################################################################################################
#$ Loss and Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr = LEARNING_RATE)

#########################################################################################################################################################################################
#$ Training Variables

#NOTE Keeping track of the training accuracy during each epoch. We calculate the accuracy during the batch size and we print it in the end for tracking accuracy on each epoch
total_acc_train = 0

#NOTE Keeping track of the training loss during each epoch. We calculate the loss during the batch size and we use the loss value to optimize and modify the model parameters
total_loss_train = 0

#NOTE Keeping track of the validation accuracy during each epoch. Lets us know if there is any over fitting
total_acc_val = 0

#NOTE Keeping track of the validation loss during each epoch.
total_loss_val = 0

#########################################################################################################################################################################################
#$ Training Plot Lists

#NOTE Lists to store the training and validation accuracy and loss values for each epoch. Done to visualize the training on a graph
total_acc_train_plot = []

total_loss_train_plot = []

total_acc_val_plot = []

total_loss_val_plot = []

#NOTE Training Loop: This loop iterates over the specified number of epochs (EPOCHS) to train the model
if not os.path.exists("Pytorch Projects\\ImageClassification\\model_weights.pth"):
    for epoch in range(EPOCHS):
        #NOTE Initialize variables to track training accuracy and loss for each epoch
        total_acc_train = 0
        total_loss_train = 0
        total_acc_val = 0
        total_loss_val = 0
        
        #NOTE Iterate over the training data loader to process each batch of data
        for data in train_loader:
            #NOTE Reset the gradients for the next batch
            optimizer.zero_grad()
            
            #NOTE Extract the inputs and labels from the batch data
            inputs, labels = data
            
            #NOTE Pass the inputs through the model to get the predictions
            #NOTE We use the squeeze function to modify the shape of the output and make it returns a tensor with all specified dimensions of input of size 1 removed. 
            outputs = model(inputs)
            
            #NOTE Calculate the loss between the predictions and labels using the binary cross-entropy loss function
            batch_loss = criterion(outputs, labels)
            
            #NOTE Backpropagate the loss to update the model parameters
            batch_loss.backward()
            
            #NOTE Accumulate the total loss for the epoch
            total_loss_train += batch_loss.item()
            
            #NOTE Calculate the accuracy of the predictions by comparing them to the labels.
            #NOTE DIFFERENT FROM TABULAR because more then 1 class which means we need to cross compare the class data
            acc = (torch.argmax(outputs, axis = 1) == labels).sum().item()
            
            #NOTE Accumulate the total accuracy for the epoch
            total_acc_train += acc
            
            #NOTE Update the model parameters using the Adam optimizer
            optimizer.step() 
            
        #NOTE Validation Loop: This loop iterates over the validation data loader to evaluate the model's performance
        with torch.no_grad():
            #NOTE Iterate over the validation data loader to process each batch of data
            for data in validation_loader:
                #NOTE Extract the inputs and labels from the batch data
                inputs, labels = data
                
                #NOTE Pass the inputs through the model to get the predictions.
                #NOTE We use the squeeze function to modify the shape of the output and make it returns a tensor with all specified dimensions of input of size 1 removed. 
                outputs = model(inputs)
                
                #NOTE Calculate the accuracy of the predictions by comparing them to the labels
                acc = (torch.argmax(outputs, axis = 1) == labels).sum().item()
                
                #NOTE Accumulate the total accuracy for the epoch
                total_acc_val += acc
                
                #NOTE Calculate the loss between the predictions and labels using the binary cross-entropy loss function
                batch_loss = criterion(outputs, labels)
                
                #NOTE Accumulate the total loss for the epoch
                total_loss_val += batch_loss.item()
                
        
        #NOTE Append the training and validation accuracy and loss values to the plot lists
        total_loss_train_plot.append(round(total_loss_train/1000, 4))
        total_loss_val_plot.append(round(total_loss_val/1000, 4))
        total_acc_train_plot.append(round(total_acc_train/(train_dataset.__len__())*100, 4))
        total_acc_val_plot.append(round(total_acc_val/(validation_dataset.__len__())*100,4))

        Print(f'''Epoch {epoch+1}/{EPOCHS}, Train Loss: {round(total_loss_train/100, 4)} Train Accuracy {round((total_acc_train)/train_dataset.__len__() * 100, 4)}
                    Validation Loss: {round(total_loss_val/100, 4)} Validation Accuracy: {round((total_acc_val)/validation_dataset.__len__() * 100, 4)}''')
        Print("="*25)
        
    #########################################################################################################################################################################################
    #$ Testing the Model

    with torch.no_grad(): #NOTE This with statement tells PyTorch not to train the model. just use it with the testing values
        total_loss_test = 0
        total_acc_test = 0
        
        for data in test_loader:
            inputs, labels = data
            
            prediction = model(inputs)
            
            acc = (torch.argmax(prediction, axis = 1) == labels).sum().item()
            total_acc_test += acc
            
            batch_loss_test = criterion((prediction),labels)
            total_loss_test += batch_loss_test.item()
            
    Print(f"Accuracy Score is: {round((total_acc_test/test_dataset.__len__())*100, 4)}%")

    #########################################################################################################################################################################################
    #$ Plotting

    if PRINTSTEPS:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

        axs[0].plot(total_loss_train_plot, label='Training Loss')
        axs[0].plot(total_loss_val_plot, label='Validation Loss')
        axs[0].set_title('Training and Validation Loss over Epochs')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].legend()

        axs[1].plot(total_acc_train_plot, label='Training Accuracy')
        axs[1].plot(total_acc_val_plot, label='Validation Accuracy')
        axs[1].set_title('Training and Validation Accuracy over Epochs')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')
        axs[1].legend()

        plt.tight_layout()

        plt.show()

    torch.save(model.state_dict(), "Pytorch Projects\\ImageClassification\\model_weights.pth")
else:
    model.load_state_dict(torch.load("Pytorch Projects\\ImageClassification\\model_weights.pth", weights_only = True))
    model.eval()

#########################################################################################################################################################################################
#$ Using the Model

def predict_image(image_path):
    image = _transform(Image.open(image_path).convert('RGB')).to(device)
    
    output = torch.argmax(model(image.unsqueeze(0)), axis = 1).item()
    return label_encoder.inverse_transform([output])

image = Image.open("Pytorch Projects\\ImageClassification\\predictiontest\\test3.jpeg")
plt.imshow(image)
plt.show()

Print("Prediction for the image is: ", predict_image("Pytorch Projects\\ImageClassification\\predictiontest\\test3.jpeg"))