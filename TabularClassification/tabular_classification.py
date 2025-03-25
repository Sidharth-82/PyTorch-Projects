#Python3 Shebang
#########################################################################################################################################################################################

#$ Importing dataset
# import opendatasets as od
# od.download("https://www.kaggle.com/datasets/mssmartypants/rice-type-classification") #Binary type dataset. Either says it is Jasmin or Gonan rice


#import torch packages: Main Framework, NN for layers, Adam Optimizer, Dataset Class and Dataloader for creating objects, Summary to visualize model layers and parameters
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

#import Sci-Kit Learn Packages: Dataset Splitting, Calculating test accuracy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#import Matplotlib packages: Plotting training progress
import matplotlib.pyplot as plt

#import Pandas packages: Data reading and preprocessing
import pandas as pd

#import Numpy
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu' # detect the GPU if any, if not use CPU, change cuda to mps if you have a mac

PRINTSTEPS = False #Note Set True to print midsteps

def Print(*args):
    if PRINTSTEPS:
        print(args)

#########################################################################################################################################################################################
#$ CSV Processing
data_csv = pd.read_csv("Pytorch Projects\\TabularClassification\\rice-type-classification\\riceClassification.csv") #read CSV using Pandas
data_csv.dropna(inplace=True) #remove null/missing values
data_csv.drop(["id"], axis=1, inplace=True) #Drop the id coloumn

classes = data_csv["Class"].unique()
Print(f"Output Possibilities: {classes}") #Displays Possible Outputs

#NOTE Output Possibilities: ['0: Gonan' '1: Jasmin']
Print(f"Data Shape (rows, cols): {data_csv.shape}")
Print(f"First 5 rows {data_csv.head()}") #Displays first 5 rows of data

#########################################################################################################################################################################################
#$ Data Pre-Processing

original_data = data_csv.copy()

#NOTE Divides all values by the maximum value within that coloum to make all values within +/- 1
for col in data_csv.columns:
    data_csv[col] = data_csv[col]/data_csv[col].abs().max()

Print(f"First 5 rows {data_csv.head()}")

#########################################################################################################################################################################################
#$ Data Splitting
#NOTE Training Size: 70%
train_size = 0.7
#NOTE Validation Size 15%
val_size = 0.15
#NOTE Testing Size: 15%
test_size = 0.15

X = np.array(data_csv.iloc[:,:-1]) #get the inputs, all rows and all coloums except last coloum
Y = np.array(data_csv.iloc[:, -1]) #get the last coloum. ie the output. types of rice
#NOTE X is the input and Y is the output

#NOTE Creates a randomized split of the values. train = 0.7, test + val = 0.3
X_train, X_testval, Y_train, Y_testval = train_test_split(X,Y,test_size= test_size + val_size)
#NOTE Creates a randomized split of the test+val values, test = 0.15, val = 0.15, 
X_test, X_val, Y_test, Y_val = train_test_split(X_testval, Y_testval, test_size = (test_size/(test_size+val_size)))

Print("Training set is: ", X_train.shape[0], " rows which is ", round(X_train.shape[0]/data_csv.shape[0],4)*100, "%") # Print training shape
Print("Validation set is: ",X_val.shape[0], " rows which is ", round(X_val.shape[0]/data_csv.shape[0],4)*100, "%") # Print validation shape
Print("Testing set is: ",X_test.shape[0], " rows which is ", round(X_test.shape[0]/data_csv.shape[0],4)*100, "%") # Print testing shape

#########################################################################################################################################################################################
#$ Dataset Object

#NOTE Created a Dataset Class to convert csv data into a PyTorch Datset using Tensors
class RiceDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype = torch.float32).to(device)
        self.Y = torch.tensor(Y, dtype = torch.float32).to(device)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
training_data = RiceDataset(X_train, Y_train)
validation_data = RiceDataset(X_val, Y_val)
testing_data = RiceDataset(X_test, Y_test)

#########################################################################################################################################################################################
#$ Training Hyper Parameters

#NOTE BATCH_SIZE: The number of samples in a batch, used for training the model
BATCH_SIZE = 32
#NOTE EPOCHS: The number of times the model will iterate over the entire training dataset
EPOCHS = 10
#NOTE HIDDEN_LAYERS: The number of neurons in the hidden layer of the neural network
HIDDEN_NEURONS = 10
#NOTE LEARNING_RATE: The rate at which the model learns from the data, a higher value can lead to faster convergence but may also cause overshooting
LEARNING_RATE = 0.001

#########################################################################################################################################################################################
#$ Data Loader

#NOTE Creates a data loader for the training, validation and testing datasets
#NOTE The data loader is a PyTorch object that is used to load the data in batches, 
#NOTE allowing for efficient training and testing of the model
#NOTE The batch_size parameter determines the number of samples in each batch
#NOTE The shuffle parameter determines whether the data is shuffled before each epoch
train_dataloader = DataLoader(training_data, batch_size= BATCH_SIZE, shuffle = True)
validation_dataloader = DataLoader(validation_data, batch_size= BATCH_SIZE, shuffle = False)
testing_dataloader = DataLoader(testing_data, batch_size= BATCH_SIZE, shuffle = False)

#########################################################################################################################################################################################
#$ Model Class

#NOTE Creates a PyTorch model class that can be used to train and test the model
class RiceModel(nn.Module):
    def __init__(self):
        super(RiceModel, self).__init__()
        
        #NOTE Input layer with X.shape[1] input features (10) and HIDDEN_NEURONS output features
        #NOTE Linear Conversion rate between input and output.
        self.input_layer = nn.Linear(X.shape[1], HIDDEN_NEURONS)
        
        #NOTE Hidden Layer with HIDDEN_NEURONS input features and 1 output feature.
        #NOTE Can have more to improve model in case of more complex data input
        self.hidden = nn.Linear(HIDDEN_NEURONS, 1)
        
        #NOTE Activation Function form of Sigmoid
        #NOTE Sigmoid is used here because the output is a binary classification problem (0 or 1, Gonan or Jasmin rice)
        #NOTE The sigmoid function maps any real-valued number to a value between 0 and 1, which is ideal for binary classification problems
        #NOTE The output of the sigmoid function can be interpreted as a probability, where values close to 1 represent a high probability of the positive class (Jasmin rice)
        #NOTE and values close to 0 represent a low probability of the positive class hence it more likely to be the Negative Class (Gonan)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        #NOTE Forward propagation is the process of providing an input and following it across the Neural Network to output a predicted probability value.
        # The input data 'x' enters the input layer, where it is transformed into a higher-dimensional space to capture complex relationships. not in this case, the layer input is 10 and output is 10
        #CODE Print(f"Forward Propagation Input Shape: {x.shape}, First few lines: {x}")
        x = self.input_layer(x)
        #CODE Print(f"Input Layer Ouput Shape: {x.shape}, First few lines: {x[:5]}")
        # The output of the input layer is then passed through the hidden layer, where it is further transformed to extract relevant features.
        x = self.hidden(x)
        #CODE Print(f"Hidden Layer Ouput Shape: {x.shape}, First few lines: {x[:5]}")
        # The output of the hidden layer is then passed through the sigmoid activation function, which maps the output to a probability value between 0 and 1.
        x = self.sigmoid(x)
        #CODE Print(f"Sigmoid Layer Output Shape: {x.shape}, First few lines: {x}")
        return x
    
#########################################################################################################################################################################################
#$ Model Creation

model = RiceModel().to(device) #Creates the Model using the device
#CODE summary(model, (X.shape[1],)) #prints a summary of the model

#########################################################################################################################################################################################
#$ Loss Function and Optimizer

# Define the loss function and optimizer for the model
# NOTE The loss function is Binary Cross Entropy Loss (BCELoss) which is suitable for binary classification problems
# NOTE The optimizer is Adam, a popular stochastic gradient descent optimizer, with a learning rate of LEARNING_RATE
criterion = nn.BCELoss()  # Loss function for binary classification
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)  # Adam optimizer with specified learning rate

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

#########################################################################################################################################################################################
#$ Training Loop

#NOTE Training Loop: This loop iterates over the specified number of epochs (EPOCHS) to train the model
for epoch in range(EPOCHS):
    #NOTE Initialize variables to track training accuracy and loss for each epoch
    total_acc_train = 0
    total_loss_train = 0
    total_acc_val = 0
    total_loss_val = 0
    
    #NOTE Iterate over the training data loader to process each batch of data
    for data in train_dataloader:
        #NOTE Extract the inputs and labels from the batch data
        inputs, labels = data
        
        #NOTE Pass the inputs through the model to get the predictions
        #NOTE We use the squeeze function to modify the shape of the output and make it returns a tensor with all specified dimensions of input of size 1 removed. 
        prediction = model(inputs).squeeze(1)
        
        #NOTE Calculate the loss between the predictions and labels using the binary cross-entropy loss function
        batch_loss = criterion(prediction, labels)
        
        #NOTE Accumulate the total loss for the epoch
        total_loss_train += batch_loss.item()
        
        #NOTE Calculate the accuracy of the predictions by comparing them to the labels
        acc = ((prediction).round() == labels).sum().item()
        
        #NOTE Accumulate the total accuracy for the epoch
        total_acc_train += acc
        
        #NOTE Backpropagate the loss to update the model parameters
        batch_loss.backward()
        
        #NOTE Update the model parameters using the Adam optimizer
        optimizer.step()
        
        #NOTE Reset the gradients for the next batch
        optimizer.zero_grad()
        
    #NOTE Validation Loop: This loop iterates over the validation data loader to evaluate the model's performance
    with torch.no_grad():
        #NOTE Iterate over the validation data loader to process each batch of data
        for data in validation_dataloader:
            #NOTE Extract the inputs and labels from the batch data
            inputs, labels = data
            
            #NOTE Pass the inputs through the model to get the predictions.
            #NOTE We use the squeeze function to modify the shape of the output and make it returns a tensor with all specified dimensions of input of size 1 removed. 
            prediction = model(inputs).squeeze(1)
            
            #NOTE Calculate the loss between the predictions and labels using the binary cross-entropy loss function
            batch_loss = criterion(prediction, labels)
            
            #NOTE Accumulate the total loss for the epoch
            total_loss_val += batch_loss.item()
            
            #NOTE Calculate the accuracy of the predictions by comparing them to the labels
            acc = ((prediction).round() == labels).sum().item()
            
            #NOTE Accumulate the total accuracy for the epoch
            total_acc_val += acc
            
    #NOTE Append the training and validation accuracy and loss values to the plot lists
    total_loss_train_plot.append(round(total_loss_train/1000, 4))
    total_loss_val_plot.append(round(total_loss_val/1000, 4))
    total_acc_train_plot.append(round(total_acc_train/(training_data.__len__())*100, 4))
    total_acc_val_plot.append(round(total_acc_val/(validation_data.__len__())*100,4))
    
#########################################################################################################################################################################################
#$ Testing the Model

with torch.no_grad(): #NOTE This with statement tells PyTorch not to train the model. just use it with the testing values
    total_loss_test = 0
    total_acc_test = 0
    
    for data in testing_dataloader:
        inputs, labels = data
        
        prediction = model(inputs).squeeze(1)
        batch_loss_test = criterion((prediction),labels)
        total_loss_test += batch_loss_test.item()
        
        acc = ((prediction).round() == labels).sum().item()
        total_acc_test += acc
Print(f"Accuracy Score is: {round((total_acc_test/X_test.shape[0])*100, 2)}%")

#########################################################################################################################################################################################
#$ Plotting the Model

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

axs[0].plot(total_loss_train_plot, label='Training Loss')
axs[0].plot(total_loss_val_plot, label='Validation Loss')
axs[0].set_title('Training and Validation Loss over Epochs')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].set_ylim([0, 2])
axs[0].legend()

axs[1].plot(total_acc_train_plot, label='Training Accuracy')
axs[1].plot(total_acc_val_plot, label='Validation Accuracy')
axs[1].set_title('Training and Validation Accuracy over Epochs')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].set_ylim([0, 100])
axs[1].legend()

plt.tight_layout()

plt.show()

#########################################################################################################################################################################################
#$ Inference (Using the Model)

area = float(input("Area: "))/original_data['Area'].abs().max()
MajorAxisLength = float(input("Major Axis Length: "))/original_data['MajorAxisLength'].abs().max()
MinorAxisLength = float(input("Minor Axis Length: "))/original_data['MinorAxisLength'].abs().max()
Eccentricity = float(input("Eccentricity: "))/original_data['Eccentricity'].abs().max()
ConvexArea = float(input("Convex Area: "))/original_data['ConvexArea'].abs().max()
EquivDiameter = float(input("EquivDiameter: "))/original_data['EquivDiameter'].abs().max()
Extent = float(input("Extent: "))/original_data['Extent'].abs().max()
Perimeter = float(input("Perimeter: "))/original_data['Perimeter'].abs().max()
Roundness = float(input("Roundness: "))/original_data['Roundness'].abs().max()
AspectRation = float(input("AspectRation: "))/original_data['AspectRation'].abs().max()

my_inputs = [area, MajorAxisLength, MinorAxisLength, Eccentricity, ConvexArea, EquivDiameter, Extent, Perimeter, Roundness, AspectRation]

print("="*20)
model_inputs = torch.Tensor(my_inputs).to(device)
prediction = (model(model_inputs))
print(prediction)
print("Class is: ", round(prediction.item()))