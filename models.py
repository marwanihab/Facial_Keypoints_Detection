## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):
    
    def init_weights(self, m):
        if type(m) == nn.Conv2d:
            I.uniform_(m.weight, a=0, b=1)
        elif type(m) == nn.Linear: 
            I.xavier_uniform_(m.weight, gain=1)
        
        
    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        ## Our input image will be of size 224x224x1
        ## For each layer the output size = (W-F)/S +1 
        
        #Using NAIMISHNET architecture
        
        #Convolution Layers
        
        #input size 224x224x1
        #output 221x221x32
        #output after pooling 110x110x32
        self.conv1 = nn.Conv2d(1, 32, 4)
        #pooling layer
        self.drop1 = nn.Dropout(p= 0.1)
        
        
        #input size 110x110x32
        #output 108x108x64
        #output after pooling 54x54x64
        self.conv2 = nn.Conv2d(32, 64, 3)  
        #pooling layer 
        self.drop2 = nn.Dropout(p= 0.2)
        
        
        #input size 54x54x64
        #output 53x53x128
        #output after pooling 26x26x128
        self.conv3 = nn.Conv2d(64, 128, 2)
        #pooling layer 
        self.drop3 = nn.Dropout(p= 0.3)
        
        
        #input size 26x26x128
        #output 26x26x256
        #output after pooling 13x13x256
        self.conv4 = nn.Conv2d(128, 256, 1)
        #pooling layer 
        self.drop4 = nn.Dropout(p= 0.4)
        #flattern Layer 
        self.dense1 = nn.Linear(13*13*256, 1000)
        
        
        
        #Pooling Layer
        self.pool = nn.MaxPool2d(2, 2)
        
        
        #Dropout Layers
        self.drop5 = nn.Dropout(p= 0.5)
        self.drop6 = nn.Dropout(p= 0.6)
        
        #Dense Layers(fully connected layers)
        self.dense2 = nn.Linear(1000, 300)
        self.dense3 = nn.Linear(300, 136)
        
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.elu(self.conv1(x)))
        x = self.drop1(x)
        
        x = self.pool(F.elu(self.conv2(x)))
        x = self.drop2(x)
        
        x = self.pool(F.elu(self.conv3(x)))
        x = self.drop3(x)
        
        x = self.pool(F.elu(self.conv4(x)))
        x = self.drop4(x)
        
        #flatten layer 
        x = x.view(x.size(0), -1)
        
        x = F.elu(self.dense1(x))
        x = self.drop5(x)
        
        x = F.relu(self.dense2(x))
        x = self.drop6(x)
        
        x = self.dense3(x)
        
  
      
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
