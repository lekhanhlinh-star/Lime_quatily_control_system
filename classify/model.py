from ultralytics import YOLO
import supervision as sv
from torchvision.transforms import functional as F
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from skimage import exposure

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as Fs
import torchvision
import pytorch_lightning as pl
from pytorch_lightning import  Callback
import torchmetrics

import numpy as np
from PIL import Image



class LitModel(pl.LightningModule):
    def __init__(self, input_shape, num_classes, learning_rate=2e-4):
        super().__init__()
        
        # log hyperparameters
        self.save_hyperparameters()
        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.accuracy = torchmetrics.Accuracy(task="binary")
        self.model_transfer = torchvision.models.efficientnet_b3(weights='IMAGENET1K_V1')
      
        for param in self.model_transfer.features.parameters():
            param.requires_grad = False
            
        self.fully_connected=torch.nn.Sequential(
                            torch.nn.Dropout(p=0.2, inplace=True), 
                            torch.nn.Linear(in_features=1000, 
                                            out_features=num_classes, # same number of output units as our number of classes
                                            bias=True))

    # will be used during inference
    def forward(self, x):
        x=self.model_transfer(x)
        x=self.fully_connected(x)
        return x
    

    

    
def load_model(path):
    model_classification = LitModel.load_from_checkpoint(path).to("cuda:0")
    return model_classification

model_classification=load_model(r"weights\classification_lemon-v14.ckpt")
model_classification.eval()
def predict_classification(crop_img_list, class_labels = ["Bad","Good",] ):
   
    if len(crop_img_list)==0:
        return 
   
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.GaussianBlur((3,3)),
        transforms.ToTensor(),
        
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    predicted_labels = []
   
    for input_img_crop in crop_img_list:
        # print(input_img_crop)
     

        # # Convert the image to a numpy array
        # image_np = np.array(input_img_crop)

        # # Apply gamma correction to the image
        # # image_gamma = exposure.equalize_adapthist(image_np)

        # # Convert the corrected image back to PIL format
        # image_gamma_pil = Image.fromarray(np.uint8(image_gamma*255))
        # print(image_gamma_pil)

     
       

        input_tensor = transform(  input_img_crop).unsqueeze(0).to("cuda:0")
        print(input_tensor.shape)

        with torch.no_grad():
            output_tensor = model_classification(input_tensor)
            output_tensor= F.softmax(output_tensor, dim=-1)
        
        output_data = output_tensor.detach().cpu().numpy()

        for i in range(output_data.shape[0]):
            predicted_index = np.argmax(output_data[i])
            if (predicted_index==1):
                if(output_data[i][1]<0.7):
                    predicted_label = class_labels[0]
                    predicted_labels.append(predicted_label)
                else:
                    predicted_label = class_labels[predicted_index]
                    predicted_labels.append(predicted_label)
   
                    
            else:
                predicted_label = class_labels[predicted_index]
                predicted_labels.append(predicted_label)
    print(output_data)    
    return predicted_labels 