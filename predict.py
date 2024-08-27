from pathlib import Path

import torch    
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import process_model_weights
#from inputs import *

# weights_path = ""

# model = process_model_weights.vgg_face_dag(weights_path=weights_path)

# model.eval()

def predict_from_image(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
    
    m, predicted = torch.max(outputs,1)

    return m.item(), predicted.item()

def display_image(image_path):
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def get_most_frequent_images(labels, count=30):
    frequent_labels = np.array(labels['id'].value_counts()[labels['id'].value_counts()==count].index)
    return frequent_labels

