## extract embeddings from the fine-tuned model 

from PIL import Image

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.io import read_image
from torchvision import transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from process_model_weights import *

#weights_path = Path("models/fine_tune/seed0_epoch16.pth")
weights_path = Path("vggface.pth")
seed = 0
torch.manual_seed(seed)

model = vgg_face_dag(weights_path=weights_path)
print(model)