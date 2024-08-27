## copied over from colab - modified for cluster environment

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

weights_path = Path("vggface.pth")
seed = 0
torch.manual_seed(seed)

# data loader strx
class CelebADataset(Dataset):
  def __init__(self,
               annotations_file = 'celebA_image_subset/annotations_celebA_subset.pkl',
               img_dir = 'celebA_image_subset/') -> None:
      self.img_labels = pd.read_pickle(annotations_file)
      self.img_dir = img_dir
      self.transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  ## other mean/std set doesnt work
        #transforms.Normalize(mean=[129.186279296875, 104.76238250732422, 93.59396362304688], std=[1, 1, 1]) 
    ])

  def __len__(self):
      return len(self.img_labels)

  def __getitem__(self, idx):
      img_path = Path(self.img_dir, self.img_labels.iloc[idx, 0])
      image = Image.open(img_path)
      image = self.transform(image).squeeze(0)

      label = torch.tensor(self.img_labels.iloc[idx, 1])
      return (image, label)

class VGGFaceTrainer: 
    def __init__(self, model, train_loader, val_loader, seed,
                num_epochs=15, lr=0.001):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.seed = seed
        self.num_epochs = num_epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.fc8.parameters(), lr=lr)
        self.loss_log = []
        self.val_log = []

        self.save_dir = 'models/fine_tune'

    def train(self):
        
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0

            for inputs, labels in self.train_loader:
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            
            self.loss_log.append(running_loss / len(self.train_loader))
            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {running_loss/len(self.train_loader)}")

            self.validate()
            self.save_model(epoch)
    
    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            self.val_log.append(accuracy)
            print(f"val accuracy: {accuracy}%")

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), f"{self.save_dir}/seed{self.seed}_epoch{epoch}.pth")
        # overwrite loss and val logs 
        pd.DataFrame(self.loss_log).to_csv(f"{self.save_dir}/seed{self.seed}_loss_log.csv")
        pd.DataFrame(self.val_log).to_csv(f"{self.save_dir}/seed{self.seed}_val_log.csv")

def replace_fc(model, num_outputs=10):
    model = nn.Sequential(*(list(model.children())[:-1]))
    model.fc8 = nn.Linear(in_features=4096, out_features=10, bias=True, )
    return model

if __name__ == '__main__':
    model = vgg_face_dag(weights_path=weights_path)
    
    model = replace_fc(model)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc8.parameters():
        param.requires_grad = True

    train_dataset = CelebADataset()
    val_dataset = CelebADataset()

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    trainer = VGGFaceTrainer(model=model, train_loader=train_loader,
                        val_loader=val_loader, seed=seed, num_epochs=30, lr=0.001)

    trainer.train()
