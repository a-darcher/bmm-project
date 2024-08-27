import sys
import os

from pathlib import Path

from PIL import Image

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import random

from sabine_code.real_rnn import *

import itertools

from process_model_weights import *


seq_length = 5

files = ["002514.jpg",
 "039593.jpg",
 "094741.jpg",
 "152801.jpg",
 "126154.jpg",
 "034026.jpg",
 "012857.jpg",
 "016242.jpg",
 "041553.jpg",
 "068982.jpg",]

class CelebASeriesDataset(Dataset):
    def __init__(self, root, train=True, transform=None, series_length=5):
        # Define the transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load the CelebA dataset
        celeba_dataset = datasets.CelebA(root='./celebA_dataset', download=True, transform=transform)

        # Select a subset of the dataset based on file names
        selected_file_names = files
        selected_indices = [i for i, name in enumerate(celeba_dataset.filename) if name in selected_file_names]
        self.selected_dataset = torch.utils.data.Subset(celeba_dataset, selected_indices)
        self.series_length = series_length

        # Generate all possible unique sequences
        self.sequences, self.labels = self.generate_sequences()

        # Split the dataset into train, validation, and test sets
        self.train_sequences, self.train_labels, self.val_sequences, self.val_labels, self.test_sequences, self.test_labels = self.split_dataset()

    def generate_sequences(self):
        sequences = []
        labels = []
        unique_sequences = set()

        for idx in range(len(self.selected_dataset)):
            first_image, first_label = self.selected_dataset[idx]

            for repeated_item_idx in range(self.series_length):
                for positions in itertools.combinations(range(self.series_length), 2):
                    remaining_positions = [i for i in range(self.series_length) if i not in positions]
                    indices = torch.where(torch.tensor([self.selected_dataset[i][1] for i in range(len(self.selected_dataset))]) != first_label)[0]
                    remaining_items = [self.selected_dataset[i][0] for i in indices]

                    for perm in itertools.permutations(remaining_items, len(remaining_positions)):
                        sequence = tuple(torch.stack([first_image if i in positions else perm[j] for j, i in enumerate(remaining_positions)]))
                        if sequence not in unique_sequences:
                            unique_sequences.add(sequence)
                            label = torch.zeros(self.series_length)
                            label[positions[1]] = 1
                            sequences.append(sequence)
                            labels.append(label)

        return torch.stack(sequences), torch.stack(labels)

    def split_dataset(self):
        total_length = len(self.sequences)
        train_size = int(0.5 * total_length)
        val_size = int(0.25 * total_length)
        test_size = total_length - train_size - val_size

        train_sequences = self.sequences[:train_size]
        train_labels = self.labels[:train_size]
        val_sequences = self.sequences[train_size:train_size + val_size]
        val_labels = self.labels[train_size:train_size + val_size]
        test_sequences = self.sequences[train_size + val_size:]
        test_labels = self.labels[train_size + val_size:]

        return train_sequences, train_labels, val_sequences, val_labels, test_sequences, test_labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# class CelebASeriesDataset(Dataset):

#     def __init__(self, root, train=True, transform=None, series_length=5):
#         # Define the transforms
#         self.transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  ## other mean/std set doesnt work
#     ])

#         # Load the CelebA dataset
#         celeba_dataset = datasets.CelebA(root='./celebA_dataset', download=True, transform=transform)
        
#         # # Get the file names of the first 10 images

#         # file_names = [celeba_dataset.filename[i] for i in range(10)]

#         # print(file_names)

#         # Select a subset of the dataset based on file names
#         selected_file_names = files
#         selected_indices = [i for i, name in enumerate(celeba_dataset.filename) if name in selected_file_names]
#         self.selected_dataset = torch.utils.data.Subset(celeba_dataset, selected_indices)
#         self.series_length = series_length

#     def __len__(self):
#         return len(self.selected_dataset)

#     def __getitem__(self, idx):
#         first_image, first_label = self.selected_dataset[idx]

#         sequences = []
#         labels = []

#         for repeated_item_idx in range(self.series_length):
#             for positions in itertools.combinations(range(self.series_length), 2):
#                 remaining_positions = [i for i in range(self.series_length) if i not in positions]
#                 remaining_items = [self.selected_dataset[i][0] for i in range(len(self.selected_dataset)) if self.selected_dataset[i][1] != first_label]

#                 for perm in itertools.permutations(remaining_items, len(remaining_positions)):
#                     sequence = [None] * self.series_length
#                     sequence[positions[0]] = first_image
#                     sequence[positions[1]] = first_image
#                     for idx, pos in enumerate(remaining_positions):
#                         sequence[pos] = perm[idx]

#                     label = torch.zeros(self.series_length)
#                     label[positions[1]] = 1

#                     sequences.append(torch.stack(sequence))
#                     labels.append(label)

#         return torch.stack(sequences), torch.stack(labels)

# Create the dataset and dataloader
series_length = int(seq_length)  
batch_size = 128

dataset = CelebASeriesDataset(root='celebA_dataset/celeba/img_align_celeba/', series_length=series_length)

# Calculate the split indices
total_length = len(dataset)
train_size = int(0.5 * total_length)
val_size = int(0.25 * total_length)
test_size = total_length - train_size - val_size

# Get the indices for each split
train_indices = list(range(train_size))
val_indices = list(range(train_size, train_size + val_size))
test_indices = list(range(train_size + val_size, total_length))

# Create the train, validation, and test datasets
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

# Create the dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)

print(f"Total dataset length: {len(dataset)}")
print(f"Train dataset length: {len(train_dataset)}")
print(f"Validation dataset length: {len(val_dataset)}")
print(f"Test dataset length: {len(test_dataset)}")
assert len(train_dataset) + len(val_dataset) + len(test_dataset) == len(dataset), "Sum of split sizes does not match total dataset length"

##############

#load best cnn checkpoint
backbone = CNN(num_classes=10, in_channels=1)

#loop over the checkpoint files
best_accuracy = 0
for checkpoint in os.listdir('./checkpoints'):
    if checkpoint.endswith('.pth') and checkpoint.startswith(model_name) and 'RNN' not in checkpoint and 'ood' not in checkpoint and 'combined' in checkpoint:
        accuracy = max(best_accuracy, float(checkpoint.split('_')[-1][:-4]))
        if accuracy > best_accuracy:
            best_accuracy = accuracy

backbone.load_state_dict(torch.load('./checkpoints/{}_model_combined_acc_{}.pth'.format(model_name, best_accuracy))) #_combined


weights_path = Path("vggface.pth")
backbone = nn.Sequential(*(list(vgg_face_dag(weights_path=weights).children())[:-1]))

embedding_model = backbone

class embedding_backbone(nn.Module):
    def __init__(self):
        super(embedding_backbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=3 // 2)
        self.conv1_1 = nn.Conv2d(64, 64, kernel_size=3, padding=3 // 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=3 // 2)
        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=3, padding=3 // 2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        out = F.relu(F.max_pool2d(self.conv1(x), 2))
        out = F.relu(self.conv1_1(out))
        out = F.relu(F.max_pool2d(self.conv2(out), 2))
        out = F.relu(self.conv2_1(out))
        out = F.relu(F.max_pool2d(self.conv3(out), 2))
        return out

# class embedding_backbone(nn.Module):
#     def __init__(self):
#         super(embedding_backbone, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=3 // 2)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=3 // 2)
#
#     def forward(self, x):
#         out = F.relu(F.max_pool2d(self.conv1(x), 2))
#         out = F.relu(F.max_pool2d(self.conv2(out), 2))
#         return out
# embedding_model = embedding_backbone()
# # remove fc layers from backbone
# embedding_model.conv1 = backbone.conv1
# embedding_model.conv1_1 = backbone.conv1_1
# embedding_model.conv2 = backbone.conv2
# embedding_model.conv2_1 = backbone.conv2_1
# embedding_model.conv3 = backbone.conv3
embedding_model.eval()
embedding_model.to('cuda')
for param in embedding_model.parameters():
    param.requires_grad = False

model = Vanilla_RNN(input_size=4096, hidden_size=512, num_layers=1, num_classes=1).to('cuda')
# model = LSTM_RNN(input_size=4096, hidden_size=512, num_layers=1, num_classes=1).to('cuda')
# model = CV_RNN(input_size=1568, hidden_size=32, num_layers=1, num_classes=1, batch_size=batch_size).to('cuda')

# Define the loss function and optimizer
# criterion = torch.nn.BCEWithLogitsLoss()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



# Train the model
def train(model, backbone, dataloader, optimizer, criterion, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(dataloader):


            images = images.to('cuda')
            labels = labels.to('cuda')
            # Reshape the images
            embeddings = []
            for j in range(images.size(1)):
                img = images[:, j]
                # get embedding from backbone: activity from the last convolutional layer
                embedding = embedding_model(img)
                # embedding = embedding_model(img)
                embedding = embedding.view(embedding.size(0), -1)
                embeddings.append(embedding)
            embeddings = torch.stack(embeddings, dim=1)
            preds = model(embeddings)
            # import pdb; pdb.set_trace()
            loss = criterion(preds.squeeze(), labels.float().squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # import pdb; pdb.set_trace()
            #compute the accuracy
            preds = torch.softmax(preds, dim=1)
            correct = (preds.argmax(dim=1).squeeze() == labels.argmax(dim=1).squeeze()).float().sum()

            if i % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}, Accuracy: {correct.item()}/{labels.size(0)} ({100 * correct.item() / (labels.size(0)):.2f}%)')
        validate(model, backbone, val_dataloader)

def validate(model, backbone, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to('cuda')
            labels = labels.to('cuda')
            # Reshape the images
            embeddings = []
            for j in range(images.size(1)):
                img = images[:, j]
                # get embedding from backbone: activity from the last convolutional layer
                embedding = embedding_model(img)
                # embedding = embedding_model(img)
                embedding = embedding.view(embedding.size(0), -1)
                embeddings.append(embedding)
            embeddings = torch.stack(embeddings, dim=1)
            preds = model(embeddings)
            # compute the accuracy
            preds = torch.softmax(preds, dim=1)
            temp = (preds.argmax(dim=1).squeeze() == labels.argmax(dim=1).squeeze()).float().sum()
            correct += temp
            total += labels.size(0)
    print(f'Validation Accuracy: {100 * correct.item() / (total):.2f}%')
    torch.save(model.state_dict(), 'checkpoints/{}_ReLU_RNN512_FMNIST_{}.pth'.format(model_name, seq_length))

def test(model, backbone, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to('cuda')
            labels = labels.to('cuda')
            # Reshape the images
            embeddings = []
            for j in range(images.size(1)):
                img = images[:, j]
                # get embedding from backbone: activity from the last convolutional layer
                embedding = embedding_model(img)
                # embedding = embedding_model(img)
                embedding = embedding.view(embedding.size(0), -1)
                embeddings.append(embedding)
            embeddings = torch.stack(embeddings, dim=1)
            preds = model(embeddings)
            # compute the accuracy
            preds = torch.softmax(preds, dim=1)
            correct = (preds.argmax(dim=1).squeeze() == labels.argmax(dim=1).squeeze()).float().sum()
            total += labels.size(0)
    print(f'Test Accuracy: {100 * correct.item() / total:.2f}%')

train(model, embedding_model, train_dataloader, optimizer, criterion, num_epochs=10)
test(model, embedding_model, test_dataloader)