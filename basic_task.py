import sys

from collections import Counter

import torch
from torch import nn

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import random
import os

import pandas as pd

from celeb_backbone import *

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.W_xh = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        self.activ = nn.ReLU()

    def forward(self, x):
        batch_size, sequence_length, _ = x.size()

        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        output = []
        for t in range(sequence_length):
            x_t = x[:, t, :]
            h = self.activ(self.W_xh(x_t) + self.W_hh(h) + self.b_h)
            output.append(h)
        return torch.stack(output, dim=1)


class Vanilla_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(Vanilla_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = RNN(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.rnn(x)
        outputs = self.fc(out[:, -1, :])  
        return outputs.unsqueeze(1)

        
class LSTM_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        outputs = torch.zeros(out.size(0), out.size(1), 1).to(x.device)
        for i in range(out.size(1)):
            outputs[:, i, :] = self.fc(out[:, i, :])
        return outputs


#############
#############

class CelebAEmbeddingsDataset(Dataset):
    def __init__(self, root="celebA_embeddings/", series_length=5, ):
        self.root = root
        self.series_length = series_length 
        self.stim_dir = Path(root, f"seq{series_length}")

    def __len__(self):
        files = list(self.stim_dir.glob('*.pt'))
        return len(files)

    def __getitem__(self, idx):
        idx = idx + 1
        item_path = Path(self.root, f"seq{self.series_length}", f"{idx:06d}.pt")
        item = torch.load(item_path, weights_only=False)

        sequence = item["sequence"]
        sequence = sequence.squeeze(2) 
        
        label = item["label"]      
        label = torch.tensor(label, dtype=torch.long)

        return (sequence, label)


#############
#############

def get_random_subset(total_indices, num_samples, seed=None):
    """
    Randomly pull a subset of items from the total_indices list and remove them from the original list.
    
    Args:
        num_samples (int): The number of items to randomly select.
        seed (int, optional): A seed value for the random number generator. If provided, the random behavior will be reproducible.
    
    Returns:
        list: The randomly selected subset of items.
    """
    if num_samples > len(total_indices):
        raise ValueError("num_samples cannot be greater than the length of total_indices.")
    
    if seed is not None:
        random.seed(seed)
    
    random_subset = random.sample(total_indices, num_samples)
    for index in random_subset:
        total_indices.remove(index)
    
    return random_subset, total_indices



def data_prepper(series_length, batch_size, seed_value, root="celebA_embeddings/",):
    
    dataset = CelebAEmbeddingsDataset(series_length=series_length, root=root)

    total_length = len(dataset)
    train_size = int(0.7 * total_length)
    val_size = int(0.15 * total_length)
    test_size = total_length - train_size - val_size


    total_indices = list(range(total_length))

    train_indices, remainder = get_random_subset(total_indices, train_size, seed=seed_value)
    val_indices, test_indices = get_random_subset(total_indices, val_size, seed=seed_value)

    assert total_length == len(val_indices) + len(test_indices) + len(train_indices)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8, pin_memory=True)


    data_checks(train_dataset, val_dataset, test_dataset)
    
    return train_dataloader, val_dataloader, test_dataloader


def data_checks(train_dataset, val_dataset, test_dataset):

    print("train:")
    train_labels = [np.where(train_dataset[i][1])[0][0] for i in range(len(train_dataset))]
    train_counts = Counter(train_labels)
    print(train_counts)

    print("\nval")
    val_labels = [np.where(val_dataset[i][1])[0][0] for i in range(len(val_dataset))]
    val_counts = Counter(val_labels)
    print(val_counts)

    print("\ntest")
    test_labels = [np.where(test_dataset[i][1])[0][0] for i in range(len(test_dataset))]
    test_counts = Counter(test_labels)
    print(test_counts)

def train(model, criterion, optimizer, train_dataloader, val_dataloader, num_epochs=5):
    batches_list = []
    loss_list = []



    train_accuracy_list = []
    val_accuracy_list = []

    for epoch in range(num_epochs):
        model.train()
        for i, (sequences, labels) in enumerate(train_dataloader):
        
            sequences = sequences.to('cuda')
            labels = labels.to('cuda')
            sequences = sequences.squeeze(2) 
            preds = model(sequences)
            
            loss = criterion(preds.squeeze(), labels.float().squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds_ = torch.softmax(preds, dim=2)
            correct = (preds_.argmax(dim=2).T == labels.argmax(dim=1)).float().sum()
            
            batches_list.append(i)
            loss_list.append(loss.item())

            if i % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}, Accuracy: {correct.item()}/{labels.size(0)} ({100 * correct.item() / (labels.size(0)):.2f}%)')
        

        acc = validate(model, train_dataloader, epoch)
        train_accuracy_list.append(acc)

        acc = validate(model, val_dataloader, epoch)
        val_accuracy_list.append(acc)

    return batches_list, loss_list, train_accuracy_list, val_accuracy_list

def validate(model, val_dataloader, criterion, epoch):
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sequences, labels in val_dataloader:
            
            sequences = sequences.to('cuda')
            labels = labels.to('cuda')

            sequences = sequences.squeeze(2) 
            preds = model(sequences)
            
            loss = criterion(preds.squeeze(), labels.float().squeeze())

            preds_ = torch.softmax(preds, dim=2)
            temp = (preds_.argmax(dim=2).T == labels.argmax(dim=1)).float().sum()
            correct += temp.item()
            total += labels.size(0)
            
    sequence_length = labels.size(1)
    
    path = Path(f"models/vanilla_rnn/seq{sequence_length}/checkpoints/")
    path.mkdir(exist_ok=True, parents=True)

    print(f'Validation Accuracy: {100 * correct / total}%')
    torch.save(model.state_dict(), path / f'epoch{epoch}.pth')
    return 100 * correct / total

def test(model, test_dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for sequences, labels in test_dataloader:
            
            sequences = sequences.to('cuda')
            labels = labels.to('cuda')

            sequences = sequences.squeeze(2) 
            preds = model(sequences)

            preds_ = torch.softmax(preds, dim=2)
            temp = (preds_.argmax(dim=2).T == labels.argmax(dim=1)).float().sum()
            correct += temp.item()
            total += labels.size(0)

    print(f'Test Accuracy: {100 * correct / total}%')



if __name__ == '__main__':
    dataset = CelebAEmbeddingsDataset()

    series_length = 5
    batch_size = 128

    total_length = len(dataset)
    train_size = int(0.7 * total_length)
    val_size = int(0.15 * total_length)
    test_size = total_length - train_size - val_size


    total_indices = list(range(total_length))

    train_indices, remainder = get_random_subset(total_indices, train_size, seed=seed_value)
    val_indices, test_indices = get_random_subset(total_indices, val_size, seed=seed_value)

    assert total_length == len(val_indices) + len(test_indices) + len(train_indices)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8, pin_memory=True)


    model = Vanilla_RNN(input_size=4096, hidden_size=1024, num_layers=1, num_classes=series_length).to('cuda')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5

    batches_list = []
    loss_list = []

    for epoch in range(num_epochs):
        model.train()
        for i, (sequences, labels) in enumerate(train_dataloader):
        
            sequences = sequences.to('cuda')
            labels = labels.to('cuda')
            sequences = sequences.squeeze(2) 
            preds = model(sequences)
            
            loss = criterion(preds.squeeze(), labels.float().squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds_ = torch.softmax(preds, dim=2)
            correct = (preds_.argmax(dim=2).T == labels.argmax(dim=1)).float().sum()
            
            batches_list.append(i)
            loss_list.append(loss.item())

            if i % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}, Accuracy: {correct.item()}/{labels.size(0)} ({100 * correct.item() / (labels.size(0)):.2f}%)')

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for sequences, labels in val_dataloader:
                    
                    sequences = sequences.to('cuda')
                    labels = labels.to('cuda')

                    sequences = sequences.squeeze(2) 
                    preds = model(sequences)

                    preds_ = torch.softmax(preds, dim=2)
                    temp = (preds_.argmax(dim=2).T == labels.argmax(dim=1)).float().sum()
                    correct += temp.item()
                    total += labels.size(0)
                    
            print(f'Validation Accuracy: {100 * correct / total}%')
            torch.save(model.state_dict(), 'models/vanilla_rnn/seq5/checkpoints/epoch{epoch}.pth')

    model.eval()
    correct = 0
    total = 0


    with torch.no_grad():
        for sequences, labels in test_dataloader:
            
            sequences = sequences.to('cuda')
            labels = labels.to('cuda')

            sequences = sequences.squeeze(2) 
            preds = model(sequences)

            preds_ = torch.softmax(preds, dim=2)
            temp = (preds_.argmax(dim=2).T == labels.argmax(dim=1)).float().sum()
            correct += temp.item()
            total += labels.size(0)

    print(f'Test Accuracy: {100 * correct / total}%')
