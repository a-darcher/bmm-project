
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

        # Define the weights
        self.W_xh = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

        self.activ = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size, sequence_length, _ = x.size()

        # Initialize hidden state with zeros
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        output = []
        for t in range(sequence_length):
            # Get the input at this time step
            x_t = x[:, t, :]

            # Compute the hidden state
            h = self.activ(self.W_xh(x_t) + self.W_hh(h) + self.b_h)
            output.append(h)
        return torch.stack(output, dim=1)


class Vanilla_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(Vanilla_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.rnn = RNN(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.rnn(x)
        outputs = self.fc(out[:, -1, :])  # Get the output at the last time step
        return outputs.unsqueeze(1)  # Add the missing dimension

        
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