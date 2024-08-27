from pathlib import Path
import argparse
import itertools
from PIL import Image

import random

from collections import Counter
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.io import read_image
from torchvision import transforms

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

from tqdm import tqdm

from process_model_weights import *
from celeb_backbone import *


# construct file indexer

classes = list(range(0,10))

original_files = [
 "057876.jpg", #002514.jpg",
 "039593.jpg", #
 "094741.jpg", #
 "152801.jpg", #
 "126154.jpg", #
 "034026.jpg", #
 "012857.jpg", #
 "016242.jpg", #
 "153728.jpg", #
 "068982.jpg",]

older_files = [
    "002514.jpg",     
    "017554.jpg",     
    "093689.jpg",     
    "052500.jpg",     
    "162586.jpg",     
    "120490.jpg",     
    "048720.jpg",     
    "031455.jpg",     
    "036953.jpg",     
    "158002.jpg",     
]


younger_files = [
    "119296.jpg",     
    "029394.jpg",     
    "014392.jpg",     
    "055349.jpg",     
    "040676.jpg",     
    "042867.jpg",     
    "052814.jpg",     
    "114366.jpg",     
    "041553.jpg",     
    "117606.jpg",     
]

df = pd.DataFrame()
df["classes"] = classes
df["og_files"] = original_files 
df["older_files"] = older_files
df["younger_files"] = younger_files


def alter_sequences(df, sequence_length):
    sequences, labels, positions, ns = generate_sequences(df['og_files'], sequence_length)

    for i in tqdm(range(len(sequences))):

        og_file = sequences[i][positions[i][0]]
        row = df.loc[df["og_files"] == og_file].iloc[0]
        older_filename = row["older_files"]
        younger_filename = row["younger_files"]

        sequences[i][positions[i][0]] = younger_filename
        sequences[i][positions[i][1]] = older_filename

    return sequences
