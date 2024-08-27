from pathlib import Path
from collections import defaultdict
import itertools
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
import pandas as pd


from process_model_weights import *

annotations_file='celebA_image_subset/annotations_celebA_subset.pkl'
img_labels = pd.read_pickle(annotations_file)

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

ids = np.unique(img_labels["id"])



class VGGEmbeds():
    def __init__(self, img_dir= 'celebA_image_subset/', weights = "vggface.pth", 
                 annotations_file='celebA_image_subset/annotations_celebA_subset.pkl',):
        self.img_dir = img_dir
        self.weights = weights
        self.img_labels = pd.read_pickle(annotations_file)
        self.model = nn.Sequential(*(list(vgg_face_dag(weights_path=self.weights).children())[:-1]))
        self.model.eval()
        self.model.cuda()
        
    def transform_image_forward(self, filename):

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
            ])

        image = Image.open(Path(self.img_dir, filename))
        image = transform(image).unsqueeze(0)

        if torch.cuda.is_available():
            image = image.cuda()

        return image 

    def embedding(self, filename):
        image = self.transform_image_forward(filename)
        with torch.no_grad():
            output = self.model(image)
        return output

    def whole_catergory_embeddings(self, label, ):
        embeddings = []
        df = self.img_labels[self.img_labels["id"]==label]
        for item, value in df.iterrows():
            out = self.embedding(value["filenames"])
            embeddings.append(out.detach().numpy())
        return embeddings

    def exemplar_embeddings(self, exemplars_df):
        
        embeddings = []
        for item, value in exemplars_df.iterrows():
            out = self.embedding(value["filenames"])
            embeddings.append(out[0])
        return embeddings
    


def generate_sequences(items, sequence_length=5):
    sequences = []
    labels = []
    ns = []
    pos_ = []

    for repeated_item in items:
        for positions in itertools.combinations(range(sequence_length), 2):
            
            remaining_positions = [i for i in range(sequence_length) if i not in positions]
            remaining_items = [item for item in items if item != repeated_item]

            for perm in itertools.permutations(remaining_items, len(remaining_positions)):
                sequence = [None] * sequence_length
                sequence[positions[0]] = repeated_item
                sequence[positions[1]] = repeated_item

                n = positions[1] - positions[0]

                for idx, pos in enumerate(remaining_positions):
                    sequence[pos] = perm[idx]

                label = np.zeros(sequence_length)
                label[positions[1]] = 1

                sequences.append(sequence)
                labels.append(label)
                ns.append(n)
                pos_.append(positions)

    return sequences, labels, pos_, ns


def custom_collate(batch):
    sequences, labels, positions, ns = zip(*batch)
    return list(sequences), list(labels), list(positions), list(ns)

# Create a custom dataset
class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels, positions, ns):
        self.sequences = sequences
        self.labels = labels
        self.positions = positions
        self.ns = ns
        self.filename_sequences = self.sequences.copy()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.positions[idx], self.ns[idx]


def uniform_random_samples(values, num_samples, seed=88):

    random.seed(seed)
    
    value_indices = defaultdict(list)
    for index, value in enumerate(values):
        value_indices[value].append(index)

    unique_values = list(value_indices.keys())
    num_unique_values = len(unique_values)

    samples = []
    for _ in range(num_samples):
        # If we've used all unique values, reset the list
        if not unique_values:
            unique_values = list(value_indices.keys())
        
        # Randomly select a value
        selected_value = random.choice(unique_values)
        unique_values.remove(selected_value)
        
        # Randomly select an index from the chosen value's indices
        selected_index = random.choice(value_indices[selected_value])
        samples.append(selected_index)
    
    return samples
## create sequences and save as embeddings 

def create_and_save_embeddings(items, sequence_length, checkpoint=1, save_dir="celebA_embeddings"):
    path = Path(f"{save_dir}/seq{sequence_length}")
    backbone = VGGEmbeds()

    sequences, labels = generate_sequences(items)
    filenames = [f"{i:06d}.pt" for i in range(checkpoint, len(labels) + 1)]
    
    if checkpoint > 1:
        sequences = sequences[checkpoint-1:]
        labels = labels[checkpoint-1:]
    else: 
        df = pd.DataFrame()
        df["labels"] = labels
        df["filenames"] = filenames
        df.to_parquet(path / "labels.parquet")

    assert len(filenames) == len(sequences)

    for s, filename in zip(sequences, filenames):

        print(filename)
        tensor =  torch.stack([backbone.embedding(file) for file in s])

        torch.save(tensor, path / filename)

if __name__ == '__main__':

    sequence_length = 7
    sequences, labels, positions, ns = generate_sequences(files, sequence_length=sequence_length)

    samples = uniform_random_samples(values, 15000)
    print(Counter(np.array(ns)[samples]))

    sequences = np.array(sequences)[samples]
    labels = np.array(labels)[samples]
    positions = np.array(positions)[samples]
    ns = np.array(ns)[samples]

    save_dir = "celebA_embeddings"
    path = Path(f"{save_dir}/seq{sequence_length}")
    path.mkdir(exist_ok=True, parents=True)
    print(path)

    backbone = VGGEmbeds()

    checkpoint = 1
    filenames = [f"{i:06d}.pt" for i in range(checkpoint, len(labels) + 1)]
    print(f"Len filenames: {len(filenames)}")

    if checkpoint > 1:
        sequences = sequences[checkpoint-1:]
        labels = labels[checkpoint-1:]
        positions = positions[checkpoint-1:]
        ns = ns[checkpoint-1:]

    assert len(filenames) == len(sequences) == len(labels)


    dataset = EmbeddingDataset(sequences, labels, positions, ns)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True, collate_fn=custom_collate)

    # Process data in batches
    with torch.no_grad():  # Disable gradient calculation
        for i, (batch_sequences, batch_labels, batch_positions, batch_ns) in enumerate(tqdm(dataloader)):
            batch_tensors = []
            for sequence in batch_sequences:
                sequence_tensors = []
                for file in sequence:
                    # if isinstance(file, str):
                    # #     # Assume file is a path to an image, load it here
                    # #     # You might need to adjust this based on your actual data format
                    # #     img = load_image(file)  # You need to implement this function
                    # #     if torch.cuda.is_available():
                    # #         img = img.cuda()
                    # # else:
                    # #     img = file
                    # #     if torch.cuda.is_available():
                    # #         img = img.cuda()
                    sequence_tensors.append(backbone.embedding(file))
                batch_tensors.append(torch.stack(sequence_tensors))
            
            batch_tensor = torch.stack(batch_tensors)
            
            for j, tensor in enumerate(batch_tensor):
                idx = i * len(batch_sequences) + j
                saver = {
                    "sequence": tensor.cpu(),  # Move back to CPU for saving
                    "label": batch_labels[j],
                    "positions": batch_positions[j],
                    "n-distance": batch_ns[j]
                }
                torch.save(saver, path / filenames[idx])