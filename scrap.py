
class SequenceDataset(Dataset):
    def __init__(self, 
                img_dir, filenames, sequence_length=5):
        self.img_dir = img_dir
        self.filenames = filenames
        self.sequence_length = sequence_length

        self.transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  ## other mean/std set doesnt work
        #transforms.Normalize(mean=[129.186279296875, 104.76238250732422, 93.59396362304688], std=[1, 1, 1]) 
    ])

        self.sequences, self.labels = self.generate_sequences(self.filenames, self.sequence_length)

    def generate_sequences(self, items, sequence_length):
        sequences = []
        labels = []

        for repeated_item in items:
            print("generating sequences..")
            for positions in itertools.combinations(range(sequence_length), 2):
                
                remaining_positions = [i for i in range(sequence_length) if i not in positions]
                remaining_items = [item for item in items if item != repeated_item]

                for perm in itertools.permutations(remaining_items, len(remaining_positions)):
                    sequence = [None] * sequence_length
                    sequence[positions[0]] = repeated_item
                    sequence[positions[1]] = repeated_item
                    for idx, pos in enumerate(remaining_positions):
                        
                        item_name = perm[idx] 
                        img = Image.open(Path(self.img_dir, item_name))
                        img = self.transform(img)

                        sequence[pos] = img

                    label = np.zeros(sequence_length)
                    label[positions[1]] = 1

                    sequences.append(sequence)
                    labels.append(label)

        return sequences, labels

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):

        sequence = self.sequences[index]
        label = self.labels[index]
        return (sequence, label)
