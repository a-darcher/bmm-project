from pathlib import Path
import pandas as pd



weights_path = Path("PyVGGFace/models/vggface.pth")

gt_path = Path("archive/identity_CelebA.txt")
labels = pd.read_csv('archive/identity_CelebA.txt', sep=" ", header=None)
labels.columns = ["filename", "id"]

file_path = Path("vgg_face_torch/names.txt")
with open(file_path, 'r') as file:
    names_list = file.readlines()
names_list = [name.strip() for name in names_list]