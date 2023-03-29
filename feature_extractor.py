import glob
import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import models

from vanilla_cnn_type import VanillaCNN

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x


def main():
    # change file/directory paths to t_type for T-type feature extraction
    torch.autograd.set_detect_anomaly(True)
    model_path = "gouwens-data/models/vanilla_cnn_gouwens_met_type_modified_b2_lr1e-4_e7_rs2_512"

    # model
    model = VanillaCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Metadata containing 44 extracted features from Gouwens dataset
    metadata = pd.read_csv("./gouwens-data/extracted_features.csv")

    # Collect image inputs to run through the model
    test_cells = set()
    files = glob.glob("./gouwens-data/training_images_met_type/*/*.png") + glob.glob("./gouwens-data/test_images_met_type/*/*.png")
    for file in files:
        test_cells.add(int(os.path.splitext(os.path.basename(file))[0]))
    print(f"Test dataset: {len(test_cells)} cells.")

    trans = transforms.ToTensor()
    
    cell_id, cell_type, embeddings = [], [], []
    for _, row in metadata.iterrows():
        if row['Specimen ID'] in test_cells:
            # Inference of Gouwens images through the model
            img = Image.open(f"./gouwens-data/preprocessed_images/{row['Specimen ID']}.png")
            output = model(trans(img).unsqueeze(0))
            embeddings.append(output.squeeze().detach().numpy())
            cell_type.append(row['MET type'])   # Column names: "MET type" or "T type"
            cell_id.append(row['Specimen ID'])

    cell_id = np.array(cell_id)
    cell_type = np.array(cell_type)
    embeddings = np.array(embeddings)

    embeddings_df = pd.DataFrame(embeddings, columns=[f"Node {i}" for i in range(embeddings.shape[1])])
    embeddings_df["Specimen ID"] = cell_id
    embeddings_df = embeddings_df.set_index("Specimen ID")
    print(embeddings_df.head())
    embeddings_df.to_csv("./gouwens-data/met_type_embeddings.csv")


if __name__ == "__main__":
    main()