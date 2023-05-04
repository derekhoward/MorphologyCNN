import glob
import os
import shutil
import numpy as np
import pandas as pd
#import pdb


# ******************************************************************************
# Creating subfolders with subclasses as names
# ******************************************************************************

df = pd.read_csv("./combined-data/combined_metadata.csv", index_col=0)
cells, labels = [], []
subclasses = {}

for _, row in df.iterrows():
    if row["dataset"] == "gouwens":
        subclass = row["t-type"][:3]
        if subclass == "Ser":
            continue

        cells.append(row["cell_id"])
        labels.append(subclass)

        if subclass in subclasses:
            subclasses[subclass] += 1
        else:
            subclasses[subclass] = 1

# Create directory
train_dir = f"./gouwens-data/training_images_subclass"
if not os.path.isdir(train_dir):
    os.mkdir(train_dir)

subclass_cell_counts = {}
for cell, label in zip(cells, labels):
    try:
        src = f"./gouwens-data/preprocessed_images/{cell}.png"
        dst = f"./gouwens-data/training_images_subclass/{label}/{cell}.png"
        #pdb.set_trace()
        os.makedirs(f"./gouwens-data/training_images_subclass/{label}", exist_ok=True)
        shutil.copy(src, dst)

        if label in subclass_cell_counts:
            subclass_cell_counts[label] += 1
        else:
            subclass_cell_counts[label] = 1
    except:
        print(f"File not found: {cell}.png")

print(subclass_cell_counts)
