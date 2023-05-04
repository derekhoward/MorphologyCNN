import glob
import os
from pathlib import Path
import shutil
import numpy as np
import pandas as pd


# ******************************************************************************
# Creating subfolders with T-types as names
# ******************************************************************************
RSTATE = 5
df = pd.read_csv("./gouwens-data/filtered_t_types.csv")#, index_col=0)

# Identify cell types with 10+ cells
cell_type_counts = {}
for _, row in df.iterrows():
    if os.path.exists(f"./gouwens-data/preprocessed_images/{row['Specimen ID']}.png"):
        if row["t-type"] not in cell_type_counts:
            cell_type_counts[row["t-type"]] = 1
        else:
            cell_type_counts[row["t-type"]] += 1

sig_cell_types = set()
total_cells = 0
for cell_type, count in cell_type_counts.items():
    if count >= 10:
        sig_cell_types.add(cell_type)
        total_cells += count

print("Significant T-types: ", len(sig_cell_types))
print("Total cells: ", total_cells)

cells, labels = [], []
types = set()
# add a line for shuffling the dataset (so you can get multiple versions of this?)
df = df.sample(frac=1, random_state=RSTATE).reset_index(drop=True)
for _, row in df.iterrows():
    if row["t-type"] in sig_cell_types and os.path.exists(f"./gouwens-data/preprocessed_images/{row['Specimen ID']}.png"):
        cells.append(row["Specimen ID"])
        labels.append(row["t-type"])
        if row["t-type"] not in types:
            types.add(row["t-type"])

# create subdirectories
train_dir = Path(f"./gouwens-data/training_images_t_type_{RSTATE}")
test_dir = Path(f"./gouwens-data/test_images_t_type_{RSTATE}")
train_dir.mkdir(exist_ok=True)
test_dir.mkdir(exist_ok=True)

for t in types:
    #os.mkdir(f"./gouwens-data/training_images_t_type/{t}")
    #os.mkdir(f"./gouwens-data/test_images_t_type/{t}")
    ttrain = train_dir / t
    ttrain.mkdir(exist_ok=True, parents=True)
    ttest = test_dir / t
    ttest.mkdir(exist_ok=True, parents=True)
    
    
type_freq = {}
for t in labels:
    if t in type_freq:
        type_freq[t] += 1
    else:
        type_freq[t] = 1

test_cell_count = 2     # Number of cells per type in the test dataset

cnt = 0
for cell, label in zip(cells, labels):
    src = f"./gouwens-data/preprocessed_images/{cell}.png"
    
    if type_freq[label] > test_cell_count:
        dst = train_dir / label / (str(cell) + '.png')
        #dst = f"./gouwens-data/training_images_t_type/{label}/{cell}.png"
    else:
        # Save 2 images per class label for test set
        #dst = f"./gouwens-data/test_images_t_type/{label}/{cell}.png"
        dst = test_dir / label / (str(cell) + '.png')

    shutil.copy(src, dst)
    cnt += 1
    type_freq[label] -= 1

print(f"Created {len(types)} subfolders with {cnt} cells.")
