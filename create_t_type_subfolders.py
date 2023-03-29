import glob
import os
import shutil
import numpy as np
import pandas as pd


# ******************************************************************************
# Creating subfolders with T-types as names
# ******************************************************************************

df = pd.read_csv("./gouwens-data/filtered_t_types.csv", index_col=0)

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
for _, row in df.iterrows():
    if row["t-type"] in sig_cell_types and os.path.exists(f"./gouwens-data/preprocessed_images/{row['Specimen ID']}.png"):
        cells.append(row["Specimen ID"])
        labels.append(row["T-type"])
        if row["T-type"] not in types:
            types.add(row["T-type"])

# create subdirectories
for t in types:
    os.mkdir(f"./gouwens-data/training_images_t_type/{t}")
    os.mkdir(f"./gouwens-data/test_images_t_type/{t}")

type_freq = {}
for t in labels:
    if t in type_freq:
        type_freq[t] += 1
    else:
        type_freq[t] = 1

# create subdirectories
train_dir = f"./gouwens-data/training_images_t_type"
test_dir = f"./gouwens-data/test_images_t_type"
if not os.path.isdir(train_dir):
    os.mkdir(train_dir)
if not os.path.isdir(test_dir):
    os.mkdir(test_dir)

for t in types:
    train_subdir = f"./gouwens-data/training_images_t_type/{t}"
    test_subdir = f"./gouwens-data/test_images_t_type/{t}"
    if not os.path.isdir(train_subdir):
        os.mkdir(train_subdir)
    if not os.path.isdir(test_subdir):
        os.mkdir(test_subdir)

test_cell_count = 2     # Number of cells per type in the test dataset

cnt = 0
for cell, label in zip(cells, labels):
    src = f"./gouwens-data/preprocessed_images/{cell}.png"
    
    if type_freq[label] > test_cell_count:
        dst = f"./gouwens-data/training_images_t_type/{label}/{cell}.png"
    else:
        # Save 2 images per class label for test set
        dst = f"./gouwens-data/test_images_t_type/{label}/{cell}.png"

    shutil.copy(src, dst)
    cnt += 1
    type_freq[label] -= 1

print(f"Created {len(types)} subfolders with {cnt} cells.")
