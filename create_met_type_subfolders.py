import glob
import os
import shutil
import numpy as np
import pandas as pd


# ******************************************************************************
# Creating subfolders with MET-types as names
# ******************************************************************************

df = pd.read_csv("./gouwens-data/filtered_met_types.csv")

# Identify cell types with 10+ cells
cell_type_counts = {}
for _, row in df.iterrows():
    if os.path.exists(f"./gouwens-data/preprocessed_modified_images_360/{row['Specimen ID']}.png"):
        if row["MET type"] not in cell_type_counts:
            cell_type_counts[row["MET type"]] = 1
        else:
            cell_type_counts[row["MET type"]] += 1

sig_cell_types = set()
total_cells = 0
for cell_type, count in cell_type_counts.items():
    if count >= 10:
        sig_cell_types.add(cell_type)
        total_cells += count

print("Significant MET-types: ", len(sig_cell_types))
print("Total cells: ", total_cells)

cells, labels = [], []
types = set()
for _, row in df.iterrows():
    if row["MET type"] in sig_cell_types and os.path.exists(f"./gouwens-data/preprocessed_modified_images/{row['Specimen ID']}.png"):
        cells.append(row["Specimen ID"])
        labels.append(row["MET type"])
        if row["MET type"] not in types:
            types.add(row["MET type"])

# create subdirectories
train_dir = f"./{dataset}-data/training_images_met_type"
test_dir = f"./{dataset}-data/test_images_met_type"
if not os.path.isdir(train_dir):
    os.mkdir(train_dir)
if not os.path.isdir(test_dir):
    os.mkdir(test_dir)

for t in types:
    train_subdir = f"./gouwens-data/training_images_met_type/{t}"
    test_subdir = f"./gouwens-data/test_images_met_type/{t}"
    if not os.path.isdir(train_subdir):
        os.mkdir(train_subdir)
    if not os.path.isdir(test_subdir):
        os.mkdir(test_subdir)

type_freq = {}
for t in labels:
    if t in type_freq:
        type_freq[t] += 1
    else:
        type_freq[t] = 1

test_cell_count = 2     # Number of cells per type in the test dataset

cnt = 0
for cell, label in zip(cells, labels):
    src = f"./gouwens-data/preprocessed_modified_images/{cell}.png"
    
    if type_freq[label] > test_cell_count:
        dst = f"./gouwens-data/training_images_t_type_modified/{label}/{cell}.png"
    else:
        # Save 2 images per class label for test set
        dst = f"./gouwens-data/test_images_t_type_modified/{label}/{cell}.png"

    shutil.copy(src, dst)
    cnt += 1
    type_freq[label] -= 1

print(f"Created {len(types)} subfolders with {cnt} cells.")
