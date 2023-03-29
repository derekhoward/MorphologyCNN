import glob
import os
import shutil
import numpy as np
import pandas as pd

# ******************************************************************************
# Create subfolders per dataset with labels as names
# ******************************************************************************

df = pd.read_csv("./combined-data/combined_metadata.csv", index_col=0)
dataset_df = {
    "gouwens": df[df["dataset"] == "gouwens"],
    "scala": df[df["dataset"] == "scala"]
}

for dataset in ["gouwens", "scala"]:

    # Identify cell types with 10+ cells
    cell_type_counts = {}
    for _, row in dataset_df[dataset].iterrows():
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
    
    print("Dataset: ", dataset)
    print("Significant cell types: ", len(sig_cell_types))
    print("Total cells: ", total_cells)

    cells, labels = [], []
    types = set()
    for _, row in dataset_df[dataset].iterrows():
        if row["t-type"] in sig_cell_types:
            cells.append(row["cell_id"])
            labels.append(row["t-type"])
            if row["t-type"] not in types:
                types.add(row["t-type"])

    # create subdirectories
    train_dir = f"./{dataset}-data/training_images_t_type"
    test_dir = f"./{dataset}-data/test_images_t_type"
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    if not os.path.isdir(test_dir):
        os.mkdir(test_dir)

    for t in types:
        train_subdir = f"./{dataset}-data/training_images_t_type/{t}"
        test_subdir = f"./{dataset}-data/test_images_t_type/{t}"
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

    for cell, label in zip(cells, labels):
        src = f"./{dataset}-data/preprocessed_images/{cell}.png"
        if type_freq[label] > test_cell_count:
            # Save to train subdirectory
            dst = f"./{dataset}-data/training_images_t_type/{label}/{cell}.png"
        else:
            # Save to test subdirectory
            dst = f"./{dataset}-data/test_images_t_type/{label}/{cell}.png"

        shutil.copy(src, dst)
        type_freq[label] -= 1
