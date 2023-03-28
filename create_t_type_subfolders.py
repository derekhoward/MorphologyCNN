import glob
import os
import shutil
import numpy as np
import pandas as pd


# ******************************************************************************
# Creating subfolders with T-types as names
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

print(subclasses)
# cells = list(df["cell_id"])
# labels = list(df["t-type"])
# labels = [label[:3] for label in labels]
# types = list(set(labels))

# # create subdirectories
# for subclass in subclasses:
#     os.mkdir(f"./gouwens-data/training_images_subclass_modified/{subclass}")
#     os.mkdir(f"./gouwens-data/test_images_subclass_modified/{t}")

# type_freq = {}
# for t in labels:
#     if t in type_freq:
#         type_freq[t] += 1
#     else:
#         type_freq[t] = 1

success = {}
for cell, label in zip(cells, labels):
    try:
        src = f"./gouwens-data/preprocessed_modified_images/{cell}.png"
        dst = f"./gouwens-data/training_images_subclass_modified/{label}/{cell}.png"
        shutil.copy(src, dst)
        if label in success:
            success[label] += 1
        else:
            success[label] = 1
    except:
        print(f"File not found: {cell}.png")

    # src = f"./combined-data/preprocessed_images/{cell}.png"
    # type_freq[label] -= 1
    # if type_freq[label] > 1:
    #     dst = f"./combined-data/training_images_type/{label}/{cell}.png"
    # else:
    #     # Save 2 images per class label for test set
    #     dst = f"./combined-data/test_images_type/{label}/{cell}.png"
    # shutil.copy(src, dst)

print(success)
