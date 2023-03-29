import glob
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models

from vanilla_cnn_t_type import VanillaCNN

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x


def main():
    torch.autograd.set_detect_anomaly(True)

    # Load trained model weights
    model_path = "gouwens-data/models/vanilla_cnn_gouwens_t_type_modified_b2_lr1e-4_e4_rs7_256"

    # model
    model = VanillaCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    metadata = pd.read_csv("gouwens-data/filtered_t_types.csv")

    types = [
        'Lamp5 Lsp1', 
        'Lamp5 Plch2', 
        'Pvalb Reln', 
        'Pvalb Sema3e', 
        'Pvalb Tpbg', 
        'Sncg Vip', 
        'Sst Calb2', 
        'Sst Chodl', 
        'Sst Crhr2', 
        'Sst Esm1', 
        'Sst Hpse', 
        'Sst Mme', 
        'Sst Myh8', 
        'Sst Nts', 
        'Sst Rxfp1', 
        'Sst Tac1', 
        'Sst Tac2', 
        'Vip Crispld2', 
        'Vip Lmo1', 
        'Vip Pygm'
    ]

    test_cells = {}
    for _, row in metadata.iterrows():
        if row["dataset"] == "scala" and row["t-type"] in types:
            test_cells[row["cell_id"]] = types.index(row["t-type"])

    trans = transforms.ToTensor()
    
    predictions, labels = [], []
    correct = 0
    mrr_sum = 0.
    cell_label_pred = {}
    for cell, label in test_cells.items():
        try:
            img = Image.open(f"./scala-data/preprocessed_images/{cell}.png")
            output = model(trans(img).unsqueeze(0))
            output = output.detach().numpy()[0]
            pred = np.argmax(output)
            argsorted_output = np.argsort(-output).tolist()
            label_rank = argsorted_output.index(label) + 1
            if pred == label:
                correct += 1
            mrr_sum += 1 / label_rank

            predictions.append(pred)
            labels.append(label)

            if types[label] not in cell_label_pred:
                cell_label_pred[types[label]] = [(cell, pred, label_rank)]
            else:
                cell_label_pred[types[label]].append((cell, pred, label_rank))
            
        except:
            print(f"Image file not found for cell: {cell}")

    print(f"Correct: {correct} / Total: {len(test_cells)}")
    print(f"Scala test accuracy: {correct / len(test_cells)}")
    print(f"Scala MRR (all): {mrr_sum / len(test_cells)}")

    for label, v in cell_label_pred.items():
        print("T-type: ", label)
        corr_cells, wrong_cells = [], []
        for cell, pred, label_rank in v:
            if pred == types.index(label):
                corr_cells.append(cell)
            else:
                wrong_cells.append((cell, pred, label_rank))
        corr_count = len(corr_cells)
        wrong_count = len(wrong_cells)
        print(f"\tAccuracy: {corr_count / (corr_count + wrong_count)} ({corr_count} / {corr_count + wrong_count})")

        # Uncomment to print correct, wrong, and rank of predictions per T-type
        
        # for cc in corr_cells:
        #     print(f"\t\tCorrect: {cc}")
        # for wc, wp, rank in wrong_cells:
        #     # print(f"\t\tWrong: {wc}\tPredicted: {types[wp]}\tLabel Rank: {rank}")
        #     print(f"\t\tWrong: {wc}\tPredicted: {subclasses[wp]}\tLabel Rank: {rank}")

    # Generate confusion matrix for inference
    cm = confusion_matrix(labels, predictions, labels=list(range(len(types))))
    cmn = (cm.astype('float') + 0.00001 * cm.astype('float')) / (cm.sum(axis=1) + 0.00001)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(12, 10))
    hm = sns.heatmap(cmn, yticklabels=types)
    hm.set_xticklabels(types, size=17, rotation=90)
    hm.set_yticklabels(types, size=17)
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=17)
    plt.subplots_adjust(bottom=0.28, left=0.28)
    plt.savefig("./scala-data/plots/inference_t_type.png")


if __name__ == "__main__":
    main()