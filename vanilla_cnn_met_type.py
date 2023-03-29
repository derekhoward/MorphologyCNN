import glob
import os
import copy
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

torch.manual_seed(2)

class VanillaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3)
        self.conv1_BN = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, 4)
        self.conv2_BN = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(4, 4)
        self.fc1 = nn.Linear(8 * 27 * 27, 512)
        self.fc2 = nn.Linear(512, 18)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1_BN(self.conv1(x))))
        x = self.pool2(F.relu(self.conv2_BN(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CustomTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angle=0, flip=''):
        self.angle = angle
        self.flip = flip

    def __call__(self, x):
        x = transforms.functional.rotate(x, self.angle)
        if self.flip == 'h':
            x = transforms.functional.hflip(x)
        elif self.flip == 'v':
            x = transforms.functional.vflip(x)
        return x

def getDataLoader(input_dir, batch_size, augment):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    data = datasets.ImageFolder(input_dir, transform=transform)
    train_size = int(0.8 * len(data))
    valid_size = len(data) - train_size
    train_data, valid_data = torch.utils.data.random_split(
        data, [train_size, valid_size], generator=torch.Generator().manual_seed(0)
    )
    if augment:
        augment_train, augment_valid = [], []
        for angle in [0, 180]:
            for flip in ['']:
                custom_transform = CustomTransform(angle=angle, flip=flip)
                augment_t = copy.deepcopy(train_data)
                augment_v = copy.deepcopy(valid_data)
                augment_t.dataset.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(custom_transform)
                ])
                augment_v.dataset.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(custom_transform)
                ])
                augment_train.append(augment_t)
                augment_valid.append(augment_v)

        augment_train_data = torch.utils.data.ConcatDataset(augment_train)
        augment_valid_data = torch.utils.data.ConcatDataset(augment_valid)
        print(augment_train_data.__len__())
        print(augment_valid_data.__len__())
        train_loader = DataLoader(augment_train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(augment_valid_data, batch_size=batch_size, drop_last=True)
    else:   
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_data, batch_size=batch_size, drop_last=True)

    return train_loader, valid_loader, data.classes

def getTestDataLoader(test_dir, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    data = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return test_loader

def one_hot(label, num_classes):
    one_hot_label = np.zeros(num_classes)
    one_hot_label[label] = 1.0
    return one_hot_label

def accuracy(output, labels):
    correct = 0
    for i, p in enumerate(output):
        pred = np.argmax(p.flatten())
        if labels[i] == pred:
            correct += 1
    return correct / len(labels)

def mrr(output, labels):
    score_sum = 0
    count = 0
    for i, p in enumerate(output):
        pred = p.detach().numpy()
        argsorted_output = np.argsort(-pred).tolist()
        label_rank = argsorted_output.index(labels[i]) + 1
        mrr = 1 / label_rank
        score_sum += mrr
        count += 1
    return score_sum, count

def plot_results(num_epochs, y_train, y_valid, metric, outfile):
    x_values = np.arange(num_epochs)

    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    plt.plot(x_values, y_train, label="train")
    plt.plot(x_values, y_valid, label="valid")
    plt.title(f"{metric} (t-type) for Current CNN")
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend()
    plt.savefig(outfile)
    plt.clf()

def train(input_dir, model, loss_fn, optimizer, batch_size, num_epochs, augment=False, test_dir=None, test_batch_size=None):
    train_loader, valid_loader, classes = getDataLoader(input_dir, batch_size=batch_size, augment=augment)
    train_acc, valid_acc = [], []
    train_loss, valid_loss = [], []
    cm_inputs, cm_outputs, cm_labels = [], [], []
    time_start = time()
    
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch + 1}")
        t_acc, t_loss = [], []
        for i, data in enumerate(train_loader):
            inputs, labels = data

            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output.squeeze(), labels)
            
            loss.backward()
            optimizer.step()

            t_acc.append(accuracy(output.data, labels))
            t_loss.append(loss.item())
        
        train_acc.append(np.average(t_acc))
        train_loss.append(np.average(t_loss))
        print(f"\tTraining accuracy: {np.average(t_acc)}")
        print(f"\tTraining loss: {np.average(t_loss)}")

        # Validation
        v_acc, v_loss = [], []
        v_mrr_sum, v_mrr_count = 0, 0
        for i, data in enumerate(valid_loader):
            inputs, labels = data
            output = model(inputs)
            loss = loss_fn(output.squeeze(), labels)

            v_acc.append(accuracy(output.data, labels))
            v_loss.append(loss.data)
            score_sum, count = mrr(output.data, labels)
            v_mrr_sum += score_sum
            v_mrr_count += count

            if epoch == num_epochs - 1:
                for i in range(len(labels)):
                    cm_inputs.append(inputs[i].numpy())
                    cm_labels.append(labels[i].numpy().item())
                    cm_outputs.append(np.argmax(output.data[i].flatten()).item())
                            
        valid_acc.append(np.average(v_acc))
        valid_loss.append(np.average(v_loss))
        print(f"\tValidation accuracy: {np.average(v_acc)}")
        print(f"\tValidation loss: {np.average(v_loss)}")
        print(f"\tValidation MRR: {np.average(v_mrr_sum / v_mrr_count)}")

        # Confusion matrix
        if epoch == num_epochs - 1:
            cm = confusion_matrix(cm_labels, cm_outputs, labels=list(range(len(classes))))
            cmn = (cm.astype('float') + 0.00001 * cm.astype('float')) / (cm.sum(axis=1) + 0.00001)[:, np.newaxis]
            fig, ax = plt.subplots(figsize=(12, 10))
            hm = sns.heatmap(cmn, yticklabels=classes)
            hm.set_xticklabels(classes, size=17, rotation=90)
            hm.set_yticklabels(classes, size=17)
            cbar = hm.collections[0].colorbar
            cbar.ax.tick_params(labelsize=17)
            plt.subplots_adjust(bottom=0.28, left=0.28)
            plt.savefig("./gouwens-data/plots/vanilla_cm_met_type_modified.png")
    
    print(f"Training time: {round(time() - time_start, 1)}s")

    # Uncomment below to plot accuracy and loss over epochs
    # plot_results(num_epochs, train_acc, valid_acc, "Accuracy", "./gouwens-data/plots/vanilla_accuracy_t_type.png")
    # plot_results(num_epochs, train_loss, valid_loss, "Loss", "./gouwens-data/plots/vanilla_loss_t_type.png")

    # Uncomment below to save model weights
    # model_path = './gouwens-data/models/vanilla_cnn_gouwens_met_type_modified_b2_lr1e-4_e7_rs2_512'
    # torch.save(model.state_dict(), model_path)

    if test_dir:
        print("Running model on test set ...")
        test_acc = []
        test_loader = getTestDataLoader(test_dir, test_batch_size)
        for i, data in enumerate(test_loader):
            inputs, labels = data
            output = model(inputs)
            test_acc.append(accuracy(output.data, labels))
        print("Test accuracy: ", np.average(test_acc))

def main():
    torch.autograd.set_detect_anomaly(True)
    input_dir = "./gouwens-data/training_images_met_type/"

    # hyperparameters
    batch_size = 2
    learning_rate = 0.0001
    num_epochs = 7

    # model
    model = VanillaCNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    augment = False

    # test_dir = "./combined-data/test_images_type/"
    train(input_dir, model, loss_fn, optimizer, batch_size, num_epochs, augment)


if __name__ == "__main__":
    main()