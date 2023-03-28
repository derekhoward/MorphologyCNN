import glob
import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from PIL import Image
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

torch.manual_seed(24)

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
        data, [train_size, valid_size], generator=torch.Generator().manual_seed(20)
    )
    if augment:
        augment_train, augment_valid = [], []
        for angle in [0, 90, 180, 270]:
            for flip in ['h', 'v', '']:
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
    plt.title(f"{metric} for Transfer Learning")
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend()
    plt.savefig(outfile)
    plt.clf()

def train(input_dir, model, loss_fn, optimizer, batch_size, num_epochs, augment=False, test_dir=None, test_batch_size=None):
    train_loader, valid_loader, classes = getDataLoader(input_dir, batch_size=batch_size, augment=augment)
    train_acc, valid_acc = [], []
    train_loss, valid_loss = [], []
    cm_inputs, cm_output, cm_labels = [], [], []
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            optimizer.step()

            t_acc.append(accuracy(output.data, labels))
            t_loss.append(loss.item())
        
        train_acc.append(np.average(t_acc))
        train_loss.append(np.average(t_loss))
        print(f"\tTraining accuracy: {np.average(t_acc)}")
        print(f"\tTraining loss: {np.average(t_loss)}")

        # Validation
        v_acc, v_loss = [], []
        for i, data in enumerate(valid_loader):
            inputs, labels = data
            output = model(inputs)
            loss = loss_fn(output.squeeze(), labels)

            v_acc.append(accuracy(output.data, labels))
            v_loss.append(loss.data)
        
        valid_acc.append(np.average(v_acc))
        valid_loss.append(np.average(v_loss))
        print(f"\tValidation accuracy: {np.average(v_acc)}")
        print(f"\tValidation loss: {np.average(v_loss)}")
    
    print(f"Training time: {round(time() - time_start, 1)}s")
    plot_results(num_epochs, train_acc, valid_acc, "Accuracy", "./gouwens-data/plots/vanilla_accuracy_t_type_vgg16.png")
    plot_results(num_epochs, train_loss, valid_loss, "Loss", "./gouwens-data/plots/vanilla_loss_t_type_vgg16.png")
    

def main():
    torch.autograd.set_detect_anomaly(True)
    input_dir = "./gouwens-data/training_images_t_type"

    # hyperparameters
    batch_size = 4
    learning_rate = 0.01
    num_epochs = 20

    augment = False

    # # ResNet
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    
    # # freeze params
    # for param in model.parameters():
    #     param.requires_grad = False 
    
    # for param in model.layer4.parameters():
    #     param.requires_grad = True

    # fc = model.fc
    # new_fc = nn.Sequential(
    #     nn.Linear(fc.in_features, 128),
    #     nn.ReLU(),
    #     nn.Linear(128, 5)
    # )
    # model.fc = new_fc
    # print(model)

    # VGG16
    model = models.vgg16(pretrained=True)
    # freeze params
    for i, param in enumerate(model.parameters()):
        param.requires_grad = False 
        if i > 20:
            param.requires_grad = True

    in_dim = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_dim, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3, inplace=False),
        nn.Linear(4096, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3, inplace=False),
        nn.Linear(512, 20)
    )
    # print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(input_dir, model, loss_fn, optimizer, batch_size, num_epochs, augment)


if __name__ == "__main__":
    main()