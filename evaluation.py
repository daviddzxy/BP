import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt

from numpy.linalg import norm
from torch.utils.data import DataLoader
from torch import optim
from sklearn.model_selection import train_test_split

import networks
import datasets
import loss_functions


def euclidean_disance(x, y):
    return norm(x - y)


def most_common(arr):
    counts = np.bincount(arr)
    return np.argmax(counts)


def predict_label(encoding, encodings, labels, k):
    distances = np.zeros(len(encodings))
    for indx, x in enumerate(encodings, 0):
        distance = euclidean_disance(encoding, x)
        distances[indx] = distance

    zipped = list(zip(encodings, distances, labels))
    zipped.sort(key=lambda t: t[1])
    zipped = zipped[1:k + 1]
    labels = np.array(zipped)[:, 2].astype(int)
    result = most_common(labels)
    return result


def predict_encoding(network, dataloader):
    encodings = []
    labels = []
    with torch.no_grad():
        network.eval()
        for img0, img1, pair_label, label0, label1 in dataloader:
            img0, img1 = img0.cuda(), img1.cuda()
            out1, out2 = network(img0, img1)
            out1, out2 = out1.cpu(), out2.cpu()
            out1, out2 = out1.numpy(), out2.numpy()
            label0, label1 = label0.numpy(), label1.numpy()
            encodings.extend(out1)
            encodings.extend(out2)
            labels.extend(label0)
            labels.extend(label1)

    return np.array(encodings), np.array(labels)


def fit(epoch_count, network, loss, dataloader, learning_rate):
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    loss_history = []
    for epoch in range(0, epoch_count):
        for i, data in enumerate(dataloader, 0):
            image0, image1, pair_label, label0, label1 = data
            pair_label = pair_label.type(torch.FloatTensor)
            image0, image1, pair_label = image0.cuda(), image1.cuda(), pair_label.cuda()
            out1, out2 = network(image0, image1)
            loss_contrastive = loss(out1, out2, pair_label)
            loss_contrastive.backward()
            optimizer.step()

        print("Epoch no. {}\n \nCurrent loss {}\n".format(epoch, loss_contrastive.item()))
        loss_history.append(loss_contrastive.item())

    return loss_history


def plot_results(encodings, labels):
    color = ['red' if l == 0 else 'blue' for l in labels]
    plt.scatter(encodings[:, 0], encodings[:,1], c=color)
    plt.show()


def plot_loss(loss_history):
    plt.plot(loss_history)
    plt.show()


def main():
    path_diff_tra_ADC_BVAL_np_min_max = './Data/t2_tra_np_3D_min_max'

    images = os.listdir(path_diff_tra_ADC_BVAL_np_min_max)
    random.shuffle(images)
    train, test = train_test_split(images, test_size=0.2)

    train_dataset = datasets.SiameseNetworkDataset(path_diff_tra_ADC_BVAL_np_min_max, train, 900)
    test_dataset = datasets.SiameseNetworkDataset(path_diff_tra_ADC_BVAL_np_min_max, test, 40)

    dataset_loader_train = DataLoader(train_dataset, shuffle=1, num_workers=4, batch_size=2, drop_last=True)
    dataset_loader_test = DataLoader(test_dataset, shuffle=1, num_workers=4, batch_size=2, drop_last=True)

    network = networks.SiameseNet(networks.Net3DChannel1()).cuda()

    loss = loss_functions.ContrastiveLoss()

    loss_history = fit(50, network, loss, dataset_loader_train, 0.00005)

    plot_loss(loss_history)

    encodings, labels = predict_encoding(network, dataset_loader_train)
    plot_results(encodings, labels)

    encodings, labels = predict_label(network, dataset_loader_test)
    plot_results(encodings, labels)


if __name__ == "__main__":
    main()