import torch
import platform
import numpy as np
import random
import os
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch import optim
from sklearn.model_selection import train_test_split

import networks
import datasets
import loss_functions


def eval_loop(network, dataloader):
    out_list = []
    with torch.no_grad():
        embed_network = network
        embed_network.eval()
        for img0, img1, pair_label, label0, label1 in dataloader:
            img0, img1 = img0.cuda(), img1.cuda()
            out1, out2 = network.forward(img0, img1)
            out_list.append([out1.cpu().numpy(), out2.cpu().numpy(), label0.cpu().numpy(), label1.cpu().numpy()])

    return out_list

def train_loop(epoch_count, network, loss, dataloader):
    optimizer = optim.Adam(network.parameters(), lr=0.00005)

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


        print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
        loss_history.append(loss_contrastive.item())

    return loss_history



def plot_result(result_list):
    for item in result_list:
        x0, y0 = zip(*item[0])
        x1, y1 = zip(*item[1])
        label0 = item[2][0]
        label1 = item[3][0]

        point0 = 'co' if label0 == 0 else 'bo'
        point1 = 'co' if label1 == 0 else 'bo'

        plt.xlabel('dim1')
        plt.ylabel('dim2')
        plt.plot(x0, y0, point0)
        plt.plot(x1, y1, point1)

    plt.show()




def main():
    path_t2_tra_np_min_max = './Data/t2_tra_np_min_max'
    path_diff_tra_ADC_BVAL_np_min_max = './Data/diff_ADC_BVAL_np_min_max'

    images = os.listdir(path_t2_tra_np_min_max)
    #images = os.listdir(path_diff_tra_ADC_BVAL_np_min_max)
    random.shuffle(images)
    train, test = train_test_split(images, test_size=0.2)


    """"
    images = os.listdir(path_t2_tra_np_min_max)

    dataset = datasets.SiameseNetworkDataset(path_t2_tra_np_min_max, images)   # bude dobre tam pasovat cely zoznnam dopred, aby som
                                                              # mohol rozdelit trenovacie data a validacne
    network = networks.SiameseNet().cuda()

    loss = loss_functions.ContrastiveLoss()

    train_loop(70, network, loss, dataset)

    """

    #train_dataset = datasets.SiameseNetworkDataset(path_diff_tra_ADC_BVAL_np_min_max, train, 50)
    #test_dataset = datasets.SiameseNetworkDataset(path_diff_tra_ADC_BVAL_np_min_max, test, 15)

    train_dataset = datasets.SiameseNetworkDataset(path_t2_tra_np_min_max, train, 50)
    test_dataset = datasets.SiameseNetworkDataset(path_t2_tra_np_min_max, test, 15)

    dataset_loader = DataLoader(train_dataset, shuffle=1, num_workers=8, batch_size=8)

    #network = networks.Channel2SiameseNet().cuda()
    network = networks.SiameseNet().cuda()

    loss = loss_functions.ContrastiveLoss()

    train_loop(120, network, loss, dataset_loader)

    dataset_train_loader = DataLoader(train_dataset, shuffle=1, num_workers=8, batch_size=1)
    dataset_test_loader = DataLoader(test_dataset, shuffle=1, num_workers=8, batch_size=1)

    train_result = eval_loop(network, dataset_train_loader)
    test_result = eval_loop(network, dataset_test_loader)

    plot_result(train_result)
    plot_result(test_result)


if __name__ == "__main__":
    main()