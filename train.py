import torch
import platform
import numpy as np
import os
from torch.utils.data import DataLoader
from torch import optim

import matplotlib.pyplot as plt

import networks
import datasets
import loss_functions

def train_loop(epoch_count, network, loss, dataset, dataloader):
    dataset_loader = DataLoader(dataset, shuffle=1, num_workers=8, batch_size=16)
    optimizer = optim.Adam(network.parameters(), lr=0.0001)

    counter = []
    loss_history = []
    iteration_number = 0

    for epoch in range(0, epoch_count):
        for i, data in enumerate(dataset_loader, 0):
            image0, image1, pair_label, label0, label1 = data
            pair_label = pair_label.type(torch.FloatTensor)
            image0, image1, pair_label = image0.cuda(), image1.cuda(), pair_label.cuda()
            out1, out2 = network(image0, image1)
            loss_contrastive = loss(out1, out2, pair_label)
            loss_contrastive.backward()
            optimizer.step()

        print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
        iteration_number += 10
        counter.append(iteration_number)
        loss_history.append(loss_contrastive.item())



def main():
    if platform.system() == 'Windows':
        path_t2_tra_np_min_max = '.\Data\\t2_tra_np_min_max'
        path_diff_tra_ADC_BVAL_np_min_max = '.\Data\\diff_ADC_BVAL_np_min_max'
    elif platform.system() == 'Linux':
        path_t2_tra_np_min_max = './Data/t2_tra_np_min_max'
        path_diff_tra_ADC_BVAL_np_min_max = './Data/diff_ADC_BVAL_np_min_max'

    """"
    images = os.listdir(path_t2_tra_np_min_max)

    dataset = datasets.SiameseNetworkDataset(path_t2_tra_np_min_max, images)   # bude dobre tam pasovat cely zoznnam dopred, aby som
                                                              # mohol rozdelit trenovacie data a validacne
    network = networks.SiameseNet().cuda()

    loss = loss_functions.ContrastiveLoss()

    train_loop(70, network, loss, dataset)

    """


    images = os.listdir(path_diff_tra_ADC_BVAL_np_min_max)

    dataset = datasets.SiameseNetworkDataset(path_diff_tra_ADC_BVAL_np_min_max, images)

    dataset_loader = DataLoader(dataset, shuffle=1, num_workers=8, batch_size=16)

    network = networks.Channel2SiameseNet().cuda()

    loss = loss_functions.ContrastiveLoss()

    train_loop(150, network, loss, dataset, dataset_loader)

    torch.save(network.state_dict(), './model')
    """
    
    images = os.listdir(path_diff_tra_ADC_BVAL_np_min_max)

    dataset = datasets.SiameseNetworkDataset(path_diff_tra_ADC_BVAL_np_min_max, images)

    dataset_loader = DataLoader(dataset, shuffle=1, num_workers=8, batch_size=1)
    
    """

    out_list = []

    with torch.no_grad():
        network = networks.Channel2SiameseNet().cuda()
        network.load_state_dict(torch.load('./model'))
        network.eval()
        for img0, img1, pair_label, label0, label1 in dataset_loader:
            img0, img1 = img0.cuda(), img1.cuda()
            out1, out2 = network.forward(img0, img1)
            out_list.append([out1.cpu().numpy(), out2.cpu().numpy(), label0.cpu().numpy(), label1.cpu().numpy()])


    for item in out_list:
        x0, y0 = zip(*item[0])
        x1, y1 = zip(*item[1])
        label0 = item[2][0]
        label1 = item[3][0]

        if label0 == 0:
            point0 = 'bo'
        else:
            point0 = 'ro'

        if label1 == 0:
            point1 = 'bo'
        else:
            point1 = 'ro'

        plt.plot(x0, y0, point0)
        plt.plot(x1, y1, point1)

    plt.show()

if __name__ == "__main__":
    main()