import torch
import os
import platform
from torch.utils.data import DataLoader
from torch import optim

import networks
import datasets
import loss_functions

path_t2_tra_np_min_max = ''

def train_loop(network, loss, dataset):
    dataset_loader = DataLoader(dataset, shuffle=1, num_workers=8, batch_size=16)
    optimizer = optim.Adam(network.parameters(), lr=0.0001)

    counter = []
    loss_history = []
    iteration_number = 0

    for epoch in range(0, 500):
        for i, data in enumerate(dataset_loader, 0):
            image0, image1, pair_label = data
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


    images = os.listdir(path_t2_tra_np_min_max)

    dataset = datasets.SiameseNetworkDataset(path_t2_tra_np_min_max, images)   # bude dobre tam pasovat cely zoznnam dopred, aby som
                                                              # mohol rozdelit trenovacie data a validacne
    network = networks.SiameseNet().cuda()

    loss = loss_functions.ContrastiveLoss()
    
    train_loop(network, loss, dataset)

    """
    images = os.listdir(path_diff_tra_ADC_BVAL_np_min_max)

    dataset = datasets.SiameseNetworkDataset(path_diff_tra_ADC_BVAL_np_min_max, images)

    network = networks.Channel2SiameseNet().cuda()

    loss = loss_functions.ContrastiveLoss()

    train_loop(network, loss, dataset)
    """



if __name__ == "__main__":
    main()