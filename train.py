import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import platform
from torch.utils.data import DataLoader,Dataset
from torch import optim

path_t2_tra_np_min_max = ''

class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        self.fc1 = nn.Sequential(
            #nn.Linear(8 * 32 * 32, 500),
            nn.Linear(5408, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
       # output = output.view(output.size()[0], -1)
        output = output.view(-1, 8 * 26 * 26)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class SiameseNetworkDataset(Dataset):
    def __init__(self, data_file, images):
        self.data_file = data_file
        self.images = images



    def __getitem__(self, index=None):
        image0 = random.choice(self.images)
        label0 = self._get_label(image0)

        determine_class = random.randint(0, 1)
        label1 = -1
        pair_label = -1

        while determine_class != label1:
            image1 = random.choice(self.images)
            label1 = self._get_label(image1)

        if label0 == label1:
            pair_label = 0
        else:
            pair_label = 1

        image0_data = np.load(os.path.join(self.data_file, image0))
        image1_data = np.load(os.path.join(self.data_file, image1))

        return torch.from_numpy(image0_data).float(),\
               torch.from_numpy(image1_data).float(),\
               label0, label1, pair_label



    def _get_label(self, image):
        if image.__contains__('False'):
            label = 0
        else:
            label = 1

        return label

    def __len__(self):
       return len(self.images)

def main():
    if platform.system() == 'Windows':
        path_t2_tra_np_min_max = '.\Data\\t2_tra_np_min_max'
    elif platform.system() == 'Linux':
        path_t2_tra_np_min_max = './Data/t2_tra_np_min_max'




    images = os.listdir(path_t2_tra_np_min_max)
    dataset = SiameseNetworkDataset(path_t2_tra_np_min_max, images)

    images = os.listdir(path_t2_tra_np_min_max)

    dataset = SiameseNetworkDataset(path_t2_tra_np_min_max, images)   # bude dobre tam pasovat cely zoznnam dopred, aby som
                                                              # mohol rozdelit trenovacie data a validacne

    dataset_loader = DataLoader(dataset, num_workers=8, batch_size=8)

    network = SiameseNet().cuda()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(network.parameters(), lr=0.0005)

    counter = []
    loss_history = []
    iteration_number = 0

    for epoch in range(0, 500):
        for i, data in enumerate(dataset_loader, 0):
            image0, image1, label0, label1, pair_label = data
            pair_label = pair_label.type(torch.FloatTensor)
            image0, image1, pair_label = image0.cuda(), image1.cuda(), pair_label.cuda()
            out1, out2 = network(image0, image1)
            loss_contrastive = criterion(out1, out2, pair_label)
            loss_contrastive.backward()
            optimizer.step()
            if i % 5 == 0:
                print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
    

if __name__ == "__main__":
    main()