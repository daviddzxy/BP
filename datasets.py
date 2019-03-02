import random
import torch
import numpy as np
import os
from datetime import datetime


from torch.utils.data import Dataset

class SiameseNetworkDataset(Dataset):
    def __init__(self, data_file, images, no_pairs):
        self.data_file = data_file
        self.negative_images = []
        self.positive_images = []
        self.pairs = []

        for image in images:
            if image.__contains__('False'):
                self.negative_images.append(image)
            else:
                self.positive_images.append(image)

        self._make_random_pairs(no_pairs)

    def __getitem__(self, index):
        img0, img1, pair_label = self.pairs.__getitem__(index)

        img_data0 = np.load(os.path.join(self.data_file, img0))
        img_data1 = np.load(os.path.join(self.data_file, img1))

        img0_label = self._get_label(img0)
        img1_label = self._get_label(img1)

        return \
            torch.from_numpy(img_data0).float(),\
            torch.from_numpy(img_data1).float(),\
            pair_label,\
            img0_label,\
            img1_label


    def __len__(self):
        return len(self.pairs)

    def _make_random_pairs(self, no_pairs):
        """
        Vytvori pary obrazkov vsetkych kombinacii, teda dokopy 4 * no_of_pairs
        """

        random.seed(datetime.now())
        for i in range(no_pairs):
            self.pairs.append([random.choice(self.negative_images), random.choice(self.negative_images), 0])
            self.pairs.append([random.choice(self.positive_images), random.choice(self.positive_images), 0])
            self.pairs.append([random.choice(self.negative_images), random.choice(self.positive_images), 1])
            self.pairs.append([random.choice(self.positive_images), random.choice(self.negative_images), 1])

        random.shuffle(self.pairs)

    def _get_label(self, name):
        if name.__contains__("False"):
            return 0
        else:
            return 1



