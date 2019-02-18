import random
import torch
import numpy as np
import os

from torch.utils.data import Dataset

class SiameseNetworkDataset(Dataset):
    def __init__(self, data_file, images):
        self.data_file = data_file
        self.negative_images = []
        self.positive_images = []
        self.pairs = []

        for image in images:
            if image.__contains__('False'):
                self.negative_images.append(image)
            else:
                self.positive_images.append(image)

        self._make_random_pairs(200)

    def __getitem__(self, index):
        img0, img1, label = self.pairs.__getitem__(index)

        img_data0 = np.load(os.path.join(self.data_file, img0))
        img_data1 = np.load(os.path.join(self.data_file, img1))

        return torch.from_numpy(img_data0).float(),\
               torch.from_numpy(img_data1).float(),\
               label

    def __len__(self):
        return len(self.pairs)

    def _make_random_pairs(self, no_of_pairs):
        """
        Vytvori pary obrazkov vsetkych kombinacii, teda dokopy 4 * no_of_pairs
        """
        for i in range(no_of_pairs):
            self.pairs.append([random.choice(self.negative_images), random.choice(self.negative_images), 0])
            self.pairs.append([random.choice(self.positive_images), random.choice(self.positive_images), 0])
            self.pairs.append([random.choice(self.negative_images), random.choice(self.positive_images), 1])
            self.pairs.append([random.choice(self.positive_images), random.choice(self.negative_images), 1])

        random.shuffle(self.pairs)