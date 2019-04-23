import torch
import numpy as np
import os
from torch.utils.data import Dataset


class SiameseDataset(Dataset):
    def __init__(self, file_path, images, get_pair=True):
        self.file_path = file_path
        self.pairs = []
        self.images = images
        self.strategy = None
        self._make_pairs(images)
        self.change_strategy(get_pair=get_pair)

    def change_strategy(self, get_pair=True):
        self.strategy = self._get_pair if get_pair else self._get_single
        return self

    def __getitem__(self, index):
        return self.strategy(index)

    def __len__(self):
        return len(self.pairs) if self.strategy == self._get_pair else len(self.images)

    def _get_label(self, name):
        return 0 if name.__contains__("False") else 1

    def _make_pairs(self, images):
        for i in range(len(images)):
            j = i
            while j < len(images):
                self.pairs.append([images.__getitem__(i),  images.__getitem__(j)])
                j = j + 1

    def _get_pair(self, index):
        path0, path1 = self.pairs.__getitem__(index)
        data0 = np.load(os.path.join(self.file_path, path0))
        data1 = np.load(os.path.join(self.file_path, path1))

        label0 = self._get_label(path0)
        label1 = self._get_label(path1)

        pair_label = 0 if label0 == label1 else 1

        return \
            torch.from_numpy(data0).float(),\
            torch.from_numpy(data1).float(),\
            pair_label,\
            label0,\
            label1

    def _get_single(self, index):
        path = self.images.__getitem__(index)
        data = np.load(os.path.join(self.file_path, path))
        label = self._get_label(path)

        return \
            torch.from_numpy(data).float(), \
            path, \
            label



