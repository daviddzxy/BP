import numpy as np
import os


def load(path):
    result_list = []
    data_paths = os.listdir(path)
    data_paths = [os.path.join(path, data_path) for data_path in data_paths]
    for path in data_paths:
        numpy_array = np.load(path)
        result_list.append([numpy_array, path])
    return result_list


def horizontal_flip(img):
    return np.flip(img, -2)


def vertical_flip(img):
    return np.flip(img, -1)


def augment(data_path, save_path):
    os.listdir(data_path)
    data_path_list = load(data_path)
    data_list, path_list = list(map(list, zip(*data_path_list)))
    for data, path in zip(data_list, path_list):
        np.save(os.path.join(save_path, (path.rsplit(os.sep, 1)[1].split(".")[0]) + " H"), horizontal_flip(data))
        np.save(os.path.join(save_path, (path.rsplit(os.sep, 1)[1]. split(".")[0]) + " V"), vertical_flip(data))


if __name__ == '__main__':
    path_t2_tra_np = './Data/t2_tra_np'
    path_t2_tra_3D_np = './Data/t2_tra_np_3D'
    path_diff_tra_ADC_BVAL_np = './Data/diff_ADC_BVAL_np'
    path_diff_tra_ADC_BVAL_3D_np = './Data/diff_ADC_BVAL_3D_np'

    augment(path_t2_tra_np, path_t2_tra_np)
    augment(path_t2_tra_3D_np, path_t2_tra_3D_np)
    augment(path_diff_tra_ADC_BVAL_np, path_diff_tra_ADC_BVAL_np)
    augment(path_diff_tra_ADC_BVAL_3D_np, path_diff_tra_ADC_BVAL_3D_np)
