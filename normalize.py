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


def scaling_0_1(data, mode='2D'):
    """
    Function that scales numpy arrays into 0, 1 interval
    :param data: numpy array of shape N x C x W x H or N x C x D x W x H
    N - Number of stacked images
    C - Number of channels
    W - Width of the image
    H - Height of the image
    D - Depth of the image
    :param mode: 2D or 3D mode
    :return: returns np.array
    """
    if mode == '2D':
        for cnl in range(data.shape[1]):
            data[:, cnl, :, :] = (data[:, cnl, :, :] - data[:, cnl, :, :].min()) / (data[:, cnl, :, :].max() - data[:, cnl, :, :].min())
    elif mode == '3D':
        for cnl in range(data.shape[1]):
            data[:, cnl, :, :, :] = (data[:, cnl, :, :, :] - data[:, cnl, :, :, :].min()) / (data[:, cnl, :, :, :].max() - data[:, cnl, :, :, :].min())

    return data


def scaling_z_score(data, mode='2D'):
    """
    Function that uses z score scaling
    :param data: numpy array of shape N x C x W x H or N x C x D x W x H
    N - Number of stacked images
    C - Number of channels
    W - Width of the image
    H - Height of the image
    D - Depth of the image
    :param mode: 2D or 3D mode
    :return: returns np.array
    """
    if mode == '2D':
        for cnl in range(data.shape[1]):
            data[:, cnl, :, :] = (data[:, cnl, :, :] - data[:, cnl, :, :].mean()) / data[:, cnl, :, :].std()
    elif mode == '3D':
        for cnl in range(data.shape[1]):
            data[:, cnl, :, :, :] = (data[:, cnl, :, :, :] - data[:, cnl, :, :, :].mean()) / data[:, cnl, :, :, :].std()
    return data


def normalize(data_path, save_path, type='z_score', mode = '2D'):
    normalization = scaling_z_score if type == 'z_score' else scaling_0_1
    data_path_list = load(data_path)
    data_list, path_list = list(map(list, zip(*data_path_list)))
    data_list = normalization(np.array(data_list), mode=mode)
    for data, path in zip(data_list, path_list):
        np.save(os.path.join(save_path, path.rsplit(os.sep, 1)[1]), data)


def main():
    path_t2_tra_np = '../Data/t2_tra_np'
    path_t2_tra_np_norm = '../Data/t2_tra_np_norm'
    path_t2_tra_3D_np = '../Data/t2_tra_np_3D'
    path_t2_tra_np_3D_norm = '../Data/t2_tra_np_3D_norm'
    path_diff_tra_ADC_BVAL_np = '../Data/diff_ADC_BVAL_np'
    path_diff_tra_ADC_BVAL_np_norm = '../Data/diff_ADC_BVAL_np_norm'
    path_diff_tra_ADC_BVAL_np_3D = '../Data/diff_ADC_BVAL_np_3D'
    path_diff_tra_ADC_BVAL_np_3D_norm = '../Data/diff_ADC_BVAL_np_3D_norm'

    normalize(path_t2_tra_np, path_t2_tra_np_norm)
    normalize(path_t2_tra_3D_np, path_t2_tra_np_3D_norm, mode='3D')
    normalize(path_diff_tra_ADC_BVAL_np, path_diff_tra_ADC_BVAL_np_norm)
    normalize(path_diff_tra_ADC_BVAL_np_3D, path_diff_tra_ADC_BVAL_np_3D_norm, mode='3D')


if __name__ == '__main__':
    main()
