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
    if mode == '2D':
        for cnl in range(data.shape[1]):
            data[:, cnl, :, :] = (data[:, cnl, :, :] - data[:, cnl, :, :].min()) / (data[:, cnl, :, :].max() - data[:, cnl, :, :].min())
    elif mode == '3D':
        data = (data - data.min()) / (data.max() - data.min())

    return data


def scaling_z_score(data, mode='2D'):
    if mode == '2D':
        for cnl in range(data.shape[1]):
            data[:, cnl, :, :] = (data[:, cnl, :, :] - data[:, cnl, :, :].mean()) / data[:, cnl, :, :].std()
    elif mode == '3D':
        data = (data - data.mean()) / data.std()

    return data


def main():
    path_t2_tra_3D_np = './Data/t2_tra_np_3D'
    path_t2_tra_np = './Data/t2_tra_np'
    path_t2_tra_np_min_max = './Data/t2_tra_np_min_max'
    path_t2_tra_np_3D_min_max = './Data/t2_tra_np_3D_min_max'
    path_diff_tra_ADC_BVAL_np = './Data/diff_ADC_BVAL_np'
    path_diff_tra_ADC_BVAL_np_min_max = './Data/diff_ADC_BVAL_np_min_max'

    t2_data_path_list = load(path_t2_tra_np)
    data_list, path_list = list(map(list, zip(*t2_data_path_list)))
    data_list = scaling_z_score(np.array(data_list))
    for data, path in zip(data_list, path_list):
        np.save(os.path.join(path_t2_tra_np_min_max, path.rsplit(os.sep, 1)[1]), data)

    t2_3D_data_path_list = load(path_t2_tra_3D_np)
    data_list, path_list = list(map(list, zip(*t2_3D_data_path_list)))
    data_list = scaling_z_score(np.array(data_list), '3D')
    for data, path in zip(data_list, path_list):
        np.save(os.path.join(path_t2_tra_np_3D_min_max, path.rsplit(os.sep, 1)[1]), data)

    # diff_tra_path_list = load(path_diff_tra_ADC_BVAL_np)
    # data_list, path_list = list(map(list, zip(*diff_tra_path_list)))
    # data_list = np.array(data_list)
    #
    # data_list = scaling_z_score(data_list)
    # data_list = scaling_0_1(data_list)
    # for data, path in zip(data_list, path_list):
    #     np.save(os.path.join(path_diff_tra_ADC_BVAL_np_min_max, path.rsplit(os.sep, 1)[1]), data)


if __name__ == '__main__':
    main()
