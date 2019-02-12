import numpy as np
import os
import platform

path_t2_tra_np = ''
path_t2_tra_np_min_max = ''


def load(path):
    result_list = []
    data_paths = os.listdir(path)
    data_paths = [os.path.join(path, data_path) for data_path in data_paths]
    for path in data_paths:
        numpy_array = np.load(path)
        result_list.append((numpy_array, path))
    
    return result_list


def save(data, path, name):
    np.save(os.path.join(path, name), data)

def min_max_normalization(data):
    max_value = np.max(data)
    min_value = np.min(data)
    for index in range(len(data)):
        data[index] = (data[index] - min_value) / (max_value - min_value)
    return data


def main():
    if platform.system() == 'Windows':
        path_t2_tra_np = '.\Data\\t2_tra_np'
        path_t2_tra_np_min_max = '.\Data\\t2_tra_np_min_max'
    elif platform.system() == 'Linux':
        path_t2_tra_np = './Data/t2_tra_np'
        path_t2_tra_np_min_max = './Data/t2_tra_np_min_max'


    t2_data_path_list = load(path_t2_tra_np)
    data_list, path_list = list(map(list, zip(*t2_data_path_list)))
    data_list = min_max_normalization(data_list)
    for data, path in zip(data_list, path_list):
        save(data, path_t2_tra_np_min_max, path.rsplit(os.sep, 1)[1])


if __name__ == '__main__':
    main()
