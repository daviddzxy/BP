import numpy as np
import os

path_t2_tra_np = '.\Data\\t2_tra_np'
path_t2_tra_np_min_max = '.\Data\\t2_tra_np_min_max'


def load(path):
    result_list = []
    data_paths = os.listdir(path)
    data_paths = [os.path.join(path, data_path) for data_path in data_paths]
    for path in data_paths:
        numpy_array = np.load(path)
        result_list.append((numpy_array, path))
    
    return result_list


def save(data_array, path_array):
    for data, path in zip(data_array, path_array):
        np.save(path, data)


def min_max_normalization(data):
    max_value = np.max(data)
    min_value = np.min(data)
    for index in range(len(data)):
        data[index] = (data[index] - min_value) / (max_value - min_value)
    return data


def main():
    t2_data_path_list = load(path_t2_tra_np)
    data_list, path_list = list(map(list, zip(*t2_data_path_list)))
    data_list = min_max_normalization(data_list)
    save(data_list, path_t2_tra_np_min_max)


if __name__ == '__main__':
    main()
