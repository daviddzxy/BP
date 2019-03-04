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


def save(data, path, name):
    np.save(os.path.join(path, name), data)


def scaling_0_1(img, mode='2D'):
    assert img.ndim == 3
    if mode == '2D':
        for i in range(img.shape[0]):
            img[i] = (img[i] - np.min(img[i])) / (np.max(img[i]) - np.min(img[i]))
    elif mode == '3D':
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img

"""
def main():
    path_t2_tra_3D_np = './Data/t2_tra_np_3D'
    path_t2_tra_np = './Data/t2_tra_np'
    path_t2_tra_np_min_max = './Data/t2_tra_np_min_max'
    path_diff_tra_ADC_BVAL_np = './Data/diff_ADC_BVAL_np'
    path_diff_tra_ADC_BVAL_np_min_max = './Data/diff_ADC_BVAL_np_min_max'

    t2_data_path_list = load(path_t2_tra_np)
    data_list, path_list = list(map(list, zip(*t2_data_path_list)))
    data_list = min_max_normalization(data_list)
    for data, path in zip(data_list, path_list):
        save(data, path_t2_tra_np_min_max, path.rsplit(os.sep, 1)[1])

    diff_tra_path_list = load(path_diff_tra_ADC_BVAL_np)
    data_list, path_list = list(map(list, zip(*diff_tra_path_list)))
    data_list = min_max_normalization(data_list)
    for data, path in zip(data_list, path_list):
        save(data, path_diff_tra_ADC_BVAL_np_min_max, path.rsplit(os.sep, 1)[1])

#    t2_data_path_list_3D = load(path_t2_tra_3D_np)
 #   data_list, path_list = list(map(list, zip(*t2_data_path_list_3D)))
  #  data_list = min_max_normalization(data_list)
   # for data, path in zip(data_list, path_list):
    #save(data, path_t2_tra_np_min_max, path.rsplit(os.sep, 1)[1])

"""
def main():
    path_t2_tra_3D_np = './Data/t2_tra_np_3D'
    path_t2_tra_np = './Data/t2_tra_np'
    path_t2_tra_np_min_max = './Data/t2_tra_np_min_max'
    path_diff_tra_ADC_BVAL_np = './Data/diff_ADC_BVAL_np'
    path_diff_tra_ADC_BVAL_np_min_max = './Data/diff_ADC_BVAL_np_min_max'

    t2_data_path_list = load(path_t2_tra_np)
    data_list, path_list = list(map(list, zip(*t2_data_path_list)))
    for img in data_list:
        scaling_0_1(img)
    for data, path in zip(data_list, path_list):
        save(data, path_t2_tra_np_min_max, path.rsplit(os.sep, 1)[1])

    diff_tra_path_list = load(path_diff_tra_ADC_BVAL_np)
    data_list, path_list = list(map(list, zip(*diff_tra_path_list)))
    for img in data_list:
        scaling_0_1(img)
    for data, path in zip(data_list, path_list):
        save(data, path_diff_tra_ADC_BVAL_np_min_max, path.rsplit(os.sep, 1)[1])



if __name__ == '__main__':
    main()
