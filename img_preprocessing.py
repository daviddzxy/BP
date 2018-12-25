import numpy as np
import os

'''
funkcia nacita numpy polia z cesty, vrati tuple list,
na prvej pozicii je numpy pole, na druhej je cesta k nemu
'''
def load(path):
    result_list = []
    data_paths = os.listdir(path)
    data_paths = [os.path.join(path, data_path) for data_path in data_paths]
    for path in data_paths:
        numpy_array = np.load(path)
        result_list.append((numpy_array, path))

    return result_list


'''
funkcia ulozi numpy pole
'''
def save(numpy_array, path):
    np.save(path, numpy_array)

'''
funkcia skonvertuje 2D numpy pole na 3D numpy pole
'''
def convert_to_3D(numpy_array):
    numpy_array = np.reshape(numpy_array, numpy_array.shape + (1,))
    return numpy_array


def main():
    data_path_list = load('D:\BP\Classified\\Data')
    for item in data_path_list:
        new_array = convert_to_3D(item[0])
        save(new_array, item[1])

if __name__ == '__main__':
    main()