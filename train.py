import numpy as np
import preprocess


def main():
    data_path_list = preprocess.load('D:\BP\Classified\\Data')
    data, paths = list(map(list, zip(*data_path_list)))

    print(sum('POSITIVE' in path for path in paths))
    print(sum('NEGATIVE' in path for path in paths))

#    data = np.array([item for item in data])
#   print(np.array(([data_array[0] for data in data_path_list])).shape)


if __name__ ==  "__main__":
    main()