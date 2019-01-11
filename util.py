import pandas as pd
import numpy as np
import os
import posixpath
import pydicom
import pylab

path_t2_tra_pic = '.\Data\\t2_tra_pic'
path_t2_tra_np = '.\Data\\t2_tra_np'
path_diff_tra_ADC_BVAL_pic = '.\Data\\diff_ADC_BVAL_pic'
path_diff_tra_ADC_BVAL_np = '.\Data\\diff_ADC_BVAL_np'


def _find_slices(prox_id, dcm_num, modality):
    """
    funkcia vyhladava DICOM subory na disku podla ID pacienta, DCM_NUM a modality(t2_tra, diff_ADC, diff_BVAL)
    vracia cesty k tymto suborom
    :param prox_id:
    :param dcm_num:
    :param modality:
    :return:
    """
    dicom_files = []
    root = posixpath.join('D:', 'BP', 'Dataset_PROSTATEX', 'PROSTATEx DICOM', str(prox_id))
    study_dir = os.listdir(root)  # 'D:\BP\Dataset_PROSTATEX\PROSTATEx DICOM\ProstateX-0000\'
    if len(study_dir) > 1:
        print("Viac studii Error")
        print("ProxID: " + str(prox_id))
        print("DCMNum: " + str(dcm_num))
        return None
    subdir = posixpath.join(root, study_dir[0])
    for dicom_dirs in os.listdir(subdir):  # D:\BP\Dataset\PROSTATEx DICOM\ProstateX-0000\07-07-2011-MR prostaat kanker detectie WDSmc MCAPRODETW-05711
        image_folder = posixpath.join(subdir, dicom_dirs)
        if (modality == 't2_tra' and image_folder.__contains__(str(dcm_num) + "-t2tsetra")) or\
           (modality == 'diff_ADC' and (image_folder.__contains__(str(dcm_num) + "-ep2ddifftraDYNDISTADC") or image_folder.__contains__(str(dcm_num) + "-ep2ddifftraDYNDISTMIXADC"))) or\
           (modality == 'diff_BVAL' and (image_folder.__contains__(str(dcm_num) + "-ep2ddifftraDYNDISTCALCBVAL") or image_folder.__contains__(str(dcm_num) + "-ep2ddifftraDYNDISTMIXCALCBVAL"))):
            for image in os.listdir(image_folder):
                dicom_files.append(posixpath.join(image_folder, image))
            return dicom_files


def _save_as_tiff_image(dicom_files, coordinates, patch_size, save_path, image_name):
    """
    funkcia ulozi vyrez z DICOM snimku ako .tiff obrazok o rozmere patch_size x patch_size

    :param dicom_files: list ciest k DICOM rezom
    :param coordinates: suradnice urcujuce stred vyrezu
    :param patch_size: velkost vyrezu
    :param save_path: cesta kam sa vyrez ulozi
    :param image_name: nazov suboru
    :return:
    """
    slices = []
    for dicom_file in dicom_files:
        slice = pydicom.dcmread(dicom_file)
        slices.append(slice)

    slices.sort(key=lambda x: x.SliceLocation, reverse=False)

    try:
       pylab.imsave(save_path + '/' + image_name + '.tiff',
                    slices[coordinates[2]].pixel_array[coordinates[1] - (patch_size // 2):coordinates[1] + (patch_size // 2), coordinates[0] - (patch_size // 2):coordinates[0] + (patch_size // 2)], cmap=pylab.cm.gist_gray)
    except IndexError:
        print("Index Error")
        print(image_name)
        print(save_path)


def _save_as_numpy_array(dicom_files_list, coordinates, patch_size, save_path, name):
    """
    funkcia ulozi v vyrez z DICOM snimku ako 3D numpy pole, ak ma slices_list viac ako jednu polozku,
    tak funkcia tieto polia nasklada na seba, vysledne pole ma rozmer (patch_size, patch_size, len(dicom_file_list))

    :param slices_list: list listov jednotlivych rezov
    :param coordinates: suradnice urcujuce stred vyrezu, coordinates = [i,j,k]
    :param patch_size: velkost vyrezu
    :param save_path: cesta kam sa vyrez ulozi
    :param name: nazov suboru
    :return:
    """
    array_list = []
    for dicom_files in dicom_files_list:
        dicom_slices = []
        for dicom_file in dicom_files:
            dicom_slices.append(pydicom.dcmread(dicom_file)) #nacita dicom subory

        dicom_slices.sort(key=lambda x: x.SliceLocation, reverse=False)

        #vyrez patch podla danych suradnic
        try:
            array = dicom_slices[coordinates[2]].pixel_array[coordinates[1] - (patch_size // 2):coordinates[1] + (patch_size // 2), coordinates[0] - (patch_size // 2):coordinates[0] + (patch_size // 2)]
        except:
            print("Index Error")
            print(name)
            print(save_path)
            return


        array_list.append(array)
    #transofmuj 2D pole / polia na 3D
    array_list = np.stack(array_list, axis=2)
    np.save(save_path + '/' + name, array_list)



def process_t2_tra(dataframe):
    dataframe = dataframe[dataframe['DCMSerDescr'] == 't2_tse_tra']
    for index, row in dataframe.iterrows():
        slices_list = [_find_slices(row.ProxID, row.DCMSerNum, 't2_tra')]
        if None not in slices_list:
            coordinates = list(map(int, row.ijk.split()))
            name = str(row.ClinSig) + ' ' + str(row.ProxID) + " IJK " + str(row.ijk) + " DCM " + str(row.DCMSerNum)
            _save_as_numpy_array(slices_list, coordinates, 32, path_t2_tra_np, name)
            #_save_as_tiff_image(slices_list[0], coordinates, 50, path_t2_tra_pic, name)


def process_diff_tra(dataframe):
    combined_ADC = dataframe[(dataframe['DCMSerDescr'] == 'ep2d_diff_tra_DYNDIST_ADC') | (dataframe['DCMSerDescr'] == 'ep2d_diff_tra_DYNDIST_MIX_ADC')]
    combined_BVAL = dataframe[(dataframe['DCMSerDescr'] == 'ep2d_diff_tra_DYNDISTCALC_BVAL') | (dataframe['DCMSerDescr'] == 'ep2d_diff_tra_DYNDIST_MIXCALC_BVAL')]
    combined_diff = pd.merge(combined_ADC, combined_BVAL, how='left', left_on=['ProxID', 'fid', 'pos', 'ijk', 'Dim', 'zone', 'ClinSig'], right_on=['ProxID', 'fid', 'pos', 'ijk', 'Dim', 'zone', 'ClinSig'])
    for index, row in combined_diff.iterrows():
        slices_list = [_find_slices(row.ProxID, row.DCMSerNum_x, 'diff_ADC'), _find_slices(row.ProxID, row.DCMSerNum_y, 'diff_BVAL')]
        if None not in slices_list:
            coordinates = list(map(int, row.ijk.split()))
            name = str(row.ClinSig) + ' ' + str(row.ProxID) + " IJK " + str(row.ijk)
            _save_as_numpy_array(slices_list, coordinates, 32, path_diff_tra_ADC_BVAL_np, name)
            #_save_as_tiff_image(slices_list[0], coordinates, 50, path_diff_tra_ADC_BVAL_pic, name)

def main():
    findings = pd.read_csv("D:/BP/Dataset_PROSTATEX/PROSTATEx lesion information/ProstateX-Findings-Train.csv")
    images = pd.read_csv("D:/BP/Dataset_PROSTATEX/PROSTATEx lesion information/ProstateX-Images-Train.csv")
    findings = findings[['ProxID', 'pos', 'fid', 'ClinSig', 'zone']]
    images = images[['ProxID', 'pos', 'Name', 'fid', 'ijk', 'Dim', 'DCMSerDescr', 'DCMSerNum']]
    combined_df = pd.merge(images, findings, how='left', left_on=['ProxID', 'fid', 'pos'], right_on=['ProxID', 'fid', 'pos'])
    process_t2_tra(combined_df)
    process_diff_tra(combined_df)


if __name__ == '__main__':
    main()
