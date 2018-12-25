import pandas as pd
import numpy as np
import os
import posixpath
import pydicom
import pylab

'''
prox_id - id pacienta
dcm_num - cislo DICOM serie
'''
def load_slices(prox_id, dcm_num):
    dicom_files = []
    root = posixpath.join('D:', 'BP', 'Dataset_PROSTATEX','PROSTATEx DICOM', str(prox_id))
    study_dir = os.listdir(root) # 'D:\BP\Dataset_PROSTATEX\PROSTATEx DICOM\ProstateX-0000\'
    if len(study_dir) > 1:
        print("Viac studii Error")
        print("ProxID: " + str(prox_id))
        print("DCMNum: " + str(dcm_num))
        return None
    subdir = posixpath.join(root, study_dir[0])
    for dicom_dirs in os.listdir(subdir):  # D:\BP\Dataset\PROSTATEx DICOM\ProstateX-0000\07-07-2011-MR prostaat kanker detectie WDSmc MCAPRODETW-05711
        image_folder = posixpath.join(subdir, dicom_dirs)
        if image_folder.__contains__(str(dcm_num) + "-t2tsetra"):
            for image in os.listdir(image_folder):
                dicom_files.append(posixpath.join(image_folder, image))
            return dicom_files

'''
tato funkcia sa pri trenovani modelu pouzivat nebude, sluzi len na vizualizaciu vyrezanych casti
dicom_files - list ciest k slicom
coordinates - list koordinatov, urcuju stred vyrezu, coordinates[0]=i coordinates[1] = j coordiantes[2] = k 
patch_size - velkost stvorca vyrezu
save_path - kam sa vysledny obrazok ulozit
name - nazov obrazku
'''
def save_as_tiff_image(dicom_files, coordinates, patch_size, save_path, image_name):
    slices = []
    for dicom_file in dicom_files:
        slice = pydicom.dcmread(dicom_file)
        slices.append(slice)

    slices.sort(key=lambda x: x.SliceLocation, reverse=True)

    try:
       pylab.imsave(save_path + '/' + image_name + '.tiff',
                    slices[coordinates[2]].pixel_array[coordinates[1] - (patch_size // 2) : coordinates[1] + (patch_size // 2),coordinates[0] - (patch_size // 2) : coordinates[0] + (patch_size // 2)], cmap = pylab.cm.gist_gray)
    except IndexError:
        print("Index Error")
        print(image_name)
        print(save_path)

'''
dicom_files - list ciest k slicom
coordinates - list koordinatov, urcuju stred vyrezu, coordinates[0]=i coordinates[1] = j coordiantes[2] = k 
patch_size - velkost stvorca vyrezu
save_path - kam sa vysledny obrazok ulozit
name - nazov obrazku
'''
def save_as_numpy_array(dicom_files, coordinates, patch_size, save_path, name):
    slices = []
    for dicom_file in dicom_files:
        slice = pydicom.dcmread(dicom_file)
        slices.append(slice)

    slices.sort(key = lambda x: x.SliceLocation, reverse = True)

    try:
       np.save(save_path + '/' + name,
        slices[coordinates[2]].pixel_array[coordinates[1] - (patch_size // 2) : coordinates[1] + (patch_size // 2),coordinates[0] - (patch_size // 2) : coordinates[0] + (patch_size // 2)])
    except:
        print("Index Error")
        print(name)
        print(save_path)


def main():
    findings = pd.read_csv("D:/BP/Dataset_PROSTATEX/PROSTATEx lesion information/ProstateX-Findings-Train.csv")
    images = pd.read_csv("D:/BP/Dataset_PROSTATEX/PROSTATEx lesion information/ProstateX-Images-Train.csv")
    find = findings[['ProxID', 'pos', 'fid', 'ClinSig', 'zone']]
    images = images[['ProxID', 'pos', 'Name', 'fid', 'ijk', 'Dim', 'DCMSerNum']]
    images = images[images.Name == 't2_tse_tra0']
    combined_df = pd.merge(images, find, how='left', left_on=['ProxID', 'fid', 'pos'], right_on=['ProxID', 'fid', 'pos'])
    combined_df = combined_df.drop_duplicates()
    path_clin_sig_truePIC = 'D:\BP\Classified\\PositivePIC'
    path_clin_sig_falsePIC = 'D:\BP\Classified\\NegativePIC'
    path_clin_sig_trueNP = 'D:\BP\Classified\\Data'
    path_clin_sig_falseNP = 'D:\BP\Classified\\Data'
    combined_df_true = combined_df[(combined_df['ClinSig'] == True)]
    combined_df_false = combined_df[(combined_df['ClinSig'] == False)]
    for index, row in combined_df_true.iterrows():
        slices = load_slices(row.ProxID, row.DCMSerNum)
        if slices is None:
            continue
        coordinates = list(map(int, row.ijk.split()))
        name = 'POSITIVE ' + str(row.ProxID) + " IJK " + str(row.ijk) + " DCM " + str(row.DCMSerNum)
        #save_as_tiff_image(slices, coordinates, 32, path_clin_sig_truePIC, name)
        save_as_numpy_array(slices, coordinates, 32, path_clin_sig_trueNP, name)

    for index, row in combined_df_false.iterrows():
        slices = load_slices(row.ProxID, row.DCMSerNum)
        if slices is None:
            continue
        coordinates = list(map(int, row.ijk.split()))
        name = 'NEGATIVE ' + str(row.ProxID) + " IJK " + str(row.ijk) + " DCM " + str(row.DCMSerNum)
        #save_as_tiff_image(slices, coordinates, 32, path_clin_sig_falsePIC, name)
        save_as_numpy_array(slices, coordinates, 32, path_clin_sig_falseNP, name)


if __name__ == '__main__':
    main()
