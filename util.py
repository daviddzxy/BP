import pandas as pd
import numpy as np
import os
import pydicom
import pylab

def _find_slices(prox_id, dcm_num, modality):
    dicom_files = []
    root = os.path.join('..', 'Dataset_PROSTATEX', 'PROSTATEx DICOM', str(prox_id))
    study_dir = os.listdir(root)  # 'D:\BP\Dataset_PROSTATEX\PROSTATEx DICOM\ProstateX-0000\'
    if len(study_dir) > 1:
        print("More studies: ProxID {}, DCMNum {}\n".format(prox_id, dcm_num))
        return None

    subdir = os.path.join(root, study_dir[0])
    for dicom_dirs in os.listdir(subdir):  # D:\BP\Dataset\PROSTATEx DICOM\ProstateX-0000\07-07-2011-MR prostaat kanker detectie WDSmc MCAPRODETW-05711
        image_folder = os.path.join(subdir, dicom_dirs)
        if (modality == 't2_tra' and image_folder.__contains__(str(dcm_num) + "-t2tsetra")) or\
           (modality == 'diff_ADC' and (image_folder.__contains__(str(dcm_num) + "-ep2ddifftraDYNDISTADC") or image_folder.__contains__(str(dcm_num) + "-ep2ddifftraDYNDISTMIXADC"))) or\
           (modality == 'diff_BVAL' and (image_folder.__contains__(str(dcm_num) + "-ep2ddifftraDYNDISTCALCBVAL") or image_folder.__contains__(str(dcm_num) + "-ep2ddifftraDYNDISTMIXCALCBVAL"))):
            for image in os.listdir(image_folder):
                dicom_files.append(os.path.join(image_folder, image))
            return dicom_files


def _get_volume(dicom_files):
    dicom_slices = []
    for dicom_file in dicom_files:
        dicom_slices.append(pydicom.dcmread(dicom_file))

    dicom_slices.sort(key=lambda x: x.SliceLocation, reverse=False)

    pixel_array = []
    for dicom_slice in dicom_slices:
        pixel_array.append(dicom_slice.pixel_array)

    return np.array(pixel_array, dtype=np.float32)

def _extract_region(volume, coordinates, patch_size = None):
    assert volume.ndim == 3
    if patch_size is not None:
        patch = volume[coordinates[2], coordinates[1] - (patch_size // 2):coordinates[1] + (patch_size // 2),
                coordinates[0] - (patch_size // 2):coordinates[0] + (patch_size // 2)]
        assert patch.shape[0] == patch_size and patch.shape[1] == patch_size
    else:
        patch = volume[coordinates[2], :, :]

    return patch

def main():
    path_t2_tra_pic = './Data/t2_tra_pic'
    path_t2_tra_np = './Data/t2_tra_np'
    path_t2_tra_3D_np = './Data/t2_tra_np_3D'
    path_diff_tra_ADC_BVAL_pic = './Data/diff_ADC_BVAL_pic'
    path_diff_tra_ADC_BVAL_np = './Data/diff_ADC_BVAL_np'

    findings = pd.read_csv("D:/BP/Dataset_PROSTATEX/PROSTATEx lesion information/ProstateX-Findings-Train.csv")
    images = pd.read_csv("D:/BP/Dataset_PROSTATEX/PROSTATEx lesion information/ProstateX-Images-Train.csv")
    findings = findings[['ProxID', 'pos', 'fid', 'ClinSig', 'zone']]
    images = images[['ProxID', 'pos', 'Name', 'fid', 'ijk', 'Dim', 'DCMSerDescr', 'DCMSerNum']]
    combined_df = pd.merge(images, findings, how='left', left_on=['ProxID', 'fid', 'pos'], right_on=['ProxID', 'fid', 'pos'])
    combined_df = combined_df[(combined_df.ProxID != 'ProstateX-0052') & (combined_df.ProxID != 'ProstateX-0025') & (combined_df.ProxID != 'ProstateX-0148')]
    #combined_df = combined_df[(combined_df.DCMSerDescr == 'ep2d_diff_tra_DYNDIST_ADC') | (combined_df.DCMSerDescr == 'ep2d_diff_tra_DYNDISTCALC_BVAL')]
    #combined_df = combined_df[(combined_df.DCMSerDescr == 't2_tse_tra')]
    #combined_df = combined_df[combined_df.zone == 'PZ']
    del images
    del findings

    combined_t2 = combined_df[combined_df['DCMSerDescr'] == 't2_tse_tra']
    for index, row in combined_t2.iterrows():
        slices = _find_slices(row.ProxID, row.DCMSerNum, 't2_tra')
        if slices is not None:
            name = str(row.ClinSig) + ' ' + str(row.ProxID) + " IJK " + str(row.ijk) + " DCM " + str(row.DCMSerNum)
            coordinates = list(map(int, row.ijk.split()))
            #3D
            volume = _get_volume(slices)
            #np.save(os.path.join(path_t2_tra_3D_np, name), volume)
            #2D
            patch = _extract_region(volume, coordinates, 50)
            #make one channel image with shape of [channels, y, x]
            pylab.imsave(os.path.join(path_t2_tra_pic, name) + '.tiff', _extract_region(volume, coordinates), cmap=pylab.cm.gist_gray)
            patch = np.transpose(patch[:, :, np.newaxis], (2, 1, 0))
            np.save(os.path.join(path_t2_tra_np, name), patch)

    combined_ADC = combined_df[(combined_df['DCMSerDescr'] == 'ep2d_diff_tra_DYNDIST_ADC') | (combined_df['DCMSerDescr'] == 'ep2d_diff_tra_DYNDIST_MIX_ADC')]
    combined_BVAL = combined_df[(combined_df['DCMSerDescr'] == 'ep2d_diff_tra_DYNDISTCALC_BVAL') | (combined_df['DCMSerDescr'] == 'ep2d_diff_tra_DYNDIST_MIXCALC_BVAL')]
    combined_diff = pd.merge(combined_ADC, combined_BVAL, how='left', left_on=['ProxID', 'fid', 'pos', 'ijk', 'Dim', 'zone', 'ClinSig'], right_on=['ProxID', 'fid', 'pos', 'ijk', 'Dim', 'zone', 'ClinSig'])
    combined_diff = combined_diff[combined_diff.ProxID != 'ProstateX-0154']
    for index, row in combined_diff.iterrows():
        slices_ADC = _find_slices(row.ProxID, row.DCMSerNum_x, 'diff_ADC')
        slices_BVAL = _find_slices(row.ProxID, row.DCMSerNum_y, 'diff_BVAL')
        if slices_ADC is not None and slices_BVAL is not None:
            coordinates = list(map(int, row.ijk.split()))
            name = str(row.ClinSig) + ' ' + str(row.ProxID) + " IJK " + str(row.ijk)
            volume_ADC = _get_volume(slices_ADC)
            volume_BVAL = _get_volume(slices_BVAL)
            patch_ADC = _extract_region(volume_ADC, coordinates, 16)
            patch_BVAL = _extract_region(volume_BVAL, coordinates, 16)
            patch_ADC = np.transpose(patch_ADC[:, :, np.newaxis], (2, 1, 0))
            patch_BVAL = np.transpose(patch_BVAL[:, :, np.newaxis], (2, 1, 0))
            patch_stack = np.stack([patch_ADC, patch_BVAL])
            np.save(os.path.join(path_diff_tra_ADC_BVAL_np, name), patch_stack)
            pylab.imsave(os.path.join(path_diff_tra_ADC_BVAL_pic, name) + '.tiff',
                         _extract_region(volume_BVAL,coordinates), cmap=pylab.cm.gist_gray)


if __name__ == '__main__':
    main()
