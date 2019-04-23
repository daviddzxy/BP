import pandas as pd
import numpy as np
import os
import pydicom
import scipy.ndimage
import pylab


def find_slices(prox_id, dcm_num, modality):
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


def get_volume(dicom_files):
    dicom_slices = []
    for dicom_file in dicom_files:
        dicom_slices.append(pydicom.dcmread(dicom_file))

    dicom_slices.sort(key=lambda x: x.SliceLocation, reverse=False)

    pixel_array = []
    for dicom_slice in dicom_slices:
        pixel_array.append(dicom_slice.pixel_array)

    return np.array(pixel_array, dtype=np.float32)


def extract_region(volume, coordinates, patch_size=None, depth=None, mode='2D'):
    # depth should be odd number
    assert volume.ndim == 3
    if mode == '2D':
        if patch_size is not None:
            patch = volume[coordinates[2], coordinates[1] - (patch_size // 2):coordinates[1] + (patch_size // 2),
                    coordinates[0] - (patch_size // 2):coordinates[0] + (patch_size // 2)]
            assert patch.shape[0] == patch_size and patch.shape[1] == patch_size
        else:
            patch = volume[coordinates[2], :, :]
    elif mode == '3D':
        assert patch_size is not None and depth is not None
        check_bot = coordinates[2] - depth // 2
        check_top = coordinates[2] + depth // 2 + 1
        over_top = 0
        over_bot = 0
        if check_bot < 0:
            over_bot = abs(check_bot)
        elif check_top > volume.shape[0]:
            over_top = check_top - volume.shape[0]

        patch = volume[(coordinates[2] - depth // 2) + over_bot - over_top:((coordinates[2] + depth // 2) + 1) + over_bot - over_top,
                    coordinates[1] - (patch_size // 2):coordinates[1] + (patch_size // 2),
                    coordinates[0] - (patch_size // 2):coordinates[0] + (patch_size // 2)]
        assert patch.shape[0] == depth and patch.shape[1] == patch_size and patch.shape[2] == patch_size
    return patch


def resample(image, curr_spacing, new_spacing):
    if not np.array_equal(curr_spacing, new_spacing):
        #need to flip spacing because of order of the axis z, y, x
        curr_spacing = np.flip(curr_spacing)
        new_spacing = np.flip(new_spacing)
        resize_factor = curr_spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        #new_spacing = spacing / real_resize_factor
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
        return image, np.flip(real_resize_factor)
    else:
        return image, 1


def main():
    path_t2_tra_pic = './Data/t2_tra_pic'
    path_t2_tra_np = './Data/t2_tra_np'
    path_t2_tra_3D_np = './Data/t2_tra_np_3D'
    path_diff_tra_ADC_BVAL_pic = './Data/diff_ADC_BVAL_pic'
    path_diff_tra_ADC_BVAL_np = './Data/diff_ADC_BVAL_np'
    path_diff_tra_ADC_BVAL_3D_np = './Data/diff_ADC_BVAL_3D_np'

    t2_spacing = np.array([0.5, 0.5, 3])
    diff_spacing = np.array([2, 2, 3])

    findings = pd.read_csv("D:/BP/Dataset_PROSTATEX/PROSTATEx lesion information/ProstateX-Findings-Train.csv")
    images = pd.read_csv("D:/BP/Dataset_PROSTATEX/PROSTATEx lesion information/ProstateX-Images-Train.csv")
    findings = findings[['ProxID', 'pos', 'fid', 'ClinSig', 'zone']]
    images = images[['ProxID', 'pos', 'Name', 'fid', 'ijk', 'Dim', 'DCMSerDescr', 'DCMSerNum', 'VoxelSpacing']]
    combined_df = pd.merge(images, findings, how='left', left_on=['ProxID', 'fid', 'pos'], right_on=['ProxID', 'fid', 'pos'])
    combined_df = combined_df[(combined_df.ProxID != 'ProstateX-0052') & (combined_df.ProxID != 'ProstateX-0025') & (combined_df.ProxID != 'ProstateX-0148')]
    #combined_df = combined_df[(combined_df.DCMSerDescr == 'ep2d_diff_tra_DYNDIST_ADC') | (combined_df.DCMSerDescr == 'ep2d_diff_tra_DYNDISTCALC_BVAL')]
    #combined_df = combined_df[(combined_df.DCMSerDescr == 't2_tse_tra')]
    #combined_df = combined_df[combined_df.zone == 'PZ']

    del images
    del findings

    combined_t2 = combined_df[combined_df['DCMSerDescr'] == 't2_tse_tra']
    
    # combined_t2 = combined_t2[combined_t2.ProxID == 'ProstateX-0001']
    for index, row in combined_t2.iterrows():
        slices = find_slices(row.ProxID, row.DCMSerNum, 't2_tra')
        if slices is not None:
            voxel_spacing = np.array([float(x) for x in row.VoxelSpacing.split(',')])
            coordinates = np.array(list(map(int, row.ijk.split())))
            # 3D
            volume = get_volume(slices)
            volume, factor = resample(volume, voxel_spacing, t2_spacing)
            coordinates = np.floor(coordinates * factor).astype(int)
            name = str(row.ClinSig) + " FID " + str(row.fid) + ' ' + str(row.ProxID) + " IJK " + str(coordinates) + " DCM " + str(row.DCMSerNum)
            np.save(os.path.join(path_t2_tra_3D_np, name), volume)
            # 2D
            patch_3D = extract_region(volume, coordinates, 24, 5, '3D')
            patch_2D = extract_region(volume, coordinates, 24)
            patch_3D = patch_3D[np.newaxis, :, :, :]
            # make one channel image with shape of [channels, y, x]
            # pylab.imsave(os.path.join(path_t2_tra_pic, name) + '.tiff', extract_region(volume, coordinates), cmap=pylab.cm.gist_gray)
            patch_2D = patch_2D[np.newaxis, :, :]
            np.save(os.path.join(path_t2_tra_np, name), patch_2D)
            np.save(os.path.join(path_t2_tra_3D_np, name), patch_3D)


    combined_ADC = combined_df[(combined_df['DCMSerDescr'] == 'ep2d_diff_tra_DYNDIST_ADC') | (combined_df['DCMSerDescr'] == 'ep2d_diff_tra_DYNDIST_MIX_ADC')]
    combined_BVAL = combined_df[(combined_df['DCMSerDescr'] == 'ep2d_diff_tra_DYNDISTCALC_BVAL') | (combined_df['DCMSerDescr'] == 'ep2d_diff_tra_DYNDIST_MIXCALC_BVAL')]
    combined_diff = pd.merge(combined_ADC, combined_BVAL, how='left', left_on=['ProxID', 'fid', 'pos', 'ijk', 'Dim', 'zone', 'ClinSig'], right_on=['ProxID', 'fid', 'pos', 'ijk', 'Dim', 'zone', 'ClinSig'])
    combined_diff = combined_diff[combined_diff.ProxID != 'ProstateX-0154']
    for index, row in combined_diff.iterrows():
        slices_ADC = find_slices(row.ProxID, row.DCMSerNum_x, 'diff_ADC')
        slices_BVAL = find_slices(row.ProxID, row.DCMSerNum_y, 'diff_BVAL')
        if slices_ADC is not None and slices_BVAL is not None:
            voxel_spacing = np.array([float(x) for x in row.VoxelSpacing_x.split(',')])
            coordinates = np.array(list(map(int, row.ijk.split())))
            volume_ADC = get_volume(slices_ADC)
            volume_BVAL = get_volume(slices_BVAL)
            volume_ADC, factor_ADC = resample(volume_ADC, voxel_spacing, diff_spacing)
            volume_BVAL, factor_BVAL = resample(volume_BVAL, voxel_spacing, diff_spacing)
            assert np.array_equal(factor_BVAL, factor_ADC)
            coordinates = np.floor(coordinates * factor_ADC).astype(int)
            name = str(row.ClinSig) + " FID " + str(row.fid) + ' ' + str(row.ProxID) + " IJK " + str(coordinates) + " DCM " + str(row.DCMSerNum_x)
            patch_ADC = extract_region(volume_ADC, coordinates, 16)
            patch_BVAL = extract_region(volume_BVAL, coordinates, 16)
            patch_3D_ADC = extract_region(volume_ADC, coordinates, 16, 15, '3D')
            patch_3D_BVAL = extract_region(volume_BVAL, coordinates, 16, 15, '3D')
            patch_3D_stack = np.stack([patch_3D_ADC, patch_3D_BVAL])
            patch_stack = np.stack([patch_ADC, patch_BVAL])
            np.save(os.path.join(path_diff_tra_ADC_BVAL_np, name), patch_stack)
            np.save(os.path.join(path_diff_tra_ADC_BVAL_3D_np, name), patch_3D_stack)
            # pylab.imsave(os.path.join(path_diff_tra_ADC_BVAL_pic, name) + '.tiff',
            #              extract_region(volume_ADC, coordinates), cmap=pylab.cm.gist_gray)


if __name__ == '__main__':
    main()
