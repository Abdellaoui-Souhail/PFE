import numpy as np
import h5py
import fastmri 
import os
from fastmri.data import transforms as T
from project.utils import dist_util, logger

def LoadDataSetSingleCoil(load_dir, variable = "data_fs"):

    
    
    f = h5py.File(load_dir, 'r')
    if variable == "data_fs" or variable == "us_masks":
        data = np.expand_dims(np.transpose(np.array(f[variable]), (0,2,1)), axis=1).astype(np.float32)
    elif variable == "data_us":
        data = np.transpose(np.array(f[variable]), (1,0,3,2))
        phase_ = (data[:,1,:,:] * 2 * np.pi) - np.pi
        data = data[:,0,:,:] * np.exp(1j * phase_)
    return data


def LoadDataSetMultiCoil(load_dir, **kwargs):

    # Court-circuité
    return 0


def get_fs_singlecoil(data_dir, phase = 'train'):
    """
    Returns fully sampled singlecoil images of shape
    [Number of images x height x width]
    """

    print("Loading T1 images")
    target_file = data_dir + "T1_2_multi_synth_recon_" + str(phase) + ".mat"
    data_fs_t1 = LoadDataSetSingleCoil(target_file)

    print("Loading T2 images")
    target_file = data_dir + "T2_2_multi_synth_recon_" + str(phase) + ".mat"
    data_fs_t2 = LoadDataSetSingleCoil(target_file)

    print("Loading PD images")
    target_file = data_dir + "PD_2_multi_synth_recon_" + str(phase) + ".mat"
    data_fs_pd = LoadDataSetSingleCoil(target_file)

    data_fs = np.concatenate((data_fs_t1, data_fs_t2, data_fs_pd), axis = 0)
    data_fs = np.squeeze(data_fs)
    print("data_fs.shape: " + str(data_fs.shape))
    return data_fs


def get_us_singlecoil(data_dir, phase = 'test', contrast= 'T1', R = 4):
    """
    Returns singlecoil undersampled images and undersampling masks of shape
    [Number of images x height x width]
    """

    print("Loading " + contrast + " " + str(R) + "x Data")
    target_file = data_dir + contrast + "_" + str(R) + "_multi_synth_recon_" + str(phase) + ".mat"
    data_us = LoadDataSetSingleCoil(target_file, variable = "data_us")
    us_masks = LoadDataSetSingleCoil(target_file, variable = "us_masks")

    data_us = np.squeeze(data_us)
    us_masks = np.squeeze(us_masks)

    print(data_us.shape, us_masks.shape)

    return data_us, us_masks


def get_fs_multicoil(data_dir, phase = 'train'):
    """ Returns fully sampled multicoil images of shape
    [Number of images x height x width]
    """

    data_fs = np.load('/usr/users/cei2023_2024_inserm_nir_irm/abdellaoui_sou/data_6.npy')

    return data_fs


def get_us_multicoil(data_dir, phase='test', contrast= 'T1', R = 4):
    """
    Returns undersampled images of shape
    [Number of images x number of coils x height x width],
    undersampling masks of shape
    [Number of images x 1 x height x width],
    coil sensitivity maps of shape
    [Number of images x number of coils x height x width]
    """

    print("Loading " + contrast + " " + str(R) + "x Data")

    target_file = data_dir + contrast + "/" + contrast + "_under_sampled_" + str(R) + "x_multicoil_" + str(phase) + ".mat"
    data_fs = LoadDataSetMultiCoil(target_file, 'images_fs', padding = False, Norm = True, channel_cat = False)
    data_us = LoadDataSetMultiCoil(target_file, 'images_us', padding = False, Norm = True, channel_cat = False)
    masks = LoadDataSetMultiCoil(target_file, 'map', padding = False, Norm = False, is_complex = False, channel_cat = False)
    coil_maps = LoadDataSetMultiCoil(target_file, 'coil_maps', padding = False, Norm = False, channel_cat = False)

    print(data_us.shape, masks.shape, coil_maps.shape)

    return data_us, masks, coil_maps
