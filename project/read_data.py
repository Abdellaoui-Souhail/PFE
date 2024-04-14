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

    # Court-circuité
    return 0


def get_fs_multicoil(data_dir, phase = 'train'):
    """ Returns fully sampled multicoil images of shape
    [Number of images x height x width]
    """

    data_fs = np.load('/usr/users/cei2023_2024_inserm_nir_irm/abdellaoui_sou/data.npy')

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

    data_us = np.load('/usr/users/cei2023_2024_inserm_nir_irm/abdellaoui_sou/data_kspace_undersampled_1.npy')
    masks = np.load('/usr/users/cei2023_2024_inserm_nir_irm/abdellaoui_sou/data_mask_undersampled_1.npy')
    coil_maps = data_us

    return data_us[0:1], masks[0:1], coil_maps[0:1]
