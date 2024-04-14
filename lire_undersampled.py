import numpy as np
import h5py
import fastmri 
from fastmri.data import transforms as T
import os 



def Mamamia(**kwargs):

    load_dir = "/mounts/Datasets4/MICCAIChallenge2023/ChallegeData/MultiCoil/cine/TrainingSet/AccFactor04"
    data = []
    # Desired dimensions
    desired_ny, desired_nx = 512, 512
    k_space_list = []
    mask_list = []

    # Adjusted to handle file indexing
    for k in range(14, 28):
        if k < 10:
            k_index = f"P00{k}"
        elif k < 100:
            k_index = f"P0{k}"
        else:
            k_index = f"P{k}"

        # Skip indices between 80 and 90
        if 80 <= k <= 90:
            continue

        file_name_1 = os.path.join(load_dir, k_index, "cine_sax.mat")
        file_name_2 = os.path.join(load_dir, k_index, "cine_sax_mask.mat")
        
        hf_1 = h5py.File(file_name_1, 'r')
        k_space = hf_1['kspace_sub04']
        k_space = k_space["real"] + 1j * k_space["imag"]
        [nframe, nslice, ncoil, ny, nx] = k_space.shape
        k_space = k_space.reshape(nframe*nslice, ncoil, ny, nx)
        
        hf_2 = h5py.File(file_name_2, 'r')
        mask = hf_2['mask04']
        mask = np.array(mask)
        mask = mask.reshape(1, 1, ny, nx)

        # Calculate padding amounts
        pad_y = (desired_ny - ny) if ny < desired_ny else 0
        pad_x = (desired_nx - nx) if nx < desired_nx else 0

        # Padding to be applied on both sides of the dimensions
        pad_y_before = pad_y // 2
        pad_y_after = pad_y - pad_y_before
        pad_x_before = pad_x // 2
        pad_x_after = pad_x - pad_x_before

        # Apply padding
        k_space = np.pad(
            k_space, pad_width=((0, 0), (0, 0), (pad_y_before, pad_y_after), (pad_x_before, pad_x_after)), mode='constant', constant_values=(0, 0)
        )

        mask = np.pad(
            mask, pad_width=((0, 0), (0, 0), (pad_y_before, pad_y_after), (pad_x_before, pad_x_after)), mode='constant', constant_values=(1, 1)
        )


        print(k_space.shape)
        print(mask.shape)


        k_space_list.append(k_space)
        mask_list.append(mask)


    # Concatenate all images at once
    data_kspace = np.concatenate(k_space_list, axis=0)
    data_mask = np.concatenate(mask_list, axis=0)
    np.save('/usr/users/cei2023_2024_inserm_nir_irm/abdellaoui_sou/data_kspace_undersampeled_2.npy', data_kspace)
    np.save('/usr/users/cei2023_2024_inserm_nir_irm/abdellaoui_sou/data_mask_undersampeled_2.npy', data_mask)
    return 0

if __name__ == "__main__":
    Mamamia()