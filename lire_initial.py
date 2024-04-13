import numpy as np
import h5py
import fastmri 
from fastmri.data import transforms as T
import os 



def LoadDataSetMultiCoil(**kwargs):

    load_dir = "/mounts/Datasets4/MICCAIChallenge2023/ChallegeData/MultiCoil/cine/TrainingSet"
    data = []
    # Desired dimensions
    desired_ny, desired_nx = 512, 512
    images_list = []

    # Adjusted to handle file indexing
    for k in range(110, 121):
        if k < 10:
            k_index = f"P00{k}"
        elif k < 100:
            k_index = f"P0{k}"
        else:
            k_index = f"P{k}"

        # Skip indices between 80 and 90
        if 80 <= k <= 90:
            continue

        file_name = os.path.join(load_dir, k_index, "cine_sax.mat")
        with h5py.File(file_name, 'r') as f:
            newvalue = f['kspace_full']
            fullmulti = newvalue["real"] + 1j * newvalue["imag"]
            [nframe, nslice, ncoil, ny, nx] = fullmulti.shape
            
            fullmulti = T.to_tensor(fullmulti)

            complex_image = fastmri.ifft2c(fullmulti)
            image_rss = fastmri.rss(complex_image, dim=2)
            image_rss = image_rss.numpy()
            image = image_rss[:,:,:,:,0] + 1j*image_rss[:,:,:,:,1]
            print(image.shape)

            # Calculate padding amounts
            pad_y = (desired_ny - ny) if ny < desired_ny else 0
            pad_x = (desired_nx - nx) if nx < desired_nx else 0

            # Padding to be applied on both sides of the dimensions
            pad_y_before = pad_y // 2
            pad_y_after = pad_y - pad_y_before
            pad_x_before = pad_x // 2
            pad_x_after = pad_x - pad_x_before

            # Apply padding
            image = np.pad(
                image, pad_width=((0, 0), (0, 0), (pad_y_before, pad_y_after), (pad_x_before, pad_x_after)), mode='constant', constant_values=(0, 0)
            )

            image_list = image.reshape(nframe * nslice, 512, 512)
            print(image_list.shape)
            images_list.append(image_list)


    # Concatenate all images at once
    data = np.concatenate(images_list, axis=0)
    np.save('/usr/users/cei2023_2024_inserm_nir_irm/abdellaoui_sou/data_7.npy', data)
    print(data.shape)
    return data

if __name__ == "__main__":
    LoadDataSetMultiCoil()