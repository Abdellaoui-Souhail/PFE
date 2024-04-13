import numpy as np
import h5py
import fastmri 
from fastmri.data import transforms as T

def LoadDataSetMultiCoil():
    
    load_dir = "/mounts/Datasets4/MICCAIChallenge2023/ChallegeData/MultiCoil/cine/TrainingSet/P"
    data = []
    desired_size = [512,512]

    for k in range(1,10):
        file_name = load_dir + "00" + str(k) + "/cine_sax.mat"
        f = h5py.File(file_name, 'r')
        newvalue = f['kspace_full']
        fullmulti = newvalue["real"] + 1j*newvalue["imag"] 
        [nframe, nslice, ncoil, ny, nx] = fullmulti.shape
        for i in range(nframe):
            for j in range(nslice):
                k_space = fullmulti[i, j]
                k_space = T.to_tensor(k_space)
                complex_image = fastmri.ifft2c(k_space)
                image_rss = fastmri.rss(complex_image, dim=0)
                image_inter = image_rss.numpy()
                image_finale = image_inter[:,:,0] + 1j*image_inter[:,:,1]
                pad_0 = desired_size[0] - image_finale.shape[0]
                pad_1 = desired_size[1] - image_finale.shape[1]
                image = np.pad(image_finale, ((0, pad_0), (0, pad_1)), mode='constant')
                image = np.expand_dims(image, axis=0)
                if len(data) == 0:
                    data = image
                else:
                    data = np.concatenate((data, image), axis=0)
                    print(data.shape)
    
    for k in range(10,80):
        file_name = load_dir + "0" + str(k) + "/cine_sax.mat"
        f = h5py.File(file_name, 'r')
        newvalue = f['kspace_full']
        fullmulti = newvalue["real"] + 1j*newvalue["imag"] 
        [nframe, nslice, ncoil, ny, nx] = fullmulti.shape
        for i in range(nframe):
            for j in range(nslice):
                k_space = fullmulti[i, j]
                k_space = T.to_tensor(k_space)
                complex_image = fastmri.ifft2c(k_space)
                image_rss = fastmri.rss(complex_image, dim=0)
                image_inter = image_rss.numpy()
                image_finale = image_inter[:,:,0] + 1j*image_inter[:,:,1]
                pad_0 = desired_size[0] - image_finale.shape[0]
                pad_1 = desired_size[1] - image_finale.shape[1]
                image = np.pad(image_finale, ((0, pad_0), (0, pad_1)), mode='constant')
                image = np.expand_dims(image, axis=0)
                if len(data) == 0:
                    data = image
                else:
                    data = np.concatenate((data, image), axis=0)
                    print(data.shape)
    
    for k in range(91,100):
        file_name = load_dir + "0" + str(k) + "/cine_sax.mat"
        f = h5py.File(file_name, 'r')
        newvalue = f['kspace_full']
        fullmulti = newvalue["real"] + 1j*newvalue["imag"] 
        [nframe, nslice, ncoil, ny, nx] = fullmulti.shape
        for i in range(nframe):
            for j in range(nslice):
                k_space = fullmulti[i, j]
                k_space = T.to_tensor(k_space)
                complex_image = fastmri.ifft2c(k_space)
                image_rss = fastmri.rss(complex_image, dim=0)
                image_inter = image_rss.numpy()
                image_finale = image_inter[:,:,0] + 1j*image_inter[:,:,1]
                pad_0 = desired_size[0] - image_finale.shape[0]
                pad_1 = desired_size[1] - image_finale.shape[1]
                image = np.pad(image_finale, ((0, pad_0), (0, pad_1)), mode='constant')
                image = np.expand_dims(image, axis=0)
                if len(data) == 0:
                    data = image
                else:
                    data = np.concatenate((data, image), axis=0)
                    print(data.shape)
    
    for k in range(100,121):
        file_name = load_dir + str(k) + "/cine_sax.mat"
        f = h5py.File(file_name, 'r')
        newvalue = f['kspace_full']
        fullmulti = newvalue["real"] + 1j*newvalue["imag"] 
        [nframe, nslice, ncoil, ny, nx] = fullmulti.shape
        for i in range(nframe):
            for j in range(nslice):
                k_space = fullmulti[i, j]
                k_space = T.to_tensor(k_space)
                complex_image = fastmri.ifft2c(k_space)
                image_rss = fastmri.rss(complex_image, dim=0)
                image_inter = image_rss.numpy()
                image_finale = image_inter[:,:,0] + 1j*image_inter[:,:,1]
                pad_0 = desired_size[0] - image_finale.shape[0]
                pad_1 = desired_size[1] - image_finale.shape[1]
                image = np.pad(image_finale, ((0, pad_0), (0, pad_1)), mode='constant')
                image = np.expand_dims(image, axis=0)
                if len(data) == 0:
                    data = image
                else:
                    data = np.concatenate((data, image), axis=0)
                    print(data.shape)
    
    return data

if __name__ == "__main__":
    LoadDataSetMultiCoil()