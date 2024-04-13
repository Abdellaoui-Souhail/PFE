import numpy as np
import h5py
import fastmri 
from fastmri.data import transforms as T
import os 



def Mamamia(**kwargs):

    
    load_dir = "/usr/users/cei2023_2024_inserm_nir_irm/abdellaoui_sou/data_"

    parties = []
    for k in range(1,8):
        file = load_dir + str(k) + ".npy"
        partie = np.load(file)
        parties.append(partie)
        
    data = np.concatenate(parties, axis=0)
    np.save('/usr/users/cei2023_2024_inserm_nir_irm/abdellaoui_sou/data.npy', data)
    print(data.shape)
    return 0

if __name__ == "__main__":
    Mamamia()