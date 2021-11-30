#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import maxabs_scale
import os


# In[2]:


class GetData:
    def __init__(self, path, signal_size=4096):
        real_classes = [d for d in os.listdir(os.path.join(path, '真实信号')) if
                        os.path.isfile(os.path.join(path, '真实信号', d))]
        simulation_classes = [d for d in os.listdir(os.path.join(path, '模拟信号')) if
                              os.path.isfile(os.path.join(path, '模拟信号', d))]
        real_classes.sort(key=lambda x: int(x[0:-4]))
        simulation_classes.sort(key=lambda x: int(x[0:-4]))
        real_signal = []
        simulation_signal = []
        for file_path in real_classes:
            real_signal.append(np.loadtxt(os.path.join(path, '真实信号', file_path)))
        for file_path in simulation_classes:
            simulation_signal.append(np.loadtxt(os.path.join(path, '模拟信号', file_path)))

        real_signal = np.array(real_signal)
        simulation_signal = np.array(simulation_signal)
        simulation_signal = np.pad(simulation_signal, ((0, 0), (0, signal_size - simulation_signal.shape[-1])),
                                   'constant', constant_values=(0, 0))

        #         self.max_abs_scaler = MaxAbsScaler(axis=1)
        #         data = np.concatenate((real_signal,simulation_signal),axis=0)
        #         self.max_abs_scaler.fit_transform(data)

        #         real_signal = self.max_abs_scaler.transform(real_signal)
        #         simulation_signal = self.max_abs_scaler.transform(simulation_signal)

        real_signal = maxabs_scale(real_signal, axis=1)
        simulation_signal = maxabs_scale(simulation_signal, axis=1)
        self.real_signal = real_signal[:, np.newaxis, :]
        #         print(self.real_signal)
        self.simulation_signal = simulation_signal[:, np.newaxis, :]

    #         print(self.simulation_signal)

    #     def get_scaler(self):
    #         return self.max_abs_scaler

    def get_data(self):
        return self.real_signal, self.simulation_signal


#     def get_origin_data(self, data): # input_size:[*,4096],output_size:[*,4096]
#         origin_data = self.max_abs_scaler(data)
#         return origin_data


# In[3]:


class GetDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        data = self.data[index]
        return torch.Tensor(data)

    def __len__(self):
        return len(self.data)

# In[4]:


# # test
# Traindata = GetTraindata('data')
# real_signal,simulation_signal = Traindata.get_data()
# max_abs_scaler = Traindata.get_scaler()
# traindataA = GetDataset(real_signal)
# traindataB = GetDataset(simulation_signal)
# max_abs_scaler.transform(np.ones([1,4096]))


# In[ ]:
