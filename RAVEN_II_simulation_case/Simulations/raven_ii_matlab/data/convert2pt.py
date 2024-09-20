# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:17:47 2024

@author: shach
"""

import scipy.io
import torch
import numpy as np

# Load the .mat file
data = scipy.io.loadmat('try1.mat')

# Get all timestamps
timestamps = data['data'].dtype.names  # Proper way to access names of fields

# Initialize lists to store wrist and grasper data
wrist_L_data_all = []
grasper1_L_data_all = []

# Extract wrist_L and grasper1_L data from each timestamp
for timestamp in timestamps:
    wrist_L_data = data['data'][timestamp][0, 0]['wrist_L'][0, 0]
    grasper1_L_data = data['data'][timestamp][0, 0]['grasper1_L'][0, 0]
    wrist_L_data_all.append(wrist_L_data)
    grasper1_L_data_all.append(grasper1_L_data)

# Convert lists to PyTorch tensors
test_input = torch.tensor(np.array(wrist_L_data_all), dtype=torch.float32)
test_target = torch.tensor(np.array(grasper1_L_data_all), dtype=torch.float32)

# Print tensor shapes to confirm
print("Wrist L Tensor Shape:", test_input.shape)
print("Grasper1 L Tensor Shape:", test_target.shape)


torch.save({
    'Observation': test_input,
    'State': test_target
}, 'data.pt')

print("Data saved successfully to 'data.pt'")