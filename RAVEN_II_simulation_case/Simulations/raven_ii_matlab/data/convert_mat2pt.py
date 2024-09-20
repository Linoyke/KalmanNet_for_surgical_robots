# -*- coding: utf-8 -*-
"""
Created on Tue May 28 18:45:27 2024

@author: shach
"""
import scipy.io
import torch
import numpy as np

# Load the .mat file
data = scipy.io.loadmat('traj_spirale7.mat')



endEffectorPositionsLeft = data['endEffectorPositionsLeft']

encoderReadingsLeft = data['configSolLeft']

#timeSteps = data['timeSteps']

# Initialize lists to store data
position_data_all = []
encoder_readings_data_all = []

# Extract position (endEffectorPositionsLeft) and encoder readings (encoderReadingsLeft)
position_data_all = endEffectorPositionsLeft
encoder_readings_data_all = encoderReadingsLeft

# Convert the data to numpy arrays
position_data_all = np.array(position_data_all)
encoder_readings_data_all = np.array(encoder_readings_data_all)

# Convert lists to PyTorch tensors
observation = torch.tensor(encoder_readings_data_all, dtype=torch.float32)
state = torch.tensor(position_data_all, dtype=torch.float32)

# Print tensor shapes to confirm
print("Observation Tensor Shape:", observation.shape)
print("State Tensor Shape:", state.shape)

# Save the tensors to a .pt file
torch.save({
    'Observation': observation,
    'State': state
}, 'traj_spirale7.pt')

print("Data saved successfully to 'data.pt'")


