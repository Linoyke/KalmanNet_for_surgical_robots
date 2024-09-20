# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:29:19 2024

@author: shach
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:29:19 2024

@author: shach
"""

import scipy.io
import torch
import numpy as np
import matplotlib.pyplot as plt

# Load the .mat file
data = scipy.io.loadmat('raven_suturing_simulation2.mat')

# Extract the relevant fields
configSolLeft = data['configSolLeft']
trajTime = data['trajTime'].flatten()  # Flatten to make it a 1D array
x = data['x'].flatten()
y = data['y'].flatten()
z = data['z'].flatten()

# Print shapes and some sample values to confirm data extraction
print("configSolLeft shape:", configSolLeft.shape)
print("trajTime shape:", trajTime.shape, "First few elements:", trajTime[:5])
print("x shape:", x.shape, "First few elements:", x[:5])
print("y shape:", y.shape, "First few elements:", y[:5])
print("z shape:", z.shape, "First few elements:", z[:5])

# Combine x, y, z into a single array for positions
position_data_all = np.vstack((x, y, z)).T

# Print the combined position data to verify
print("position_data_all shape:", position_data_all.shape, "First few rows:\n", position_data_all[:5, :])

# Combine encoder readings for left arm
encoder_readings_data_all = configSolLeft

# Print the encoder readings to verify
print("encoder_readings_data_all shape:", encoder_readings_data_all.shape, "First few rows:\n", encoder_readings_data_all[:5, :])

# Convert the data to numpy arrays (redundant but for confirmation)
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
    'State': state,
'Time_steps':trajTime}, 'traj3.pt')

print("Data saved successfully to 'data_RavenII.pt'")

# Load the saved data
data = torch.load('traj3.pt')
observations = data['Observation']
states = data['State']

# Verify the content of the data
print("Observation Tensor Shape:", observations.shape)
print("State Tensor Shape:", states.shape)

# Extract relevant variables
timeSteps = trajTime  # Use the correct time steps
endEffectorPositionsLeft = states.numpy()

# Print the first few rows of the state data to verify
print("endEffectorPositionsLeft shape:", endEffectorPositionsLeft.shape, "First few rows:\n", endEffectorPositionsLeft[:5, :])

# Plot the positions
plt.figure()

plt.subplot(3, 1, 1)
plt.plot(timeSteps, endEffectorPositionsLeft[:, 0], 'r--', label='X Position')
plt.title('X Position over Time')
plt.xlabel('Time [s]')
plt.ylabel('X [m]')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(timeSteps, endEffectorPositionsLeft[:, 1], 'g--', label='Y Position')
plt.title('Y Position over Time')
plt.xlabel('Time [s]')
plt.ylabel('Y [m]')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(timeSteps, endEffectorPositionsLeft[:, 2], 'b--', label='Z Position')
plt.title('Z Position over Time')
plt.xlabel('Time [s]')
plt.ylabel('Z [m]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


