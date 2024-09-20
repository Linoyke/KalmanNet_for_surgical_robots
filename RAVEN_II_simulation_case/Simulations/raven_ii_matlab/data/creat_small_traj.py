# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:45:55 2024

@author: shach
"""
import torch
import os
import matplotlib.pyplot as plt

# Define the directory where the .pt files are located
directory = r'C:\Users\shach\Documents\shachar\project_code\python\KalmanNet_TSP-main\Simulations\Raven_ii_matlab\data'
ADD_bias = True

# Load the .pt files
file_paths = [ 'traj_spirale0.pt','traj_spirale2.pt','traj_spirale1.pt','traj_spirale3.pt','traj_spirale4.pt','traj_spirale5.pt','traj_spirale6.pt','traj_spirale7.pt']
observations_list = []
states_list = []

for file_path in file_paths:
    data = torch.load(os.path.join(directory, file_path))
    observations = data['Observation']
    states = data['State']
    print(f"File: {file_path}, Observations shape: {observations.shape}, States shape: {states.shape}")
    observations_list.append(observations)
    states_list.append(states)

# Concatenate all observations and states into one long trajectory along the time step dimension
all_observations = torch.cat(observations_list, dim=0)
all_states = torch.cat(states_list, dim=0)

### ADD Time Variant Noise #####
if ADD_bias:
# Calculate BL as a tensor
    alpha = 0.1
    t = all_observations.size(0)
    omega = 2 / t
    time_indices = torch.arange(t, dtype=torch.float32)
    BL = alpha * (1 - torch.cos(omega * time_indices))
    std = 0.5

# Ensure BL is a tensor and expand to match the shape of obs_reshaped
    BL_expanded = BL.unsqueeze(1).expand(-1, 3)
# Generate Gaussian noise with the same shape as obs_reshaped
    gaussian_noise = torch.randn_like(all_observations)

# Adjust the mean of the Gaussian noise to BL
    noise_with_mean_BL = std*gaussian_noise + BL_expanded

# Add the noise to obs_reshaped
    all_observations_noisy = all_observations + noise_with_mean_BL

# Convert to numpy arrays for plotting
    all_observations_np = all_observations.numpy()
    all_observations_noisy_np = all_observations_noisy.numpy()


# Plotting
    plt.figure(figsize=(12, 6))

    # Plot each dimension separately
    for i in range(3):
        plt.subplot(3, 1, i+1)  # Create a subplot for each dimension
        plt.plot(all_observations_np[:10, i], label='Original Observations')
        plt.plot(all_observations_noisy_np[:10, i], label='Noisy Observations', linestyle='dashed')
        plt.title(f'Dimension {i+1}')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()

    plt.tight_layout()
    plt.show()
    print(f"Concatenated observations shape: {all_observations.shape}")
    print(f"Concatenated states shape: {all_states.shape}")
    
    # Add the noise to all_states as additional states
    # The noise needs to be inserted to the first dimension while the second dimension is which bias it is
    all_states_noisy = torch.cat((all_states, noise_with_mean_BL), dim=1)
    print(f"Concatenated states shape: {all_states_noisy.shape}")
    
    all_observations = all_observations_noisy
    all_states = all_states_noisy

# Ensure that the total length is a multiple of the desired trajectory length (50)
traj_length = 100
total_length = all_observations.shape[0]
num_traj = total_length // traj_length

if num_traj == 0:
    raise ValueError("The total length of the concatenated trajectory is less than the desired trajectory length (50).")

# Truncate the long trajectories to make them a multiple of traj_length
all_observations = all_observations[:num_traj * traj_length]
all_states = all_states[:num_traj * traj_length]

# Split the long trajectories into smaller trajectories of length 50
obs_reshaped = all_observations.view(num_traj, traj_length, -1).permute(0, 2, 1)
states_reshaped = all_states.view(num_traj, traj_length, -1).permute(0, 2, 1)

# Check the shape to ensure it's correct
print(f"Reshaped observations shape: {obs_reshaped.shape}")  # Should be [num_traj, dimensions=3, traj_length=50]
print(f"Reshaped states shape: {states_reshaped.shape}")  # Should be [num_traj, dimensions=3, traj_length=50]

# Save the new trajectories if needed
torch.save({'observations': obs_reshaped, 'states': states_reshaped}, os.path.join(directory, 'processed_trajectories_710on100.pt'))

print(f"Observations shape: {obs_reshaped.shape}")  # Should be [num_traj, dimensions=3, traj_length=50]
print(f"States shape: {states_reshaped.shape}")  # Should be [num_traj, dimensions=3, traj_length=50]
