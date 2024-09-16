# 2 arms planar RR robot simulation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datetime import datetime
import os
import matplotlib.gridspec as gridspec


def forward_kinematics(theta1, theta2, l1, l2):
    # Calculate end effector position (X, Y)
    x = l1 * torch.cos(theta1) + l2 * torch.cos(theta1 + theta2)
    y = l1 * torch.sin(theta1) + l2 * torch.sin(theta1 + theta2)
    return x, y

def inverse_kinematics(x, y, l1, l2):
    theta2 = torch.acos((x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2))
    theta1 = torch.atan2(y, x) - torch.atan2(l2 * torch.sin(theta2), l1 + l2 * torch.cos(theta2))
    return theta1, theta2

def is_theta1_valid(delta, obs_theta1):
    new_theta1 = obs_theta1 + delta
    is_in_range = -180 <= new_theta1 <= 180
    return is_in_range

def is_theta2_valid(delta, obs_theta2):
    new_theta2 = obs_theta2 + delta
    is_in_range = 0 <= new_theta2 <= 180
    return is_in_range

def is_x_y_valid(x,y,l1,l2):
    r = torch.sqrt(x**2 +y**2)
    max_r = l1 + l2
    min_r = l1 -l2
    if r < max_r and r > min_r:
        return True
    else:
        return False
    

def to_excel(end_effector_positions, end_effector_positions_true, joint_angles_obs, joint_angles, excel_file_name):
    # Convert tensors to NumPy arrays
    # Reshape tensors to have separate columns for x and y
    end_effector_positions_reshaped = end_effector_positions.view(-1, end_effector_positions.size(1))
    end_effector_positions_true_reshaped = end_effector_positions_true.view(-1, end_effector_positions_true.size(1))
    joint_angles_obs_reshaped = joint_angles_obs.view(-1, joint_angles_obs.size(1))
    joint_angles_reshaped = joint_angles.view(-1, joint_angles.size(1))

    # Convert reshaped tensors to NumPy arrays
    end_effector_positions_np = end_effector_positions_reshaped.numpy()
    end_effector_positions_true_np = end_effector_positions_true_reshaped.numpy()
    joint_angles_obs_np = joint_angles_obs_reshaped.numpy()
    joint_angles_np = joint_angles_reshaped.numpy()
    
    # Create a DataFrame
    df = pd.DataFrame({
    'end_effector_positions_x': end_effector_positions_np[0, :],
    'end_effector_positions_y': end_effector_positions_np[1, :],
    'end_effector_positions_true_x': end_effector_positions_true_np[0, :],
    'end_effector_positions_true_y': end_effector_positions_true_np[1, :],
    'joint_angles_obs_t1': joint_angles_obs_np[0, :],
    'joint_angles_obs_t2': joint_angles_obs_np[1, :],
    'joint_angles_t1': joint_angles_np[0, :],
    'joint_angles_t2': joint_angles_np[1, :],
    })
    
    # Load existing Excel file or create a new one
    if os.path.exists(excel_file_name):
        with pd.ExcelWriter(excel_file_name, engine='openpyxl', mode='a') as writer:
            # Determine the next iteration number
            next_iteration = len(writer.book.sheetnames) + 1
            # Create sheet name with unique iteration number
            sheet_name = f'Iteration_{next_iteration}'
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        df.to_excel(excel_file_name, index=False)
    
    
def plotXY(dt,num_steps,end_effector_positions):
    # Create an array representing time elapsed at each step
    times = torch.arange(0, num_steps * dt, dt)

    # Extract X and Y coordinates from the tensor
    x_positions = end_effector_positions[0, :]
    y_positions = end_effector_positions[1, :]

    # Plot XY positions with lines connecting points
    plt.plot(x_positions, y_positions, '-o', label='Trajectory')

    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('XY Position Vs Time')

    # Display the legend
    plt.legend()

    # Show the plot
    plt.show()
    
def generate_noise(omega, duration, R, alpha, time_steps):
    t = np.linspace(0, duration, time_steps)  # Generate time values
    mean = alpha*(1 - np.cos(omega * t))  # Calculate the mean function values
    R_new = np.eye(time_steps) * R
    # Generate noise samples
    noise = np.random.multivariate_normal(mean, R_new)
    
    # plt.figure()
    # plt.plot(t, noise,'-*')
    # plt.title("Noise with Mean 1 - cos(ωt) and Covariance Q")
    # plt.xlabel("Time [sec]")
    # plt.ylabel("noise value")
    # plt.grid(True)
    # plt.show()
    return noise

def simulate_robot(num_trajectories,time_steps,l1,l2,to_plot,Q_STD,R_STD,omega,alpha):
    #from Simulations.RR_planar_robot2.parameters import var_delta, var_noise
    # Allocate Empty Array for Input
    x_state = torch.empty(num_trajectories, 2, time_steps)
    
    # Allocate Empty Array for Target
    y_obs = torch.empty(num_trajectories, 2, time_steps)
    
    # true label
    x_target = torch.empty(num_trajectories, 2, time_steps)
    y_true = torch.empty(num_trajectories, 2, time_steps)
    # saving 
    noises = torch.zeros((num_trajectories, 2, time_steps))
    delta_thetas = torch.zeros((num_trajectories, 2, time_steps))
    delta_thetas_observed = torch.zeros((num_trajectories, 2, time_steps))
    
    
    # Generate random time step the robot works according to
    dt = 0.08 # sec
    
    duration = time_steps*dt #sec 
    
    # allocate memory
    end_effector_positions = torch.zeros(2, time_steps) # holds position as accepted from robot
    end_effector_positions_true = torch.zeros(2, time_steps) # holds true position
    joint_angles = torch.zeros(2, time_steps) # holds true theta1 and theta2 values
    joint_angles_obs = torch.zeros(2, time_steps) # holds observation theta1 & theta2 values (as accepted from robot)
    is_bakclash_array = torch.zeros(2, time_steps) # first col t1 secend col t2
    delta_theta = torch.zeros(2, time_steps) # holding the change in joint angles
    delta_theta_observed = torch.zeros(2, time_steps)
    first_true_theta = torch.zeros(2, time_steps)
    
    # Get current date and time
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    # creteaing trajecrtories
    for traj in range(num_trajectories):
        
        
        print("===============================")
        print("starting trajectory number: ", traj)
        # initial angle values
        theta1_deg = torch.randint(20, 90,(1,)) #deg
        theta2_deg = torch.randint(60, 90,(1,)) #deg
        # initial position using FK
        x, y = forward_kinematics(torch.deg2rad(theta1_deg), torch.deg2rad(theta2_deg), l1, l2)

        # generating noise according to u_t
        noise_t1 = generate_noise(omega, duration, R_STD, alpha, time_steps)
        noise_t2 = generate_noise(omega, duration, R_STD, alpha, time_steps)
        
        # Convert noise arrays to torch tensors
        noise_t1_tensor = torch.from_numpy(noise_t1)
        noise_t2_tensor = torch.from_numpy(noise_t2)

        # saving noises in tensor
        noises[traj,0,:] = noise_t1_tensor
        noises[traj,1,:] = noise_t2_tensor
        
        # add validity check
        theta1_obs_valid = False
        theta2_obs_valid = False
        
        while not(theta1_obs_valid):
            noise1 = noise_t1[0]
            theta1_obs_valid = is_theta1_valid(noise1, theta1_deg)
            
        while not(theta2_obs_valid):
            noise2 = noise_t2[0]
            theta2_obs_valid = is_theta2_valid(noise2, theta2_deg)
        
        obs_theta1 = theta1_deg + noise1 # deg
        obs_theta2 = theta2_deg + noise2 # deg
        
        # calc of position based on observation
        x_obs_val, y_obs_val = forward_kinematics(torch.deg2rad(obs_theta1), torch.deg2rad(obs_theta2), l1, l2)
        
        # Saving observations and true values in tensors
        joint_angles_obs[:, 0] = torch.cat([obs_theta1, obs_theta2], dim=0)
        joint_angles[:, 0] = torch.cat([theta1_deg, theta2_deg], dim=0)
        end_effector_positions[:, 0] = torch.cat([x_obs_val, y_obs_val], dim=0)
        end_effector_positions_true[:, 0] = torch.cat([x, y], dim=0)
        zero_tensor = torch.zeros(1)
        delta_theta[:, 0]  = torch.cat([zero_tensor, zero_tensor], dim=0)
        delta_theta_observed[:, 0]  = torch.cat([zero_tensor, zero_tensor], dim=0)
        for i in range(1,time_steps):
            
            xy_valid = False
            theta2_valid = False
            theta1_obs_valid = False
            theta2_obs_valid = False
            
            # is delta valid
            while not(xy_valid):
                delta_x = torch.randn(1)*Q_STD
                delta_y = torch.randn(1)*Q_STD
                xy_valid = is_x_y_valid(x+delta_x,y+delta_y,l1,l2)
                
            # updating state
            x = x + delta_x
            y = y + delta_y
            
            # getting true angles
            theta1, theta2= inverse_kinematics(x, y, l1, l2)
            theta1_deg = torch.rad2deg(theta1)
            theta2_deg = torch.rad2deg(theta2)
            
            # saving the change in angles
            delta_theta1 = theta1_deg - joint_angles[0, i-1] 
            delta_theta2 = theta2_deg - joint_angles[1, i-1] 
            
            # adding noise to delta_theta to describe the fact that the described delta_theta is not accurate
            delta_theta1_obs = delta_theta1 + torch.randn(1)*R_STD
            delta_theta2_obs = delta_theta2 + torch.randn(1)*R_STD
                    
            # checking validity, if adding the noise to theta1 and saving as obsereved data
            while not(theta1_obs_valid):
                noise1 = noise_t1[i]
                theta1_obs_valid = is_theta1_valid(noise1, theta1_deg)
                
            while not(theta2_obs_valid):
                noise2 = noise_t2[i]
                theta2_obs_valid = is_theta2_valid(noise2, theta2_deg)
            
            # updating obsereved theta1 & theta2 with noise
            # by addig the noise to the true angle
            obs_theta1 = theta1_deg + noise1 
            obs_theta2 = theta2_deg + noise2 

            # Save the angles in tensor
            joint_angles_obs[:, i] = torch.cat([obs_theta1, obs_theta2], dim=0)
            joint_angles[:, i] = torch.cat([theta1_deg, theta2_deg], dim=0)
            delta_theta[:, i] = torch.cat([delta_theta1 , delta_theta2], dim=0)
            delta_theta_observed[:, i] = torch.cat([delta_theta1_obs , delta_theta2_obs], dim=0)
            end_effector_positions_true[:, i] = torch.cat([x, y], dim=0)
            # Compute end effector positions considering observation theta1 & theta2 
            x_obs_val, y_obs_val = forward_kinematics(torch.deg2rad(obs_theta1), torch.deg2rad(obs_theta2), l1, l2)
            end_effector_positions[:, i] = torch.cat([x_obs_val, y_obs_val], dim=0)
         
        t = np.arange(0, time_steps * dt, dt)
        
        # saving traj data
        x_state[traj] = end_effector_positions # state according to the noisy observations
        x_target[traj] = end_effector_positions_true
        y_obs[traj] = joint_angles_obs
        y_true[traj] = joint_angles
        delta_thetas[traj] = delta_theta
        delta_thetas_observed[traj] = delta_theta_observed
        if to_plot:
            fig = plt.figure(figsize=(10, 8))
            gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
            
            # Create the first plot (ax1), spanning both columns at the bottom
            ax1 = fig.add_subplot(gs[1, :])
            ax1.plot(end_effector_positions[0, :], end_effector_positions[1, :], 'o',linestyle='-', label='Observations')
            ax1.plot(end_effector_positions_true[0, :], end_effector_positions_true[1, :], '^',linestyle='--', label='True Position')
            ax1.set_xlabel('X Axis position [cm]', fontsize=15)
            ax1.set_ylabel('Y Axis position [cm]', fontsize=15)
            ax1.set_title('End Effector Position', fontsize=18)
            ax1.legend(fontsize=16)
            ax1.grid(True)
            
            # Create the second plot (ax2) on the top-left
            ax2 = fig.add_subplot(gs[0, 0])
            ax2.plot(t, joint_angles_obs[0, :], 'o',linestyle='-', label='Observed Angle')
            ax2.plot(t, joint_angles[0, :], '^', linestyle='--',label='True Angle')
            ax2.set_xlabel('Time [Sec]', fontsize=15)
            ax2.set_ylabel('Angle [Deg]', fontsize=15)
            ax2.set_title(r'$\theta_1$ as Function of Time', fontsize=18)
            ax2.legend(fontsize=16)
            ax2.grid(True)
            
            # Create the third plot (ax3) on the top-right
            ax3 = fig.add_subplot(gs[0, 1])
            ax3.plot(t, joint_angles_obs[1, :], 'o',linestyle='-', label='Observed Angle')
            ax3.plot(t, joint_angles[1, :], '^', linestyle='--',label='True Angle')
            ax3.set_xlabel('Time [Sec]', fontsize=15)
            ax3.set_ylabel('Angle [Deg]', fontsize=15)
            ax3.set_title(r'$\theta_2$ as Function of Time', fontsize=18)
            ax3.legend(fontsize=16)
            ax3.grid(True)
            
            # Automatically adjust the spacing between subplots
            plt.tight_layout()
            
            plt.show()
            #plt.show()
            
            plt.figure(figsize=(10, 11))  # Adjusted size: width=8, height=15
            plt.subplot(2, 1, 1)
            plt.plot(t, end_effector_positions[0,:],label="Observations",marker='o')
            plt.plot(t, end_effector_positions_true[0,:],label="True Position", marker='s')
            plt.title("X - Axis Position ", fontsize=18)
            plt.legend(fontsize=12)
            plt.xlabel("Time [Sec]", fontsize=16)
            plt.ylabel("Position [cm]", fontsize=16)
            plt.grid(True)
            plt.tight_layout()  # Adjust subplots to fit into figure area.
            
            plt.subplot(2, 1, 2)
            plt.plot(t, end_effector_positions[1,:],label="Observations",marker='o')
            plt.plot(t, end_effector_positions_true[1,:],label="True Position", marker='s')
            plt.title("Y - Axis Position ", fontsize=18)
            plt.legend(fontsize=12)
            plt.xlabel("Time [Sec]", fontsize=16)
            plt.ylabel("Position [cm]", fontsize=16)
            plt.grid(True)
            plt.tight_layout()  # Adjust subplots to fit into figure area.
            plt.show()
        
        print('finished creating trajectory number: ', traj)
    # save to pt file
    return x_state,x_target, y_obs,y_true,noises,delta_thetas,delta_thetas_observed


# #main
# time_steps = 30 #torch.randint(10,150,(1,)) ########### change back to 1001
# train_size = 1
# # dt = 0.1
# # t = np.arange(0, time_steps * dt, dt)
# to_plot = True

# R = 0.03
# v = 4
# Q = R * v
# Q_STD = np.sqrt(Q)
# R_STD = np.sqrt(R)
# omega = 1.5
# alpha = 1
# L1 = 100 # [cm] # [mm]
# L2 = 60 # [cm] # [mm]
# x_state,x_target, y_obs, y_true, noises, delta_thetas,delta_thetas_observed = simulate_robot_EKF_case_ut(train_size,time_steps,L1,L2,to_plot,Q_STD,R_STD, omega, alpha)



