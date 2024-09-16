import pandas as pd
import ast
import torch
from Simulations.RR_planar_robot.RR_planar_simulation import simulate_robot
import time
from datetime import datetime

# =============================================================================
# Create train and validation sets according to excel file
# =============================================================================
def create_dataset(file_path,DatafolderName):
    start_time = time.time()
    
    train_obs_list = [] 
    train_true_state_list =[]
    valid_obs_list = [] 
    valid_true_state_list =[]
    
    # Get the current date and time
    now = datetime.now()
    timestamp = now.strftime("%d_%m_%Y_%H%M%S")
    
    # data file name of saved train and validation 
    dataFileName = f'DATASET____train_obs-train_true_state-cv_obs-cv_true_state_{timestamp}.pt'
    dataFileName = [dataFileName]
    fileName = DatafolderName + dataFileName[0]
    
    sheet_name = 'Instructions'
    
    # Read the sheet into a DataFrame
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # Function to convert string representation of list to a list
    def str_to_list(s):
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return s
    
    # Skip the first row (header) and iterate through each row starting from the second
    simulation_cases = []
    
    for index, row in df.iterrows():
        #if index == 0:
            #  # Skip the first row
    
        # Access and process values in each column
        simulation_case = {
            'L1': row['L1'],
            'L2': row['L2'],
            'Q_STD': row['Q_STD'],
            'R_STD': row['R_STD'],
            'Q': row['Q'],
            'R': row['R'],
            'v': row['v'],
            'TRAIN': row['TRAIN'],
            'VALIDATION': row['VALIDATION'],
            'ALPHA_TRAIN': str_to_list(row['ALPHA_TRAIN']),
            'OMEGA_TRAIN': str_to_list(row['OMEGA_TRAIN']),
            'ALPHA_VALID': str_to_list(row['ALPHA_VALID']),
            'OMEGA_VALID': str_to_list(row['OMEGA_VALID']),
            'TIME_STEPS': row['TIME_STEPS']
        }
        
        # Saving the omega and alpha cases
        alpha_values_train = str_to_list(row['ALPHA_TRAIN'])
        omega_values_train = str_to_list(row['OMEGA_TRAIN'])
        alpha_values_valid = str_to_list(row['ALPHA_VALID'])
        omega_values_valid = str_to_list(row['OMEGA_VALID'])
        print(f"alpha_values_train: {alpha_values_train}, type: {type(alpha_values_train)}")
        print(f"omega_values_train: {omega_values_train}, type: {type(omega_values_train)}")
        print(f"alpha_values_valid: {alpha_values_valid}, type: {type(alpha_values_valid)}")
        print(f"omega_values_valid: {omega_values_valid}, type: {type(omega_values_valid)}")
        
        # running over all required alpha and omega and sending for simulations
        for alpha in alpha_values_train:
            for omega in omega_values_train:
                print("######### simulation case ###########")
                [train_state,train_true_state,train_obs,train_angles_true,noises_train,train_delta_theta,train_delta_theta_obs] = simulate_robot(int(row['TRAIN']),row['TIME_STEPS'],row['L1'],row['L2'],False,row['Q_STD'],row['R_STD'],omega,alpha)
                train_obs_list.append(train_obs)
                train_true_state_list.append(train_true_state)
                
        # same for validation set
        for alpha in alpha_values_valid:
            for omega in omega_values_valid:
                [cv_state,cv_true_state,cv_obs,cv_angles_true,noises_valid,cv_delta_theta,cv_delta_theta_obs] = simulate_robot(int(row['VALIDATION']),row['TIME_STEPS'],row['L1'],row['L2'],False,row['Q_STD'],row['R_STD'],omega,alpha)
                valid_obs_list.append(cv_obs)
                valid_true_state_list.append(cv_true_state)
                
    # Concatenate the list of tensors along a new dimension
    train_obs_tensor = torch.cat(train_obs_list, dim=0)
    train_true_state_tensor = torch.cat(train_true_state_list, dim=0)
            
    # Concatenate the list of tensors along a new dimension
    cv_obs_tensor = torch.cat(valid_obs_list, dim=0)
    cv_true_state_tensor = torch.cat(valid_true_state_list, dim=0)
            
    simulation_cases.append(simulation_case)
    
    ## saving the train and validation sets tensors
    torch.save([train_obs_tensor,train_true_state_tensor ,cv_obs_tensor,cv_true_state_tensor], fileName)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time} seconds")
    return fileName

