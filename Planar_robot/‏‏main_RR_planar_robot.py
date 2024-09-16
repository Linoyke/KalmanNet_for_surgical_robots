# based on main_lor_DT
import matplotlib.pyplot as plt
from Simulations.RR_planar_robot.RR_planar_simulation import simulate_robot,\
    forward_kinematics, is_theta1_valid, is_theta2_valid, to_excel
from Simulations.RR_planar_robot.parameters import m1x_0, m, n,\
    f, h, Q_structure, R_structure, L1, L2
from KNet.KalmanNet_nn import KalmanNetNN
from datetime import datetime
from Pipelines.Pipeline_EKF import Pipeline_EKF
import Simulations.config as config
from Simulations.Extended_sysmdl import SystemModel
from Filters.EKF_test import EKFTest
import torch.nn as nn
import torch
import pandas as pd
import os
import numpy as np
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732
from Create_Train_and_Validation_Sets import create_dataset

print("Pipeline Start")
################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

###################
###  Settings   ###
###################
args = config.general_settings()
# dataset parameters
args.N_T = 200  
args.T = 30  
args.T_test = 30  
# training parameters
args.use_cuda = True # use GPU or not
args.n_steps = 1500 
args.n_batch = 30 
args.lr = 1e-3
args.wd = 1e-2 
args.CompositionLoss = False
args.lr_change_flag = False
args.epoch_thresh_for_lr = 1250
args.is_robot_case = True

if args.use_cuda:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU")
    else:
        raise Exception("No GPU found, please set args.use_cuda = False")
else:
    device = torch.device('cpu')
    print("Using CPU")
use_velocity = False
switch = 'full'

# =============================================================================
# paths and files names
# =============================================================================
path_results = 'KNet/'
DatafolderName = 'Simulations/RR_planar_robot/data' + '/'
train_valid_excel_path = 'Simulations/RR_planar_robot/data/‏‏Simulations_instructions_for_train_and_validation_sets.xlsx'
dataFileName_dataset = ['DATASET_Train_and_validation_for_R=0.03___train_obs-train_true_state-cv_obs-cv_true_state_09_08_2024_135849.pt']
dataFileName_dataset = dataFileName_dataset[0]
fileName_test = ['test_file']
fileName_test = fileName_test[0]
to_load_existing_dataset = False
to_load_existing_test_file = False

    
# =============================================================================
# parameters for test set
# =============================================================================
# noise q and r for test
q = 12.64 # state evolution, make sure these are the same as in the excel training and validation file
r = 3.16 # measurement, make sure these are the same as in the excel training and validation file
ev_std =  torch.sqrt(torch.tensor(q)) # q_std
meas_std =  torch.sqrt(torch.tensor(r)) # r_std
r1 = torch.tensor([meas_std**2])
R = r1 * R_structure
q1 = torch.tensor([ev_std**2]) 
Q = q1[0] * Q_structure
m2x_0 = Q # adjust initial second moment for EKF 
omega = 2 # for bias function b_t = alpha(1-cos(omega*t))
alpha = 2 # for bias function b_t = alpha(1-cos(omega*t))
ev_std_np = np.sqrt(q)
meas_std_np = np.sqrt(r)
# =============================================================================
# m1x_0 true or not
# =============================================================================
# is the first guess correct
is_m1x_0_true = True
is_m1x_0_validation_true = True
is_m1x_0_test_true = True

# =============================================================================
# training parameters
# =============================================================================
unsupervised = False
minimize_with_bt = True
noises_train = None
noises_cv = None
to_use_noises = False
to_use_noises_for_MSE = True

# =============================================================================
# Data for training and evaluation 
# =============================================================================
if to_load_existing_dataset:
    [train_input,train_target,cv_input,cv_target] = torch.load(
        DatafolderName + dataFileName_dataset, map_location=device)
else:  # generate data set for training
    file_name = create_dataset(train_valid_excel_path,DatafolderName)
    [train_input,train_target,cv_input,cv_target] = torch.load(
        file_name, map_location=device)
if to_load_existing_test_file:
    dataFileName_test = "d_test_200_Q=0.12_R=0.03_omega=2_alpha=2_L1=100_L2=60_with_noises_and_dtheta_noise=1-cos(wt)_01_08_2024_11_02_07.pt" # cahnge to any test file from data folder
    if to_use_noises_for_MSE:
        [test_input, test_target, test_angles_true, test_state_by_obs,test_noises, test_delta_theta, test_delta_theta_obs] = torch.load(
        DatafolderName + dataFileName_test, map_location=device)
    else:
        [test_input, test_target, test_angles_true, test_state_by_obs] = torch.load(
        DatafolderName + dataFileName_test, map_location=device)
else:
    [test_state,test_true_state,test_obs,test_angles_true,noises_test,test_delta_theta,test_delta_theta_obs] = simulate_robot(args.N_T,args.T,L1,L2,False,ev_std_np,meas_std_np,omega,alpha)
    torch.save([test_obs, test_true_state,test_angles_true,test_state,noises_test,test_delta_theta,test_delta_theta_obs], DatafolderName + fileName_test)
    if to_use_noises_for_MSE:
        [test_input, test_target, test_angles_true, test_state_by_obs,test_noises, test_delta_theta, test_delta_theta_obs] = torch.load(
            DatafolderName + fileName_test, map_location=device)
    else:
        [test_input, test_target, test_angles_true, test_state_by_obs] = torch.load(
            DatafolderName + fileName_test, map_location=device)

##### state with bias case adjusments ############
if to_use_noises_for_MSE:
    train_target_without_bias = train_target 
    train_bais_tensor = torch.zeros_like(train_target)
    train_target = torch.cat((train_target, train_bais_tensor), dim=1)
    # cv 
    cv_target_without_bias = cv_target 
    cv_bais_tensor = torch.zeros_like(cv_target)
    cv_target = torch.cat((cv_target, cv_bais_tensor), dim=1)
    # test
    test_target_without_bias = test_target 
    test_bais_tensor = torch.zeros_like(test_target)
    test_target = torch.cat((test_target, test_noises), dim=1)

else:
    # train
    train_target_without_bias = train_target 
    train_bais_tensor = torch.zeros_like(train_target)
    train_target = torch.cat((train_target, train_bais_tensor), dim=1)
    # cv 
    cv_target_without_bias = cv_target 
    cv_bais_tensor = torch.zeros_like(cv_target)
    cv_target = torch.cat((cv_target, cv_bais_tensor), dim=1)
    # test
    test_target_without_bias = test_target 
    test_bais_tensor = torch.zeros_like(test_target)
    test_target = torch.cat((test_target, test_bais_tensor), dim=1)

print("trainset size:", train_target.size())
print("cvset size:", cv_target.size())
print("testset size:", test_target.size())

args.N_E = train_target.size(0)
args.N_CV = cv_target.size(0) 
args.N_T = test_target.size(0) 

##########################################################
### initializing first moment for train valid and test ###
##########################################################

m1x_0_train = train_target[:, :, 0].unsqueeze(-1)
m1x_0_validation = cv_target[:, :, 0].unsqueeze(-1)
m1x_0_test = test_target[:, :, 0].unsqueeze(-1)
sys_model = SystemModel(f, Q, h, R, args.T, args.T_test, m, n, use_velocity, is_m1x_0_true, m1x_0_train,
                        is_m1x_0_validation_true, m1x_0_validation, is_m1x_0_test_true, m1x_0_test, noises_train, noises_cv, to_use_noises)  # parameters for GT
sys_model.InitSequence(m1x_0, m2x_0)  # x0 and P0

########################
### Evaluate Filters ###
########################
# Evaluate EKF true
print("Evaluate EKF true")
if to_use_noises_for_MSE:
    [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array,
    EKF_out,MSE_EKF_linear_avg_noise] = EKFTest(args, sys_model, test_input, test_target,to_use_noises_for_MSE = to_use_noises_for_MSE)
else:
    [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array,EKF_out] = EKFTest(args, sys_model, test_input, test_target)

# =============================================================================
# Observations
# =============================================================================
loss_fn = nn.MSELoss(reduction='mean')
MSE_observation = torch.zeros(args.N_T)
for j in range(args.N_T):  
    MSE_observation[j] = loss_fn(test_state_by_obs[j, :, 0:args.T], test_target[j, :2, 0:args.T]).item()
MSE_obs_linear_avg = torch.mean(MSE_observation)
print("obs linear mse: ", MSE_obs_linear_avg)
print("EKF true linear mse: ", MSE_EKF_linear_avg)

# =============================================================================
#  Kalman Net
# =============================================================================
if switch == 'full':
    ## KNet with full info ####################################################################################
    ################
    ## KNet full ###
    ################
    # Build Neural Network
    print("KNet with full model info")
    KNet_model = KalmanNetNN()
    KNet_model.NNBuild(sys_model, args)
    # ## Train Neural Network
    KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KNet")
    KNet_Pipeline.setssModel(sys_model)
    KNet_Pipeline.setModel(KNet_model)
    print("Number of trainable parameters for KNet:", sum(p.numel()
          for p in KNet_model.parameters() if p.requires_grad))
    KNet_Pipeline.setTrainingParams(args,unsupervised = unsupervised, minimize_with_ut = minimize_with_bt)
    [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results)
    # Test Neural Network
    if to_use_noises_for_MSE:
        [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, Knet_out,
        RunTime,MSE_test_linear_arr_noise, MSE_test_linear_avg_noise, MSE_test_dB_avg_noise] = KNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results, to_use_noises_for_MSE = to_use_noises_for_MSE)
    else:
        [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, Knet_out,
        RunTime] = KNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results)
        
    print("======================================")
    print(" ===== Summary ====== ")
    print("linear MSE on position based on observations: ", MSE_obs_linear_avg)
    print("linear MSE position based on EKF: ", MSE_EKF_linear_avg)
    print("linear MSE position based on KalmanNet: ", MSE_test_linear_avg)
    print("======================================")

####################################################################################