import torch
torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection
import torch.nn as nn
import time
from Filters.EKF_withbias_IK import ExtendedKalmanFilterWithIK

"""def EKFTest(args, SysModel, IKModel, test_input, test_target, optimizer, allStates=True, randomInit=False, test_init=None, test_lengthMask=None):
    N_T = test_target.size()[0]
    loss_fn = nn.MSELoss(reduction='mean')
    MSE_EKF_linear_arr = torch.zeros(N_T)
    EKF_out = torch.zeros([N_T, SysModel.m, test_input.size()[2]])
    KG_array = torch.zeros([N_T, SysModel.m, SysModel.n, test_input.size()[2]])

    start = time.time()

    # Instantiate the EKF with the IK neural network as the observation model
    EKF = ExtendedKalmanFilterWithIK(SysModel, IKModel, args)

    # Initialize EKF


    if randomInit:
        EKF.Init_batched_sequence(test_init, SysModel.m2x_0.view(1, SysModel.m, SysModel.m).expand(N_T, -1, -1))
    else:
        # Check if the shape is [150, 6] (which is already batch size x state dimension)
        if SysModel.m1x_0.shape == (N_T, SysModel.m):
            # Reshape m1x_0 to [N_T, m, 1] without expanding
            EKF.Init_batched_sequence(
                SysModel.m1x_0.unsqueeze(-1),  # Add a new dimension to get [N_T, m, 1]
                SysModel.m2x_0.view(1, SysModel.m, SysModel.m).expand(N_T, -1, -1)  # Expand m2x_0 across batch size
            )
        else:
            # Handle other cases (e.g., unexpected shape)
            EKF.Init_batched_sequence(
                SysModel.m1x_0.view(1, SysModel.m, 1).expand(N_T, -1, -1),  # Correctly reshape and expand to [N_T, m, 1]
                SysModel.m2x_0.view(1, SysModel.m, SysModel.m).expand(N_T, -1, -1)  # Expand across batch size
            )

    EKF.GenerateBatch(test_input)
    end = time.time()
    t = end - start

    KG_array = EKF.KG_array
    EKF_out = EKF.x

    for j in range(N_T):
    # Print current iteration
        print(f"Iteration {j + 1}/{N_T}")

        if allStates:
            if args.randomLength:
                loss = loss_fn(EKF.x[j, :, test_lengthMask[j]], test_target[j, :, test_lengthMask[j]])
            else:
                loss = loss_fn(EKF.x[j, :, :], test_target[j, :, :])
        else:
            loc = torch.tensor([True, False, False])  # Adjust based on which states to consider (e.g., position only)
            if args.randomLength:
                loss = loss_fn(EKF.x[j, loc, test_lengthMask[j]], test_target[j, loc, test_lengthMask[j]])
            else:
                loss = loss_fn(EKF.x[j, loc, :], test_target[j, loc, :])

    # Print the current loss
        print(f"Loss at iteration {j + 1}: {loss.item()}")


    # Backpropagation and parameter update
        optimizer.zero_grad()
        loss.backward(retain_graph=True)  # Backpropagate the loss on the posterior
        optimizer.step()


        MSE_EKF_linear_arr[j] = loss.item()

    MSE_EKF_linear_avg = torch.mean(MSE_EKF_linear_arr)
    MSE_EKF_dB_avg = 10 * torch.log10(MSE_EKF_linear_avg)

    print("EKF with IK (Posterior Loss) - MSE LOSS:", MSE_EKF_dB_avg, "[dB]")
    print("Inference Time:", t)

    return [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, KG_array, EKF_out]
"""
def EKFTest(args, SysModel, IKModel, test_input, test_target, optimizer, allStates=True, randomInit=False, test_init=None, test_lengthMask=None):
    N_T = test_target.size()[0]
    loss_fn = nn.MSELoss(reduction='mean')
    MSE_EKF_linear_arr = torch.zeros(N_T)
    EKF_out = torch.zeros([N_T, SysModel.m, test_input.size()[2]])
    KG_array = torch.zeros([N_T, SysModel.m, SysModel.n, test_input.size()[2]])

    start = time.time()

    # Instantiate the EKF with the IK neural network as the observation model
    EKF = ExtendedKalmanFilterWithIK(SysModel, IKModel, args)

    # Initialize EKF
    if randomInit:
        EKF.Init_batched_sequence(test_init, SysModel.m2x_0.view(1, SysModel.m, SysModel.m).expand(N_T, -1, -1))
    else:
        # Check if the shape is [150, 6] (which is already batch size x state dimension)
        if SysModel.m1x_0.shape == (N_T, SysModel.m):
            EKF.Init_batched_sequence(
                SysModel.m1x_0.unsqueeze(-1),  # Add a new dimension to get [N_T, m, 1]
                SysModel.m2x_0.view(1, SysModel.m, SysModel.m).expand(N_T, -1, -1)  # Expand m2x_0 across batch size
            )
        else:
            EKF.Init_batched_sequence(
                SysModel.m1x_0.view(1, SysModel.m, 1).expand(N_T, -1, -1),  # Correctly reshape and expand to [N_T, m, 1]
                SysModel.m2x_0.view(1, SysModel.m, SysModel.m).expand(N_T, -1, -1)  # Expand across batch size
            )

    EKF.GenerateBatch(test_input)
    end = time.time()
    t = end - start

    KG_array = EKF.KG_array
    EKF_out = EKF.x
    

    # Compute loss for the entire batch
    if allStates:
        if args.randomLength:
            loss = loss_fn(EKF.x[:, :, test_lengthMask], test_target[:, :, test_lengthMask])
        else:
            loss = loss_fn(EKF.x, test_target)
    else:
        loc = torch.tensor([True, False, False])  # Adjust based on which states to consider (e.g., position only)
        if args.randomLength:
            loss = loss_fn(EKF.x[:, loc, test_lengthMask], test_target[:, loc, test_lengthMask])
        else:
            loss = loss_fn(EKF.x[:, loc, :], test_target[:, loc, :])

    # Print the current loss
    print(f'loss of iteration {loss}')

    # Backpropagation and parameter update
    optimizer.zero_grad()
    loss.backward(retain_graph=True)  # Backpropagate the loss on the posterior
    optimizer.step()

    MSE_EKF_linear_avg = loss.item()
    MSE_EKF_dB_avg = 10 * torch.log10(torch.tensor(MSE_EKF_linear_avg))



    return [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, KG_array, EKF_out]
