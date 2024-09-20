# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 17:37:46 2024

@author: shach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class KalmanNetNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.best_loss = float('inf')
    
    def NNBuild(self, SysModel, args):
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.SysModel_h = SysModel.h  # Store the h function from SysModel
        self.InitSystemDynamics(SysModel.f, SysModel.m, SysModel.n)
        self.InitKGainNet(SysModel.prior_Q, SysModel.prior_Sigma, SysModel.prior_S, args)

    def InitSystemDynamics(self, f, m, n):
        self.f = f
        self.m = m
        self.n = n

    def InitKGainNet(self, prior_Q, prior_Sigma, prior_S, args):
        self.seq_len_input = 1
        self.batch_size = args.n_batch
        self.prior_Q = prior_Q.to(self.device)
        self.prior_Sigma = prior_Sigma.to(self.device)
        self.prior_S = prior_S.to(self.device)
        self.d_input_Q = self.m * args.in_mult_KNet
        self.d_hidden_Q = self.m ** 2
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q).to(self.device)
        self.d_input_Sigma = self.d_hidden_Q + self.m * args.in_mult_KNet
        self.d_hidden_Sigma = self.m ** 2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma).to(self.device)
        self.d_input_S = self.n ** 2 + 2 * self.n * args.in_mult_KNet
        self.d_hidden_S = self.n ** 2
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S).to(self.device)
        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.n ** 2
        self.FC1 = nn.Sequential(
                nn.Linear(self.d_input_FC1, self.d_output_FC1),
                nn.ReLU()).to(self.device)
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = self.n * self.m
        self.d_hidden_FC2 = self.d_input_FC2 * args.out_mult_KNet
        self.FC2 = nn.Sequential(
                nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
                nn.ReLU(),
                nn.Linear(self.d_hidden_FC2, self.d_output_FC2)).to(self.device)
        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = self.m ** 2
        self.FC3 = nn.Sequential(
                nn.Linear(self.d_input_FC3, self.d_output_FC3),
                nn.ReLU()).to(self.device)
        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(
                nn.Linear(self.d_input_FC4, self.d_output_FC4),
                nn.ReLU()).to(self.device)
        self.d_input_FC5 = self.m
        self.d_output_FC5 = self.m * args.in_mult_KNet
        self.FC5 = nn.Sequential(
                nn.Linear(self.d_input_FC5, self.d_output_FC5),
                nn.ReLU()).to(self.device)
        self.d_input_FC6 = self.m
        self.d_output_FC6 = self.m * args.in_mult_KNet
        self.FC6 = nn.Sequential(
                nn.Linear(self.d_input_FC6, self.d_output_FC6),
                nn.ReLU()).to(self.device)
        self.d_input_FC7 = 2 * self.n
        self.d_output_FC7 = 2 * self.n * args.in_mult_KNet
        self.FC7 = nn.Sequential(
                nn.Linear(self.d_input_FC7, self.d_output_FC7),
                nn.ReLU()).to(self.device)

    def InitSequence(self, M1_0, T):
        self.T = T
        self.m1x_posterior = M1_0.to(self.device)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_prior_previous = self.m1x_posterior

        self.y_previous = self.SysModel_h(self.m1x_posterior[:, :3].reshape(-1, 3)).reshape(self.m1x_posterior.size(0), 3, 1)

        self.first_step = True  # Add this line to initialize the flag

    def step_prior(self):
        self.m1x_prior = (self.m1x_posterior)
        self.m1y_nn = self.SysModel_h(self.m1x_prior[:, :3].reshape(-1, 3)).reshape(self.m1x_prior.size(0), 3, 1)

        self.m1y = self.SysModel_h(self.m1x_prior)  # Use the original h function from SysModel

    def step_KGain_est(self, y):
        obs_diff_nn = y - self.y_previous.squeeze(-1)
        obs_innov_diff_nn = y - self.m1y_nn.squeeze(-1)
        fw_evol_diff = self.m1x_posterior - self.m1x_posterior_previous
        fw_update_diff = self.m1x_posterior - self.m1x_prior_previous
        obs_diff_nn = F.normalize(obs_diff_nn, p=2, dim=1, eps=1e-12, out=None)
        obs_innov_diff_nn = F.normalize(obs_innov_diff_nn, p=2, dim=1, eps=1e-12, out=None)
        fw_evol_diff = F.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12, out=None)
        fw_update_diff = F.normalize(fw_update_diff, p=2, dim=1, eps=1e-12, out=None)
        KG_nn = self.KGain_step(obs_diff_nn, obs_innov_diff_nn, fw_evol_diff, fw_update_diff)
        self.KGain_nn = torch.reshape(KG_nn, (self.batch_size, self.m, self.n))

        obs_diff = y - self.y_previous.squeeze(-1)

        obs_innov_diff = y - self.m1y[:,:3].squeeze(-1)
        obs_diff = F.normalize(obs_diff, p=2, dim=1, eps=1e-12, out=None)
        obs_innov_diff = F.normalize(obs_innov_diff, p=2, dim=1, eps=1e-12, out=None)
        KG = self.KGain_step(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)
        self.KGain = torch.reshape(KG, (self.batch_size, self.m, self.n))

    def make_state_valid(self, state, epsilon=0.01, cone_axis=None, cone_angle=None):
        if cone_axis is None:
            cone_axis = torch.tensor([1.0, 0.0, 0.0], device=state.device, dtype=state.dtype, requires_grad=False)
        if cone_angle is None:
            cone_angle = torch.tensor(30.0, device=state.device, dtype=state.dtype, requires_grad=False) * torch.pi / 180.0

        new_state = state.clone()
        for i in range(state.shape[0]):
            x = state[i, 0]
            y = state[i, 1]
            z = state[i, 2]
            x = torch.clamp(x, 0.8, 1.2)
            y = torch.clamp(y, 0.1, 25.1)
            z = torch.clamp(z, -0.45, -0.45)
            position_vector = torch.tensor([x, y, z], device=state.device, dtype=state.dtype)
            angle_with_axis = torch.acos(torch.dot(position_vector, cone_axis) / position_vector.norm())
            if angle_with_axis > cone_angle:
                projection_length = torch.dot(position_vector, cone_axis)
                projected_point = projection_length * cone_axis / torch.cos(cone_angle)
                x, y, z = projected_point
            new_state[i, 0] = x
            new_state[i, 1] = y
            new_state[i, 2] = z
            new_state[i, 3] = state[i, 3]
            new_state[i, 4] = state[i, 4]
            new_state[i, 5] = state[i, 5]
        return new_state

    def KNet_step(self, y):
        y = y.requires_grad_(True).to(self.device)
        ## Subtracting Bias ##
        self.step_prior()

        y = y + self.m1x_prior[:, 3:, :].squeeze(-1)
        
        if self.first_step:
            # Skip updating the state for the first step
            self.first_step = False
            self.y_previous = y
            return self.m1x_posterior
        self.step_KGain_est(y)
    
        # Ensure y and self.m1y_nn have compatible dimensions
        if y.dim() == 2:
            y = y.unsqueeze(-1)
            if self.m1y_nn.dim() == 2:
                self.m1y_nn = self.m1y_nn.unsqueeze(-1)
    
        dy_nn = y - self.m1y_nn
        dy = y - self.m1y[:, :3]
        INOV_nn = torch.bmm(self.KGain_nn, dy_nn)
        INOV = torch.bmm(self.KGain, dy)
        self.m1x_posterior_previous = self.m1x_posterior
        KNet_out_m1x_posterior_nn = self.m1x_prior + INOV_nn
        KNet_out_m1x_posterior = self.m1x_prior + INOV
        self.m1x_posterior_nn = self.make_state_valid(KNet_out_m1x_posterior_nn)
        self.m1x_posterior = self.make_state_valid(KNet_out_m1x_posterior)
        self.m1x_prior_previous = self.m1x_prior
        self.y_previous = y
        return self.m1x_posterior  # Return only the tensor you need

    def KGain_step(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff):
        # Ensure the input tensors are of the correct shape
        obs_diff = obs_diff.view(self.batch_size, -1)
        obs_innov_diff = obs_innov_diff.view(self.batch_size, -1)
        fw_evol_diff = fw_evol_diff.view(self.batch_size, -1)
        fw_update_diff = fw_update_diff.view(self.batch_size, -1)
    
        # Expand the tensors to match the expected input size for the GRU
        obs_diff_expanded = obs_diff.unsqueeze(0).expand(self.seq_len_input, self.batch_size, -1)
        obs_innov_diff_expanded = obs_innov_diff.unsqueeze(0).expand(self.seq_len_input, self.batch_size, -1)
        fw_evol_diff_expanded = fw_evol_diff.unsqueeze(0).expand(self.seq_len_input, self.batch_size, -1)
        fw_update_diff_expanded = fw_update_diff.unsqueeze(0).expand(self.seq_len_input, self.batch_size, -1)
    
        in_FC5 = fw_update_diff_expanded
        out_FC5 = self.FC5(in_FC5)
        in_Q = out_FC5
        out_Q, self.h_Q = self.GRU_Q(in_Q, self.h_Q)
    
        in_FC6 = fw_evol_diff_expanded
        out_FC6 = self.FC6(in_FC6)
        in_Sigma = torch.cat((out_Q, out_FC6), 2)
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)
    
        in_FC1 = out_Sigma
        out_FC1 = self.FC1(in_FC1)
    
        in_FC7 = torch.cat((obs_diff_expanded, obs_innov_diff_expanded), 2)
        out_FC7 = self.FC7(in_FC7)
    
        in_S = torch.cat((out_FC1, out_FC7), 2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)
    
        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC2 = self.FC2(in_FC2)
    
        in_FC3 = torch.cat((out_S, out_FC2), 2)
        out_FC3 = self.FC3(in_FC3)
    
        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4(in_FC4)
    
        self.h_Sigma = out_FC4
        return out_FC2

    def forward(self, y):
        y = y.to(self.device)
        return self.KNet_step(y)

    def init_hidden_KNet(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_S).zero_()
        self.h_S = hidden.data
        self.h_S = self.prior_S.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1)
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Sigma).zero_()
        self.h_Sigma = hidden.data
        self.h_Sigma = self.prior_Sigma.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1)
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Q).zero_()
        self.h_Q = hidden.data
        self.h_Q = self.prior_Q.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1)
    
# Example of a training loop controlling each network independently
def train_kalman_net_nn(kalman_net_nn, data_loader, optimizer):
    kalman_net_nn.train()
    total_loss = 0
    loss_fn = nn.MSELoss()
    for inputs, targets in data_loader:
        optimizer.zero_grad()
        outputs = kalman_net_nn(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)
