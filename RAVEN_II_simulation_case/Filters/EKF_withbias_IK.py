import torch
import torch.nn as nn
from Simulations.raven_ii_matlab.parameters_withbias_trainIK import getJacobian



class ExtendedKalmanFilterWithIK(nn.Module):
    def __init__(self, SystemModel, IKModel, args):
        super(ExtendedKalmanFilterWithIK, self).__init__()
        self.device = torch.device('cuda' if args.use_cuda else 'cpu')
        
        # Process model
        self.f = SystemModel.f
        self.m = SystemModel.m
        self.Q = SystemModel.Q.to(self.device)
        
        # Observation model is now the IKModel (a neural network)
        self.h_nn = IKModel.to(self.device)
        self.n = SystemModel.n
        self.R = SystemModel.R.to(self.device)
        
        self.T = SystemModel.T
        self.T_test = SystemModel.T_test

    def Predict(self):
        
    # Ensure m1x_posterior and m1x_prior have the correct shape: [batch_size, m, 1]
        self.m1x_prior = self.m1x_posterior  # Shape should remain [150, 6] or [150, 6, 1]


    # Compute Jacobians (Ensure these operations do not introduce extra dimensions)
        self.UpdateJacobians(getJacobian(self.m1x_posterior, self.f), getJacobian(self.m1x_prior[:,:3,:], self.h_nn))
    
    # Ensure the output shapes are correct after calling getJacobian
    # Shape should be [batch_size, m, m] after Jacobian computation
    # Predict the second moment of x
        self.m2x_prior = torch.bmm(self.batched_F, self.m2x_posterior)  # No additional dimensions should be added
        self.m2x_prior = torch.bmm(self.m2x_prior, self.batched_F_T) + self.Q
    
    # Predict the first moment of y (position)
        position = self.m1x_prior[:, :3, :].squeeze(-1)   # Take only the position part, shape [batch_size, 3]
        predicted_position = self.h_nn(position)  # Pass through IK network, shape [batch_size, 3]
    
    # Add bias to the prediction
        self.m1y = predicted_position.unsqueeze(-1) + self.m1x_prior[:, 3:]  # Add bias to prediction, shape [batch_size, 3]
    
    # Predict the second moment of y
        H = self.batched_H[:, :3, :]  # Observation Jacobian, shape [batch_size, 3, m]
        m2y_partial = torch.bmm(H, self.m2x_prior[:,:3])
        self.m2y = torch.bmm(m2y_partial[:,:,:3], H.transpose(1, 2)) + self.R


    def KGain(self):
    # Compute the Kalman Gain for observation dimensions
    # Extract the sub-matrix of self.m2x_prior that corresponds to the observed states (position only)
        m2x_prior_pos = self.m2x_prior[:, :3, :3]  

        H_partial = self.batched_H[:, :3, :]
        H_T_partial = H_partial.transpose(1, 2)

    # Perform batch matrix multiplication with the reduced prior covariance
        self.KG_partial = torch.bmm(m2x_prior_pos, H_T_partial)
        self.KG_partial = torch.bmm(self.KG_partial, torch.inverse(self.m2y))

    # Expand the Kalman Gain to match the state dimensions
        self.KG = torch.zeros(self.batch_size, self.m, self.n).to(self.device)

    # Assign the computed Kalman Gain for the observed states
        self.KG[:, :3, :] = self.KG_partial  # Only updating the part related to the observed states

    # Save Kalman Gain
        self.KG_array[:, :, :, self.i] = self.KG
        self.i = self.i+1


    def Innovation(self, y):
        self.dy = y - self.m1y  # Innovation step using the observation and predicted observation
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
            b1 = state[i, 3]
            b2 = state[i, 4]
            b3 = state[i, 5]
            x = torch.clamp(x, -0.8, 0.8)
            y = torch.clamp(y, -0.25, 0.25)
            z = torch.clamp(z, -0.2, 0.55)
            b1 = torch.clamp(b1, -0.5, 0.5)
            b2 = torch.clamp(b2, -0.5, 0.5)
            b3 = torch.clamp(b3, -0.5, 0.5)
            # position_vector = torch.tensor([x, y, z], device=state.device, dtype=state.dtype)
            # angle_with_axis = torch.acos(torch.dot(position_vector, cone_axis) / position_vector.norm())
            # if angle_with_axis > cone_angle:
            #     projection_length = torch.dot(position_vector, cone_axis)
            #     projected_point = projection_length * cone_axis / torch.cos(cone_angle)
            #     x, y, z = projected_point
            new_state[i, 0] = x
            new_state[i, 1] = y
            new_state[i, 2] = z
            new_state[i, 3] = b1
            new_state[i, 4] = b2
            new_state[i, 5] = b3
        return new_state

    def Correct(self):

        self.m1x_posterior = self.m1x_prior + torch.bmm(self.KG, self.dy)
        self.m1x_posterior = self.make_state_valid (self.m1x_posterior)
        # Compute the second posterior moment
        self.m2x_posterior = torch.bmm(self.m2y, torch.transpose(self.KG, 1, 2))
        self.m2x_posterior = self.m2x_prior - torch.bmm(self.KG, self.m2x_posterior)

    def Update(self, y):
        # EKF update step
        self.Predict()
        self.KGain()
        self.Innovation(y)
        self.Correct()

        return self.m1x_posterior, self.m2x_posterior

    def UpdateJacobians(self, F, H):
        # Update the Jacobians for EKF
        self.batched_F = F.to(self.device)
        self.batched_F_T = torch.transpose(F, 1, 2)
        self.batched_H = H.to(self.device)
        self.batched_H_T = torch.transpose(H, 1, 2)

    def Init_batched_sequence(self, m1x_0_batch, m2x_0_batch):
        # Initialize the sequences for batch processing
        self.m1x_0_batch = m1x_0_batch  # [batch_size, m, 1]
        self.m2x_0_batch = m2x_0_batch  # [batch_size, m, m]

    def GenerateBatch(self, y):
        """
        Generate a batch of data for the EKF process
        input y: batch of observations [batch_size, n, T]
        """
        y = y.to(self.device)
        self.batch_size = y.shape[0]  # Batch size
        T = y.shape[2]  # Sequence length

        # Pre-allocate Kalman Gain array
        self.KG_array = torch.zeros([self.batch_size, self.m, self.n, T]).to(self.device)
        self.i = 0  # Index for KG_array allocation

        # Allocate arrays for first and second order moments
        self.x = torch.zeros(self.batch_size, self.m, T).to(self.device)
        self.sigma = torch.zeros(self.batch_size, self.m, self.m, T).to(self.device)

        # Set initial moments for t=0
        self.m1x_posterior = self.m1x_0_batch.to(self.device)
        self.m2x_posterior = self.m2x_0_batch.to(self.device)
        # Generate in a batched manner
        for t in range(T):
            yt = torch.unsqueeze(y[:, :, t], 2)
            xt, sigmat = self.Update(yt)
            self.x[:, :, t] = torch.squeeze(xt, 2)
            self.sigma[:, :, :, t] = sigmat
