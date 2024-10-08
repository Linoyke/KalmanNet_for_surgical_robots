"""# **Class: Extended Kalman Filter**
Theoretical Non Linear Kalman
"""
import torch
import datetime
from Simulations.RR_planar_robot.parameters import getJacobian
from Simulations.Constraints import make_state_valid 

class ExtendedKalmanFilter:

    def __init__(self, SystemModel, args):
        # Device
        self.args = args
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        # process model
        self.f = SystemModel.f
        self.m = SystemModel.m
        self.Q = SystemModel.Q.to(self.device)
        # observation model
        self.h = SystemModel.h
        self.n = SystemModel.n
        self.R = SystemModel.R.to(self.device)
        # sequence length (use maximum length if random length case)
        self.T = SystemModel.T
        self.T_test = SystemModel.T_test
        self.use_velocity = SystemModel.use_velocity
        self.is_m1x_0_true = SystemModel.is_m1x_0_true
        self.m1x_0_train = SystemModel.m1x_0_train
        
    def velocity_calc(self):
        if self.i == 0:
            self.velocity[:, :, self.i] = 1
        else:
            self.velocity[:, :, self.i] = (self.x[:, :, self.i] - self.x[:, :, self.i -1 ]) / 0.08
            
    # Predict
    def Predict(self):
        # Predict the 1-st moment of x
        self.m1x_prior = self.f(self.m1x_posterior).to(self.device)
        # Compute the Jacobians
        self.UpdateJacobians(getJacobian(self.m1x_posterior,self.f), getJacobian(self.m1x_prior, self.h,self.use_velocity,self.velocity,self.i))
        # Predict the 2-nd moment of x
        self.m2x_prior = torch.bmm(self.batched_F, self.m2x_posterior)
        self.m2x_prior = torch.bmm(self.m2x_prior, self.batched_F_T) + self.Q

        # Predict the 1-st moment of y
        self.m1y = self.h(self.m1x_prior)
        # Predict the 2-nd moment of y
        self.m2y = torch.bmm(self.batched_H, self.m2x_prior)
        self.m2y = torch.bmm(self.m2y, self.batched_H_T) + self.R

    # Compute the Kalman Gain
    def KGain(self):
        self.KG = torch.bmm(self.m2x_prior, self.batched_H_T)
        self.KG = torch.bmm(self.KG, torch.inverse(self.m2y))

        #Save KalmanGain
        self.KG_array[:,:,:,self.i] = self.KG
        self.i += 1

    # Innovation
    def Innovation(self, y):
        # y = y - self.m1x_prior[:,2:,:]
        self.dy = y - self.m1y

    # Compute Posterior
    def Correct(self):
        # Compute the 1-st posterior moment
        self.m1x_posterior = self.m1x_prior + torch.bmm(self.KG, self.dy)
        # adding constraint on the EKF's output to be within a desired range
        # if not in the range - takes the closest value within 
        # the problem's acceptable range
        if self.args.is_robot_case:
            self.m1x_posterior = make_state_valid(self.m1x_posterior)

        # Compute the 2-nd posterior moment
        self.m2x_posterior = torch.bmm(self.m2y, torch.transpose(self.KG, 1, 2))
        self.m2x_posterior = self.m2x_prior - torch.bmm(self.KG, self.m2x_posterior)
        
    def Update(self, y):
        if self.use_velocity:
            self.velocity_calc()
        self.Predict()
        self.KGain()
        self.Innovation(y)
        self.Correct()

        return self.m1x_posterior, self.m2x_posterior

    #########################

    def UpdateJacobians(self, F, H):

        self.batched_F = F.to(self.device)
        self.batched_F_T = torch.transpose(F,1,2)
        self.batched_H = H.to(self.device)
        self.batched_H_T = torch.transpose(H,1,2)
    
    def Init_batched_sequence(self, m1x_0_batch, m2x_0_batch):

            self.m1x_0_batch = m1x_0_batch # [batch_size, m, 1]
            self.m2x_0_batch = m2x_0_batch # [batch_size, m, m]
            # print("m1x_0 = ",self.m1x_0_batch)
            

    ######################
    ### Generate Batch ###
    ######################
    def GenerateBatch(self, y):
        """
        input y: batch of observations [batch_size, n, T]
        """
        y = y.to(self.device)
        self.batch_size = y.shape[0] # batch size
        T = y.shape[2] # sequence length (maximum length if randomLength=True)

        # Pre allocate KG array
        self.KG_array = torch.zeros([self.batch_size,self.m,self.n,T]).to(self.device)
        self.i = 0 # Index for KG_array alocation

        # Allocate Array for 1st and 2nd order moments (use zero padding)
        self.x = torch.zeros(self.batch_size, self.m, T).to(self.device)
        self.sigma = torch.zeros(self.batch_size, self.m, self.m, T).to(self.device)
        self.velocity = torch.zeros(self.batch_size, self.m, T).to(self.device)
        # Set 1st and 2nd order moments for t=0
        self.m1x_posterior = self.m1x_0_batch.to(self.device)
        self.m2x_posterior = self.m2x_0_batch.to(self.device)
        
        # Generate in a batched manner
        for t in range(0, T):
            yt = torch.unsqueeze(y[:, :, t],2)
            xt,sigmat = self.Update(yt)
            self.x[:, :, t] = torch.squeeze(xt,2)
            self.sigma[:, :, :, t] = sigmat
      

            