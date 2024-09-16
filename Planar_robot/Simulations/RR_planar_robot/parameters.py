
"""This file contains the parameters for the RR planar robot simulation.

"""


import torch
import math
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
from torch import autograd
import math

#########################
### Design Parameters ###
#########################
m = 4 # changed # 6
n = 2 # changed # 3
Q_structure = torch.eye(m)
R_structure = torch.eye(n)
###################################
L1 = 100 # robot first arm length
L2 = 60 # Robot second arm length
###################################
# initialization of m1x_0
#for L1 = 40 L2 = 25
theta1_rad = torch.tensor(math.radians(11))
theta2_rad = torch.tensor(math.radians(115))

x = L1 * torch.cos(theta1_rad) + L2 * torch.cos(theta1_rad + theta2_rad)
y = L1 * torch.sin(theta1_rad) + L2 * torch.sin(theta1_rad + theta2_rad)
m1x_0 = torch.tensor([[x], [y]])

### f will be fed to filters and KNet, note that the mismatch comes from delta_t
# f is a random walk meaning xt = xt-1 
def f(x, jacobian=False):
    jac = torch.tensor([[1.0, 0.0],
                  [0.0, 1.0]]).float()
    M = x.size()[0]
    tensor_list = [jac] * M
    jac_out = torch.stack(tensor_list, dim=0)
    if jacobian:
        return x, jac_out 
    return x


def h(x, jacobian=False):
    jac_eye = torch.eye(m)
    # Calculate second joint angle
    theta2 = torch.acos((x[:,0,:]**2 + x[:,1,:]**2 - L1**2 - L2**2) / (2 * L1 * L2)) + torch.deg2rad(x[:,3,:])
    # Calculate fisrt joint angle
    theta1 = torch.atan2(x[:,1,:], x[:,0,:]) - torch.atan2(L2 * torch.sin(theta2), L1 + L2 * torch.cos(theta2)) + torch.deg2rad(x[:,2,:])
    if jacobian:        
        jac = 0
    # concat theta1 & theta2 to y
    theta2 = theta2 * (180/ torch.pi)
    theta1= theta1 * (180/ torch.pi)
    y = torch.cat([theta1.unsqueeze(1), theta2.unsqueeze(1)], dim=1)
    if jacobian:
        return y, jac
    return y

def h_partial(x, jacobian=False):
    # val = torch.tanh(x)
    # tan_val = torch.atan2(x[:,1,:],x[:,0,:])
    # sin_val = torch.asin(x[:,1,:],torch.sqrt(x[:,1,:]**2+x[:,0,:]**2))
    # y = torch.cat([tan_val.unsqueeze(1), sin_val.unsqueeze(1)], dim=1)
    ### partial 1 ###
    # alpha = 0.2
    # multiplier = 1
    ### partial 2 ###
    alpha = 1
    multiplier = 2
    theta2 = torch.acos((x[:,0,:]**2 + x[:,1,:]**2 - L1**2 - L2**2) / (2 * L1 * L2))
    theta1 = multiplier*torch.atan2(x[:,1,:], x[:,0,:]) - torch.atan2(L2 * torch.sin(alpha*theta2), L1 + L2 * torch.cos(theta2))
    theta2 = theta2 * (180/ torch.pi)
    theta1= theta1 * (180/ torch.pi)
    y = torch.cat([theta1.unsqueeze(1), theta2.unsqueeze(1)], dim=1)
    return y

##################################
### Utils for non-linear cases ###
##################################
def getJacobian(x, g,use_v = False,v = None, i = None):
    """
    Currently, pytorch does not have a built-in function to compute Jacobian matrix
    in a batched manner, so we have to iterate over the batch dimension.
    
    input x (torch.tensor): [batch_size, m/n, 1]
    input g (function): function to be differentiated
    output Jac (torch.tensor): [batch_size, m, m] for f, [batch_size, n, m] for h
    """
    # Method 1: using autograd.functional.jacobian
    batch_size = x.shape[0]
    Jac_x0 = torch.squeeze(autograd.functional.jacobian(g, torch.unsqueeze(x[0,:,:],0)))
    Jac = torch.zeros([batch_size, Jac_x0.shape[0], Jac_x0.shape[1]])
    Jac[0,:,:] = Jac_x0
    for i in range(1,batch_size):
        Jac[i,:,:] = torch.squeeze(autograd.functional.jacobian(g, torch.unsqueeze(x[i,:,:],0)))
    return Jac
