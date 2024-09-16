## this is a constraint functions file to make sure the posterior state estimation
# and the proir observation estimation will be within the workspace of the problem
import torch
from Simulations.RR_planar_robot.parameters import L1, L2, m, n
import os
import datetime
    
def make_state_valid(state):
    new_state = state.clone()
    new_state = new_state.detach()
    theta = torch.atan2(new_state[:,1,0],new_state[:,0,0])
    r = new_state[:,:2,:]**2
    r = torch.sum(r , dim = 1)
    r = torch.sqrt(r)
    min_r = L1 - L2
    max_r = L1 + L2
    changed = False
    for i,r_val in enumerate(r):
        if min_r > r_val:
            x = (min_r+0.5)*torch.cos(theta[i])
            y = (min_r+0.5)*torch.sin(theta[i])
            new_state[i,0] = x.unsqueeze(0)
            new_state[i,1] = y.unsqueeze(0)
            changed = True
        elif max_r < r_val:
            x = (max_r-0.5)*torch.cos(theta[i])
            y = (max_r-0.5)*torch.sin(theta[i])
            new_state[i,0] = x.unsqueeze(0)
            new_state[i,1] = y.unsqueeze(0)
            changed = True
        else:
            new_state[i] = state[i]
    return new_state

def make_theta2_valid(theta2):
    if theta2 < 0:
        theta2 = 0
    elif theta2 > 180:
        theta2 = 180
    return theta2
        
    

