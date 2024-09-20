"""This file contains the parameters for the Lorenz Atractor simulation.

Update 2023-02-06: f and h support batch size speed up

"""





# """This file contains the parameters for the Lorenz Atractor simulation.

# Update 2023-02-06: f and h support batch size speed up

# """


import torch
import math
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
from torch import autograd
import numpy as np
#from ikpy.chain import Chain
#from ikpy.link import OriginLink, URDFLink
from scipy.spatial.transform import Rotation as R
import torch.nn as nn

#########################
### Design Parameters ###
#########################
m = 6
n = 3
variance = 0
m1x_0 = torch.ones(m, 1)
m2x_0 = 0 * 0 * torch.eye(m)

### Decimation
delta_t_gen = 1e-5
delta_t = 0.02
ratio = delta_t_gen / delta_t

### Taylor expansion order
J = 5
J_mod = 2

### Angle of rotation in the 3 axes
roll_deg = yaw_deg = pitch_deg = 1

roll = roll_deg * (math.pi / 180)
yaw = yaw_deg * (math.pi / 180)
pitch = pitch_deg * (math.pi / 180)

RX = torch.tensor([
    [1, 0, 0],
    [0, math.cos(roll), -math.sin(roll)],
    [0, math.sin(roll), math.cos(roll)]])
RY = torch.tensor([
    [math.cos(pitch), 0, math.sin(pitch)],
    [0, 1, 0],
    [-math.sin(pitch), 0, math.cos(pitch)]])
RZ = torch.tensor([
    [math.cos(yaw), -math.sin(yaw), 0],
    [math.sin(yaw), math.cos(yaw), 0],
    [0, 0, 1]])

RotMatrix = torch.mm(torch.mm(RZ, RY), RX)

### Auxiliar MultiDimensional Tensor B and C (they make A --> Differential equation matrix)
C = torch.tensor([
    [-10, 10, 0, 0, 0, 0],
    [28, -1, 0, 0, 0, 0],
    [0, 0, -8 / 3, 0, 0, 0],
    [0, 0, 0, -10, 10, 0],
    [0, 0, 0, 28, -1, 0],
    [0, 0, 0, 0, 0, -8 / 3]
]).float()

######################################################
### State evolution function f for Lorenz Atractor ###
######################################################
### f_gen is for dataset generation
def f_gen(x, jacobian=False):
    BX = torch.zeros([x.shape[0], m, m]).float().to(x.device)  # [batch_size, m, m]
    BX[:, 1, 0] = torch.squeeze(-x[:, 2, :])
    BX[:, 2, 0] = torch.squeeze(x[:, 1, :])
    BX[:, 4, 3] = torch.squeeze(-x[:, 5, :])
    BX[:, 5, 3] = torch.squeeze(x[:, 4, :])
    Const = C.to(x.device)
    A = torch.add(BX, Const)
    # Taylor Expansion for F
    F = torch.eye(m).to(x.device)
    F = F.reshape((1, m, m)).repeat(x.shape[0], 1, 1)  # [batch_size, m, m] identity matrix
    for j in range(1, J + 1):
        F_add = (torch.matrix_power(A * delta_t_gen, j) / math.factorial(j))
        F = torch.add(F, F_add)
    if jacobian:
        return torch.bmm(F, x), F
    else:
        return torch.bmm(F, x)

### f will be fed to filters and KNet, note that the mismatch comes from delta_t
def f(x, jacobian=False):
    jac = torch.eye(m).repeat(x.shape[0], 1, 1).to(x.device)
    if jacobian:
        return x, jac
    return x

### fInacc will be fed to filters and KNet, note that the mismatch comes from delta_t and J_mod
def fInacc(x, jacobian=False):
    BX = torch.zeros([x.shape[0], m, m]).float().to(x.device)  # [batch_size, m, m]
    BX[:, 1, 0] = torch.squeeze(-x[:, 2, :])
    BX[:, 2, 0] = torch.squeeze(x[:, 1, :])
    BX[:, 4, 3] = torch.squeeze(-x[:, 5, :])
    BX[:, 5, 3] = torch.squeeze(x[:, 4, :])
    Const = C.to(x.device)
    A = torch.add(BX, Const)
    # Taylor Expansion for F
    F = torch.eye(m).to(x.device)
    F = F.reshape((1, m, m)).repeat(x.shape[0], 1, 1)  # [batch_size, m, m] identity matrix
    for j in range(1, J_mod + 1):
        F_add = (torch.matrix_power(A * delta_t, j) / math.factorial(j))
        F = torch.add(F, F_add)
    if jacobian:
        return torch.bmm(F, x), F
    else:
        return torch.bmm(F, x)

### fInacc will be fed to filters and KNet, note that the mismatch comes from delta_t and rotation
def fRotate(x, jacobian=False):
    BX = torch.zeros([x.shape[0], m, m]).float().to(x.device)  # [batch_size, m, m]
    BX[:, 1, 0] = torch.squeeze(-x[:, 2, :])
    BX[:, 2, 0] = torch.squeeze(x[:, 1, :])
    BX[:, 4, 3] = torch.squeeze(-x[:, 5, :])
    BX[:, 5, 3] = torch.squeeze(x[:, 4, :])
    Const = C.to(x.device)
    A = torch.add(BX, Const)
    # Taylor Expansion for F
    F = torch.eye(m).to(x.device)
    F = F.reshape((1, m, m)).repeat(x.shape[0], 1, 1)  # [batch_size, m, m] identity matrix
    for j in range(1, J + 1):
        F_add = (torch.matrix_power(A * delta_t, j) / math.factorial(j))
        F = torch.add(F, F_add)
    F_rotated = torch.bmm(RotMatrix.reshape(1, m, m).repeat(x.shape[0], 1, 1), F)
    if jacobian:
        return torch.bmm(F_rotated, x), F_rotated
    else:
        return torch.bmm(F_rotated, x)

##################################################
### Observation function h for using nn for h ###
##################################################
# Define the function to load the model and use it for prediction
class InverseKinematicsNN(nn.Module):
    def __init__(self):
        super(InverseKinematicsNN, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def load_model():
    model = InverseKinematicsNN()
    model.load_state_dict(torch.load('Simulations/Raven_ii_matlab/H_Estimation/inverse_kinematics_nn_comp.pth'))
    model.eval()  # Set the model to evaluation mode
    return model

# Load the model once
model = load_model()

# Define the new h function using the neural network model
def h_nn(x, jacobian=False):
    # Extract the first three components
    x_first3 = x[:, :3, :]
    x_last3 = x[:, 3:, :]

    # Reshape input to match model's expected input shape
    input_tensor = x_first3.view(-1, 3)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    output_tensor = output_tensor.view(-1, 3, 1)  # Reshape output to match the expected shape

    # Concatenate the output of the NN with the unchanged last three components
    result = torch.cat((output_tensor, x_last3), dim=1)

    if jacobian:
        # Compute Jacobian for the first three components, assume identity for the last three
        batch_size = x.shape[0]
        jac_nn = torch.eye(3).repeat(batch_size, 1, 1).to(x.device)
        jac_identity = torch.eye(3).repeat(batch_size, 1, 1).to(x.device)
        jac = torch.cat((jac_nn, torch.zeros(batch_size, 3, 3).to(x.device)), dim=2)
        jac = torch.cat((jac, torch.cat((torch.zeros(batch_size, 3, 3).to(x.device), jac_identity), dim=2)), dim=1)
        return result, jac

    return result

# Replace the original h function with the new h_nn function
h = h_nn

def h_nonlinear(x):
    return toSpherical(x)


# def hRotate(x, jacobian=False):
#     H = H_Rotate.to(x.device).reshape((1, n, n)).repeat(x.shape[0], 1, 1)# [batch_size, n, n] rotated matrix
#     if jacobian:
#         return torch.bmm(H,x), H
#     else:
#         return torch.bmm(H,x)

# def h_nobatch(x, jacobian=False):
#     H = H_design.to(x.device)
#     y = torch.matmul(H,x)
#     if jacobian:
#         return y, H
#     else:
#         return y
###############################################
### process noise Q and observation noise R ###
###############################################
Q_non_diag = False
R_non_diag = False

Q_structure = torch.eye(m)
R_structure = torch.eye(n)

if(Q_non_diag):
    q_d = 1
    q_nd = 1/2
    Q = torch.tensor([[q_d, q_nd, q_nd],[q_nd, q_d, q_nd],[q_nd, q_nd, q_d]])

if(R_non_diag):
    r_d = 1
    r_nd = 1/2
    R = torch.tensor([[r_d, r_nd, r_nd],[r_nd, r_d, r_nd],[r_nd, r_nd, r_d]])

##################################
### Utils for non-linear cases ###
##################################

def getJacobian(x, g):
    """
    Compute Jacobian matrix in a batched manner.
    
    input x (torch.tensor): [batch_size, m/n, 1]
    input g (function): function to be differentiated
    output Jac (torch.tensor): [batch_size, m, m] for f, [batch_size, n, m] for h
    """
    batch_size = x.shape[0]

    # Flatten the input tensor by removing the last dimension, shape: [batch_size, m/n]
    x_flat = x.squeeze(-1)

    # Compute the Jacobian for the first sample
    Jac_x0 = autograd.functional.jacobian(g, x_flat[0].unsqueeze(0))  # Shape: [1, n, m]

    # Ensure Jac_x0 is squeezed to a 2D matrix
    if Jac_x0.dim() > 2:
        Jac_x0 = Jac_x0.squeeze()  # Shape: [n, m] or [m, m]

    # Initialize the Jacobian tensor for the entire batch
    Jac = torch.zeros([batch_size, Jac_x0.size(0), Jac_x0.size(1)])  # Shape: [batch_size, n/m, m]

    # Assign the first Jacobian to the first slice of Jac
    Jac[0, :, :] = Jac_x0

    # Compute the Jacobian for the rest of the batch
    for i in range(1, batch_size):
        Jac_i = autograd.functional.jacobian(g, x_flat[i].unsqueeze(0))

        # Ensure Jac_i is a 2D matrix
        if Jac_i.dim() > 2:
            Jac_i = Jac_i.squeeze()

        Jac[i, :, :] = Jac_i  # Assign to the batch Jacobian

    return Jac



def toSpherical(cart):
    """
    input cart (torch.tensor): [batch_size, m, 1] or [batch_size, m]
    output spher (torch.tensor): [batch_size, n, 1]
    """
    rho = torch.linalg.norm(cart,dim=1).reshape(cart.shape[0], 1)# [batch_size, 1]
    phi = torch.atan2(cart[:, 1, ...], cart[:, 0, ...]).reshape(cart.shape[0], 1) # [batch_size, 1]
    phi = phi + (phi < 0).type_as(phi) * (2 * torch.pi)
    
    theta = torch.div(torch.squeeze(cart[:, 2, ...]), torch.squeeze(rho))
    theta = torch.acos(theta).reshape(cart.shape[0], 1) # [batch_size, 1]

    spher = torch.cat([rho, theta, phi], dim=1).reshape(cart.shape[0],3,1) # [batch_size, n, 1]

    return spher

def toCartesian(sphe):
    """
    input sphe (torch.tensor): [batch_size, n, 1] or [batch_size, n]
    output cart (torch.tensor): [batch_size, n]
    """
    # Ensure sphe is at least 3D (handle [batch_size, n])
    if sphe.dim() == 2:
        sphe = sphe.unsqueeze(-1)

    rho = sphe[:, 0, 0]
    theta = sphe[:, 1, 0]
    phi = sphe[:, 2, 0]

    x = (rho * torch.sin(theta) * torch.cos(phi)).unsqueeze(1)
    y = (rho * torch.sin(theta) * torch.sin(phi)).unsqueeze(1)
    z = (rho * torch.cos(theta)).unsqueeze(1)

    # Concatenate x, y, z along dimension 1 to form [batch_size, 3, 1]
    cart = torch.cat([x, y, z], dim=1).unsqueeze(-1)

    return cart