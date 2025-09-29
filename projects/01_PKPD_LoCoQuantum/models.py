from quantum_models import *

NUM_FREQS = 10

class torch_wrapper_tv(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))


class SDE(torch.nn.Module):

    noise_type = "diagonal"
    sde_type = "ito"

    # noise is sigma in this notebook for the equation sigma * (t * (1 - t))
    def __init__(self, ode_drift, noise=1.0, reverse=False):
        super().__init__()
        self.drift = ode_drift
        self.reverse = reverse
        self.noise = noise

    # Drift
    def f(self, t, y):
        if self.reverse:
            t = 1 - t
        if len(t.shape) == len(y.shape):
            x = torch.cat([y, t], 1)
        else:
            x = torch.cat([y, t.repeat(y.shape[0])[:, None]], 1)
        return self.drift(x)

    # Diffusion
    def g(self, t, y):
        return torch.ones_like(t) * torch.ones_like(y) * self.noise


class SDE_func_solver(torch.nn.Module):

    noise_type = "diagonal"
    sde_type = "ito"

    # noise is sigma in this notebook for the equation sigma * (t * (1 - t))
    def __init__(self, ode_drift, noise, reverse=False):
        super().__init__()
        self.drift = ode_drift
        self.reverse = reverse
        self.noise = noise # changeable, a model itself

    # Drift
    def f(self, t, y):
        if self.reverse:
            t = 1 - t
        if len(t.shape) == len(y.shape):
            x = torch.cat([y, t], 1)
        else:
            x = torch.cat([y, t.repeat(y.shape[0])[:, None]], 1)
        return self.drift(x)

    # Diffusion
    def g(self, t, y):
        if self.reverse:
            t = 1 - t
        if len(t.shape) == len(y.shape):
            x = torch.cat([y, t], 1)
        else:
            x = torch.cat([y, t.repeat(y.shape[0])[:, None]], 1)
        noise_result = self.noise(x)
        return noise_result* torch.sqrt(t * (1 - t))
    

def metrics_calculation(pred, true, metrics=['mse_loss'], cutoff=-0.91, map_idx = 1):
    
    # if pred is a tensor, convert to numpy
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().squeeze().numpy()
        true = true.detach().cpu().squeeze().numpy()

    loss_D = {key : None for key in metrics}
    for metric in metrics:
        if metric == 'mse_loss':
            loss_D['mse_loss'] = np.mean((pred - true)**2)
            # self.log('mse_loss', self.loss_fn(pred, true))
        if metric == 'l1_loss':
            loss_D['l1_loss'] = np.mean(np.abs(pred - true))
            # self.log('l1_loss', torch.mean(torch.abs(pred - true)))

    return loss_D

import torch
import math
import time

import matplotlib.pyplot as plt
import numpy as np
from torchdyn.core import NeuralODE
from torch import optim
import torch.functional as F


PE_BASE = 0.012 # 0.012615662610100801
NUM_FREQS = 4

def mse_loss(pred, true):
    return torch.mean((pred - true) ** 2)

def positional_encoding_tensor(time_tensor, num_frequencies=NUM_FREQS, base=PE_BASE):
    # Ensure the time tensor is in the range [0, 1]
    time_tensor = time_tensor.clamp(0, 1).unsqueeze(1)  # Clamp and add dimension for broadcasting

    # Compute the arguments for the sine and cosine functions using the custom base
    frequencies = torch.pow(base, -torch.arange(0, num_frequencies, dtype=torch.float32, device=time_tensor.device) / num_frequencies)
    angles = time_tensor * frequencies

    # Compute the sine and cosine for even and odd indices respectively
    sine = torch.sin(angles)
    cosine = torch.cos(angles)

    # Stack them along the last dimension
    pos_encoding = torch.stack((sine, cosine), dim=-1)
    pos_encoding = pos_encoding.flatten(start_dim=2)

    # Normalize to have values between 0 and 1
    pos_encoding = (pos_encoding + 1) / 2  # Now values are between 0 and 1
    
    return pos_encoding

class MLP_conditional_memory(torch.nn.Module):
    """ Conditional with many available classes

    return the class as is
    """
    def __init__(self, 
                 dim, 
                 treatment_cond,
                 memory, # how many time steps
                 out_dim=None, 
                 w=64, 
                 time_varying=False, 
                 conditional=False,  
                 time_dim = NUM_FREQS * 2,
                 clip = None,
                 ):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            self.out_dim = dim 
        self.out_dim += 1 # for the time dimension
        self.treatment_cond = treatment_cond
        self.memory = memory
        self.dim = dim
        self.indim = dim + (time_dim if time_varying else 0) + (self.treatment_cond if conditional else 0) + (dim * memory)
        self.net = torch.nn.Sequential(
           torch.nn.Linear(self.indim, w),
#             torch.nn.Tanh(),
            QuantumEntanglingLinearVectorized(w),
#             QuantumEntanglingLinear_new(w),
#             QuantumEntanglingLinear(w),
            torch.nn.SELU(),
           torch.nn.Linear(w,self.out_dim),
        )
        self.default_class = 0
        self.clip = clip

    def encoding_function(self, time_tensor):
        return positional_encoding_tensor(time_tensor)    

    def forward_train(self, x):
        """forward pass
        Assume first two dimensions are x, c, then t
        """
        if self.time_varying:
            time_tensor = x[:,-1]
            encoded_time_span = self.encoding_function(time_tensor).reshape(-1, NUM_FREQS * 2)
            new_x = torch.cat([x[:,:-1], encoded_time_span], dim=1)
        else:
            new_x = x[:, :-1]
        result = self.net(new_x)
        return torch.cat([result[:,:-1], x[:,self.dim:-1], result[:,-1].unsqueeze(1)], dim=1)

    def forward(self, x):
        """ call forward_train for training
            x here is x_t
            xt = (t)x1 + (1-t)x0
            (xt - tx1)/(1-t) = x0
        """
        x1 = self.forward_train(x)
        x1_coord = x1[:,:self.dim]
        t = x[:,-1]
        pred_time_till_t1 = x1[:,-1]
        x_coord = x[:,:self.dim]
        if self.clip is None:
            vt = (x1_coord - x_coord)/(pred_time_till_t1)
        else:
            vt = (x1_coord - x_coord)/torch.clip((pred_time_till_t1),min=self.clip)

        final_vt = torch.cat([vt, torch.zeros_like(x[:,self.dim:-1])], dim=1)
        return final_vt

class MLP_Cond_Memory_Module(torch.nn.Module):
    def __init__(self, treatment_cond, memory=3, dim=2, w=64, time_varying=True, conditional=True, lr=1e-6, sigma=0.1, 
                 loss_fn=mse_loss, metrics=['mse_loss', 'l1_loss'], implementation="ODE", sde_noise=0.1, clip=None, naming=None):
        super().__init__()
        self.model = MLP_conditional_memory(dim=dim, w=w, time_varying=time_varying, conditional=conditional, 
                                            treatment_cond=treatment_cond, memory=memory, clip=clip)
        self.loss_fn = loss_fn
        self.dim = dim
        self.w = w
        self.time_varying = time_varying
        self.conditional = conditional
        self.treatment_cond = treatment_cond
        self.lr = lr
        self.sigma = sigma
        self.metrics = metrics
        self.implementation = implementation
        self.memory = memory
        self.sde_noise = sde_noise
        self.clip = clip

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def __convert_tensor__(self, tensor):
        return tensor.to(torch.float32)

    def __x_processing__(self, x0, x1, t0, t1):
        t = torch.rand(x0.shape[0], 1).to(x0.device)
        mu_t = x0 * (1 - t) + x1 * t
        data_t_diff = (t1 - t0).unsqueeze(1)
        x = mu_t + self.sigma * torch.randn(x0.shape[0], self.dim).to(x0.device)
        ut = (x1 - x0) / (data_t_diff + 1e-4)
        t_model = t * data_t_diff + t0.unsqueeze(1)
        futuretime = t1.unsqueeze(1) - t_model
        return x, ut, t_model, futuretime, t
    

class MLP_conditional_memory_sde_noise(torch.nn.Module):
    """ Conditional with many available classes

    return the class as is
    """
    def __init__(self, 
                 dim, 
                 treatment_cond,
                 memory, # how many time steps
                 out_dim=None, 
                 w=64, 
                 time_varying=False, 
                 conditional=False,  
                 time_dim = NUM_FREQS * 2,
                 clip = None,
                 ):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            self.out_dim = 1 # for noise 
        self.treatment_cond = treatment_cond
        self.memory = memory
        self.dim = dim
        self.indim = dim + (time_dim if time_varying else 0) + (self.treatment_cond if conditional else 0) + (dim * memory)
        self.net = VQC_class(self.indim, self.out_dim, n_layers=2, n_qubits=w)
        self.default_class = 0
        self.clip = clip

    def encoding_function(self, time_tensor):
        return positional_encoding_tensor(time_tensor)    

    def forward_train(self, x):
        """forward pass
        Assume first two dimensions are x, c, then t
        """
        time_tensor = x[:,-1]
        encoded_time_span = self.encoding_function(time_tensor).reshape(-1, NUM_FREQS * 2)
        new_x = torch.cat([x[:,:-1], encoded_time_span], dim=1)
        result = self.net(new_x)
        return result
    
    def forward(self,x):
        result = self.forward_train(x)
        return torch.cat([result, torch.zeros_like(x[:,1:-1])], dim=1)

def mse_loss(pred, true):
    return torch.mean((pred - true) ** 2)

class Noise_MLP_Cond_Memory_Module(torch.nn.Module):
    def __init__(self, treatment_cond, memory=3, dim=2, w=64, time_varying=True, conditional=True, lr=1e-6, sigma=0.1, 
                 loss_fn=mse_loss, metrics=['mse_loss', 'l1_loss'], implementation="ODE", sde_noise=0.1, clip=None, naming=None):
        super().__init__()
        self.flow_model = MLP_conditional_memory(dim=dim, w=w, time_varying=time_varying, conditional=conditional, 
                                                 treatment_cond=treatment_cond, memory=memory, clip=clip)
        if implementation == "SDE":
            self.noise_model = MLP_conditional_memory_sde_noise(dim=dim, w=w, time_varying=time_varying, conditional=conditional, 
                                                                treatment_cond=treatment_cond, memory=memory, clip=clip)
        else:
            self.noise_model = MLP_conditional_memory(dim=dim, w=w, time_varying=time_varying, conditional=conditional, 
                                                      treatment_cond=treatment_cond, memory=memory, clip=clip)
        self.loss_fn = loss_fn
        self.dim = dim
        self.w = w
        self.time_varying = time_varying
        self.conditional = conditional
        self.treatment_cond = treatment_cond
        self.lr = lr
        self.sigma = sigma
        self.metrics = metrics
        self.implementation = implementation
        self.memory = memory
        self.sde_noise = sde_noise
        self.clip = clip

    def forward(self, x):
        return self.flow_model(x)

    def configure_optimizers(self):
        flow_optimizer = torch.optim.Adam(self.flow_model.parameters(), lr=self.lr)
        noise_optimizer = torch.optim.Adam(self.noise_model.parameters(), lr=self.lr)
        return flow_optimizer, noise_optimizer

    def __convert_tensor__(self, tensor):
        return tensor.to(torch.float32)

    def __x_processing__(self, x0, x1, t0, t1):
        t = torch.rand(x0.shape[0], 1).to(x0.device)
        mu_t = x0 * (1 - t) + x1 * t
        data_t_diff = (t1 - t0).unsqueeze(1)
        x = mu_t + self.sigma * torch.randn(x0.shape[0], self.dim).to(x0.device)
        ut = (x1 - x0) / (data_t_diff + 1e-4)
        t_model = t * data_t_diff + t0.unsqueeze(1)
        futuretime = t1 - t_model
        return x, ut, t_model, futuretime, t

class MLP_Cond_Memory_Module(torch.nn.Module):
    def __init__(self, treatment_cond, memory=3, dim=2, w=64, time_varying=True, conditional=True, lr=1e-6, sigma=0.1, 
                 loss_fn=mse_loss, metrics=['mse_loss', 'l1_loss'], implementation="ODE", sde_noise=0.1, clip=None, naming=None):
        super().__init__()
        self.model = MLP_conditional_memory(dim=dim, w=w, time_varying=time_varying, conditional=conditional, 
                                            treatment_cond=treatment_cond, memory=memory, clip=clip)
        self.loss_fn = loss_fn
        self.dim = dim
        self.w = w
        self.time_varying = time_varying
        self.conditional = conditional
        self.treatment_cond = treatment_cond
        self.lr = lr
        self.sigma = sigma
        self.metrics = metrics
        self.implementation = implementation
        self.memory = memory
        self.sde_noise = sde_noise
        self.clip = clip

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def __convert_tensor__(self, tensor):
        return tensor.to(torch.float32)

    def __x_processing__(self, x0, x1, t0, t1):
        t = torch.rand(x0.shape[0], 1).to(x0.device)
        mu_t = x0 * (1 - t) + x1 * t
        data_t_diff = (t1 - t0).unsqueeze(1)
        x = mu_t + self.sigma * torch.randn(x0.shape[0], self.dim).to(x0.device)
        ut = (x1 - x0) / (data_t_diff + 1e-4)
        t_model = t * data_t_diff + t0.unsqueeze(1)
        futuretime = t1 - t_model
        return x, ut, t_model, futuretime, t

