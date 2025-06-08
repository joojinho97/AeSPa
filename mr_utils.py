import torch
import numpy as np
from scipy.fftpack import fftshift, ifftshift, ifft2, fft2
import os
import random


# Function to transform k-space to image-space (with shift correction)
def kspace_to_img_shifted_mc(kspace):
    shifted_k_space = torch.fft.ifftshift(kspace, dim=[1,2])
    shifted_img_space = torch.fft.ifft2(shifted_k_space)
    return torch.fft.fftshift(shifted_img_space, dim=[1,2])

# Function to transform k-space to image-space (with shift correction)
def img_to_kspace_shifted_mc(img):
    shifted_img_space = torch.fft.fftshift(img, dim=[1,2])
    shifted_k_space = torch.fft.fft2(shifted_img_space)
    return torch.fft.ifftshift(shifted_k_space, dim=[1,2])

def coil_combine(img, c_map):
    c_m = torch.sqrt(torch.sum(torch.abs(img)**2, dim=0))
    c_p = torch.angle(torch.sum(img * torch.conj(c_map), dim=0))
    return torch.complex(c_m*torch.cos(c_p), c_m*torch.sin(c_p))

def coil_unfold(c, c_map):
    return c.unsqueeze(0).expand_as(c_map) * c_map
 

def train(**kwargs):
    net=kwargs['net']
    net2=kwargs['net2']
    coords=kwargs['coords']
    width=kwargs['width']
    height=kwargs['height']
    step=kwargs['step']
    acs_image = kwargs['u_k']
    
    ri = net(coords)
    ri = ri.view(2,-1)
    
    output = torch.complex(ri[0,:], ri[1,:])
    output = output.view(width,height)

    if acs_image.shape[0]==40:
        ri2 = net2(acs_image)
        ri2 = ri2.view(40,-1)
        output2 = torch.complex(ri2[:20,:],ri2[20:,:])
        output2 = output2.view(20,width,height)
    else:
        ri2 = net2(acs_image)
        ri2 = ri2.view(32,-1)
        output2 = torch.complex(ri2[:16,:],ri2[16:,:])
        output2 = output2.view(16,width,height)
    
    return output,output2


def k_loss_l1(undersampled, predicted, mask):
    return torch.mean(torch.abs((undersampled - predicted)*mask))
   


def process_and_undersample_k_space(k_space, mask,device):
    # Check if data is in numpy array format
    if not isinstance(k_space, np.ndarray):
        raise ValueError("Data is not in numpy array format")

    # Convert data to img_space using FFT
    img_space = np.zeros_like(k_space, dtype=np.complex64)
    for i in range(k_space.shape[0]):
        img_space[i,:,:] = ifftshift(ifft2(fftshift(k_space[i,:,:])))
        
    

    # Multiply the mask with k-space data elementwise
    undersampled_k_space = np.zeros_like(k_space, dtype=np.complex64)
    for i in range(k_space.shape[0]):
        undersampled_k_space[i,:,:] = k_space[i,:,:] * mask

    # Convert to PyTorch tensors
    k_space = torch.from_numpy(k_space).to(device)

    mask = torch.from_numpy(mask).to(device)
    mask = mask.unsqueeze(0).expand_as(k_space)
    undersampled_k_space = torch.from_numpy(undersampled_k_space).to(device)
    
    return k_space, mask, undersampled_k_space 


def get_mask(img, size, batch_size, type='gaussian2d', acc_factor=8, center_fraction=0.04, fix=False,sizey=640):
    mux_in = size ** 2
    if type.endswith('2d'):
        Nsamp = mux_in // acc_factor
    elif type.endswith('1d'):
        Nsamp = size // acc_factor
        
    if type == 'gaussian2d':
        mask = torch.zeros_like(img)
        cov_factor = size * (1.5 / 128)
        mean = [sizey // 2, size // 2]
        cov = [[size * cov_factor, 0], [0, size * cov_factor]]
        if fix:
            samples = np.random.multivariate_normal(mean, cov, int(Nsamp))
            int_samples = samples.astype(int)
            int_samples = np.clip(int_samples, 0, size - 1)
            int_samples_y = np.clip(int_samples, 0, sizey - 1)
            mask[..., int_samples_y[:, 0], int_samples[:, 1]] = 1
        else:
            for i in range(batch_size):
                # sample different masks for batch
                samples = np.random.multivariate_normal(mean, cov, int(Nsamp))
                int_samples = samples.astype(int)
                int_samples = np.clip(int_samples, 0, sizey - 1)
                
                
                mask[i, :, int_samples[:, 0], int_samples[:, 1]] = 1
    elif type == 'uniformrandom2d':
        mask = torch.zeros_like(img)
        if fix:
            mask_vec = torch.zeros([1, size * size])
            samples = np.random.choice(size * size, int(Nsamp))
            mask_vec[:, samples] = 1
            mask_b = mask_vec.view(size, size)
            mask[:, ...] = mask_b
        else:
            for i in range(batch_size):
            # sample different masks for batch
                mask_vec = torch.zeros([1, size * size])
                samples = np.random.choice(size * size, int(Nsamp))
                mask_vec[:, samples] = 1
                mask_b = mask_vec.view(size, size)
                mask[i, ...] = mask_b
    elif type == 'gaussian1d':
        mask = torch.zeros_like(img)
        mean = size // 2

        std = size * (15.0 / 128)
        Nsamp_center = int(size * center_fraction)
        if fix:
            samples = np.random.normal(loc=mean, scale=std, size=int(Nsamp * 1.2))
            
            int_samples = samples.astype(int)
            int_samples = np.clip(int_samples, 0, size - 1)
            mask[... , int_samples] = 1
            c_from = size // 2 - Nsamp_center // 2
            mask[... , c_from:c_from + Nsamp_center] = 1
        else:
            for i in range(batch_size):
                samples = np.random.normal(loc=mean, scale=std, size=int(Nsamp*1.2))
                
                int_samples = samples.astype(int)
                int_samples = np.clip(int_samples, 0, size - 1)
                mask[i, :, :, int_samples] = 1
                c_from = size // 2 - Nsamp_center // 2
                mask[i, :, :, c_from:c_from + Nsamp_center] = 1
    elif type == 'uniform1d':
        mask = torch.zeros_like(img)
        if fix:
            Nsamp_center = int(size * center_fraction)
            samples = np.random.choice(size, int(Nsamp - Nsamp_center))
            mask[..., samples] = 1
            # ACS region
            c_from = size // 2 - Nsamp_center // 2
            mask[..., c_from:c_from + Nsamp_center] = 1
        else:
            for i in range(batch_size):
                Nsamp_center = int(size * center_fraction)
                samples = np.random.choice(size, int(Nsamp - Nsamp_center))
                mask[i, :, :, samples] = 1
                # ACS region
                c_from = size // 2 - Nsamp_center // 2
                mask[i, :, :, c_from:c_from+Nsamp_center] = 1
    else:
        NotImplementedError(f'Mask type {type} is currently not supported.')

    return mask
