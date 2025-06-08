import time
from contextlib import contextmanager
import torch
import numpy as np
import os
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import torch
from torch import Tensor
from typing import Tuple
import random

def plot_metrics(losses, psnrs, ssims, output_file='/path',rf=4,name='named',psnr = 0,ssim = 0):
    """
    Plots loss, PSNR, and SSIM values and saves the plot to a file.

    Parameters:
    losses (list of float): List of loss values.
    psnrs (list of float): List of PSNR values.
    ssims (list of float): List of SSIM values.
    output_file (str): File name for the output plot image.
    """
    output_file = f'{output_file}/{name}_{rf}.png'
    epochs = list(range(1, len(losses) + 1))

    plt.figure(figsize=(12, 6))

    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, losses, label='Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    # Plot PSNR
    plt.subplot(1, 3, 2)
    plt.plot(epochs, psnrs, label='PSNR', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    
    plt.title(f'PSNR : {psnr} over Epochs')
    plt.legend()

    # Plot SSIM
    plt.subplot(1, 3, 3)
    plt.plot(epochs, ssims, label='SSIM', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title(f'SSIM : {ssim} over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

def random_2d_indices(array_shape, num_indices,epoch,kshape=15):
    """
    주어진 배열 모양 내에서 랜덤한 2D 인덱스를 생성하는 함수입니다.
    
    Parameters:
    array_shape (tuple): 배열의 모양 (행, 열)
    num_indices (int): 생성할 랜덤 인덱스의 수
    
    Returns:
    tuple: 랜덤 2D 인덱스의 튜플 (rows, cols)
    """
    np.random.seed(epoch)
    width = array_shape.shape[1]
    height = array_shape.shape[0]
    channel=int(kshape)
    chas = torch.randint(0, channel, (num_indices,))
    rows = torch.randint(0, height, (num_indices,))
    cols = torch.randint(0, width, (num_indices,))
    np.random.seed(2027)

    return rows, cols, chas

def crop_center(img, cropx, cropy=None):
    # taken from https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image/39382475
    if cropy is None:
        cropy = cropx
    y, x = img.shape[-2:]

    startx = x//2 - (cropx//2)
    starty = y//2 - (cropy//2)
    return img[..., starty:starty+cropy, startx:startx+cropx]

def diff_map(img,target,save_folder,epoch):
    plt.figure()
    plt.imshow(np.abs(np.abs(img)-np.abs(target)))
    plt.colorbar()
    plt.clim(0, 0.1)
    plt.savefig(f'{save_folder}/diffmap{epoch}.jpg')
    
def total_variation_loss_cmap(img):
     c_img,h_img, w_img = img.size()
     tv_h = torch.pow(img[:,1:,:]-img[:,:-1,:], 2).sum()
     tv_w = torch.pow(img[:,:,1:]-img[:,:,:-1], 2).sum()
     return torch.abs((tv_h+tv_w)/(h_img*w_img*c_img))

def total_variation_loss_cc(img):
     h_img, w_img = img.size()
     tv_h =torch.pow(img[1:,:]-img[:-1,:], 2).sum()
     tv_w =torch.pow(img[:,1:]-img[:,:-1], 2).sum()
     return torch.abs((tv_h+tv_w)/(h_img*w_img))

def total_variation_loss_cmap_jh(img):
     c_img,h_img, w_img = img.size()
     tv_h = torch.abs(img[:,1:,:]-img[:,:-1,:]).sum()
     tv_w = torch.abs(img[:,:,1:]-img[:,:,:-1]).sum()
     return (tv_h+tv_w)/(h_img*w_img*c_img)
 
def total_variation_loss_cc_jh(img):
     h_img, w_img = img.size()
     tv_h =torch.abs(img[1:,:]-img[:-1,:]).sum()
     tv_w =torch.abs(img[:,1:]-img[:,:-1]).sum()

     return (tv_h+tv_w)/(h_img*w_img)

def cc_loss(predicted_cmap):
    return torch.mean(torch.abs(torch.abs(torch.ones_like(predicted_cmap[0]))-torch.sum(torch.abs(predicted_cmap) ** 2,dim=0)))

def dss(predicted_cmap):
    return predicted_cmap/torch.sqrt(torch.sum(torch.abs(predicted_cmap) ** 2,dim=0))
  

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def set_seed(seed):
    # Python random seed 고정
    random.seed(seed)
    
    # Numpy random seed 고정
    np.random.seed(seed)
    
    # PyTorch random seed 고정
    torch.manual_seed(seed)
    
    # CUDA 사용 시 (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 멀티 GPU 사용 시 모든 GPU에 시드 고정
    
    # PyTorch 연산에서 일관된 결과를 보장하기 위해 사용
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@contextmanager
def train_timing(log, epochs, save_folder):
    
    # designate save_folder
    os.makedirs(f'{save_folder}/images', exist_ok=True)
    os.makedirs(f'{save_folder}/weights', exist_ok=True)
    
    # Set timer 
    t0 = time.time()
    formatted_datetime = time.strftime('%Y-%m-%d -> %H-%M-%S', time.localtime(t0))
    file_name = f'{save_folder}/train -> {formatted_datetime}.txt'
    file_content = f'Train exp log and config file saved at \n ./output/{formatted_datetime} \n The time in seconds might differ slightly.'
    # Write the content to the file
    with open(file_name, 'w') as file:
        file.write(file_content)
    yield
    t1 = time.time()  
    elapsed_time = t1-t0
    elapsed_mins = int(elapsed_time / 60)
    log.info("Training time: %0.2fsec / %dmins", elapsed_time, elapsed_mins)
    log.info("Per Iteration time: %0.2fsec", elapsed_time/epochs)







def _get_center_distance(size: Tuple[int], device: str = 'cpu') -> Tensor:
    """Compute the distance of each matrix element to the center.

    Args:
        size (Tuple[int]): [m, n].
        device (str, optional): cpu/cuda. Defaults to 'cpu'.

    Returns:
        Tensor: [m, n].
    """    
    m, n = size
    i_ind = torch.tile(
                torch.tensor([[[i]] for i in range(m)], device=device),
                dims=[1, n, 1]).float()  # [m, n, 1]
    j_ind = torch.tile(
                torch.tensor([[[i] for i in range(n)]], device=device),
                dims=[m, 1, 1]).float()  # [m, n, 1]
    ij_ind = torch.cat([i_ind, j_ind], dim=-1)  # [m, n, 2]
    ij_ind = ij_ind.reshape([m * n, 1, 2])  # [m * n, 1, 2]
    center_ij = torch.tensor(((m - 1) / 2, (n - 1) / 2), device=device).reshape(1, 2)
    center_ij = torch.tile(center_ij, dims=[m * n, 1, 1])
    dist = torch.cdist(ij_ind, center_ij, p=2).reshape([m, n])
    return dist


def _get_ideal_weights(size: Tuple[int], D0: int, lowpass: bool = True, device: str = 'cpu') -> Tensor:
    """Get H(u, v) of ideal bandpass filter.

    Args:
        size (Tuple[int]): [H, W].
        D0 (int): The cutoff frequency.
        lowpass (bool): True for low-pass filter, otherwise for high-pass filter. Defaults to True.
        device (str, optional): cpu/cuda. Defaults to 'cpu'.

    Returns:
        Tensor: [H, W].
    """    
    center_distance = _get_center_distance(size, device)
    center_distance[center_distance > D0] = -1
    center_distance[center_distance != -1] = 1
    if lowpass is True:
        center_distance[center_distance == -1] = 0
    else:
        center_distance[center_distance == 1] = 0
        center_distance[center_distance == -1] = 1
    return center_distance


def _to_freq(image: Tensor) -> Tensor:
    """Convert from spatial domain to frequency domain.

    Args:
        image (Tensor): [B, C, H, W].

    Returns:
        Tensor: [B, C, H, W]
    """    
    img_fft = torch.fft.fft2(image)
    img_fft_shift = torch.fft.fftshift(img_fft)
    return img_fft_shift


def _to_space(image_fft: Tensor) -> Tensor:
    """Convert from frequency domain to spatial domain.

    Args:
        image_fft (Tensor): [B, C, H, W].

    Returns:
        Tensor: [B, C, H, W].
    """    
    img_ifft_shift = torch.fft.ifftshift(image_fft)
    img_ifft = torch.fft.ifft2(img_ifft_shift)
    img = img_ifft.real.clamp(0, 1)
    return img


def ideal_bandpass(image: Tensor, D0: int, lowpass: bool = True) -> Tensor:
    """Low-pass filter for images.

    Args:
        image (Tensor): [B, C, H, W].
        D0 (int): Cutoff frequency.
        lowpass (bool): True for low-pass filter, otherwise for high-pass filter. Defaults to True.

    Returns:
        Tensor: [B, C, H, W].
    """    
    img_fft = _to_freq(image)
    weights = _get_ideal_weights(img_fft.shape[-2:], D0=D0, lowpass=lowpass, device=image.device)
    img_fft = img_fft * weights
    img = _to_space(img_fft)
    return img

#### Butterworth

def _get_butterworth_weights(size: Tuple[int], D0: int, n: int, device: str = 'cpu') -> Tensor:
    """Get H(u, v) of Butterworth filter.

    Args:
        size (Tuple[int]): [H, W].
        D0 (int): The cutoff frequency.
        n (int): Order of Butterworth filters.
        device (str, optional): cpu/cuda. Defaults to 'cpu'.

    Returns:
        Tensor: [H, W].
    """    
    center_distance = _get_center_distance(size=size, device=device)
    weights = 1 / (1 + torch.pow(center_distance / D0, 2 * n))
    return weights


def butterworth(image: Tensor, D0: int, n: int) -> Tensor:
    """Butterworth low-pass filter for images.

    Args:
        image (Tensor): [B, C, H, W].
        D0 (int): Cutoff frequency.
        n (int): Order of the Butterworth low-pass filter.

    Returns:
        Tensor: [B, C, H, W].
    """    
    img_fft = _to_freq(image)
    weights = _get_butterworth_weights(image.shape[-2:], D0, n, device=image.device)
    img_fft = weights * img_fft
    img = _to_space(img_fft)
    return img


#### Gaussian
def _get_gaussian_weights(size: Tuple[int], D0: float, device: str = 'cpu') -> Tensor:
    """Get H(u, v) of Gaussian filter.

    Args:
        size (Tuple[int]): [H, W].
        D0 (float): The cutoff frequency.
        device (str, optional): cpu/cuda. Defaults to 'cpu'.

    Returns:
        Tensor: [H, W].
    """    
    center_distance = _get_center_distance(size=size, device=device)
    weights = torch.exp(- (torch.square(center_distance) / (2 * D0 ** 2)))
    return weights


def gaussian(image: Tensor, D0: float) -> Tensor:
    """Gaussian low-pass filter for images.

    Args:
        image (Tensor): [B, C, H, W].
        D0 (int): Cutoff frequency.

    Returns:
        Tensor: [B, C, H, W].
    """    
    weights = _get_gaussian_weights(image.shape[-2:], D0=D0, device=image.device)
    image_fft = _to_freq(image)
    image_fft = image_fft * weights
    image = _to_space(image_fft)
    return image
