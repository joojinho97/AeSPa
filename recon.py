import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from mamba_ssm import Mamba
import sys
from model import *
from utils import ideal_bandpass
import random
import hydra
from hydra.core.config_store import ConfigStore
from config import MRINeRF_Config
import logging
from torch.utils.tensorboard import SummaryWriter
from mr_utils import kspace_to_img_shifted_mc, img_to_kspace_shifted_mc, coil_combine, coil_unfold,k_loss_l1, process_and_undersample_k_space,get_mask,train
from utils import train_timing

from model import CCM,CSM, mambalayer
from utils import set_seed, crop_center, random_2d_indices, plot_metrics, dss, total_variation_loss_cc_jh, total_variation_loss_cmap_jh, cc_loss
torch.fx.wrap('base_cpp.forward')
torch.autograd.set_detect_anomaly(True)
cs = ConfigStore.instance()
cs.store(name="mri_nerf_config", node=MRINeRF_Config)
maked_k_space = 0
log = logging.getLogger(__name__)
torch.autograd.set_detect_anomaly(True)


@hydra.main(config_path="conf", config_name='AeSPa')
def main(cfg: MRINeRF_Config):
    print(cfg)
    set_seed(2027)


    if torch.cuda.is_available():
        # Set the device to the first available GPU
        device = torch.device("cuda:" + cfg.hyper_params.gpu)
        print("Running on the GPU" + cfg.hyper_params.gpu)
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    try:
        k_space = os.path.join(cfg.files.k_space, cfg.files.subject, cfg.files.slice)
        coil_map = os.path.join(cfg.files.coil_map, cfg.files.subject, cfg.files.slice)
        np.load(k_space)
    except:
        print('cfg.files.subject',cfg.files.subject)
        print('cfg.files.k_space',cfg.files.k_space)
        print('cfg.files.slice',cfg.files.slice)
        k_space = os.path.join(cfg.files.k_space)
        coil_map = os.path.join(cfg.files.coil_map)

    model = cfg.models.model
    epochs = cfg.hyper_params.epochs
    reduction_factor = cfg.params.reduction_factor
    acs = cfg.params.acs

    # Load the npy files
    print('k_space directory',k_space)
    print('coil_map directory',coil_map)
    k = np.squeeze(np.load(k_space))
    print('k.shape[0]',k.shape[0])

    kwargs = {
        'log': log,
        'k': k.shape[0]
    }
    
    acs_start = (k.shape[-1] - acs) // 2
    acs_end = (k.shape[-1] + acs) // 2 - 1

    if cfg.params.mask =='uniform1d' and reduction_factor==4:
        mask = np.zeros((k.shape[-2], k.shape[-1]), dtype=float)
        mask[:,(k.shape[-1]-acs)//2:(k.shape[-1]+acs)//2-1] = 1
        mask[:,::(cfg.params.reduction_factor+2)] = 1
        save_folder = os.path.join(cfg.paths.save_folder, f'accel{reduction_factor}_norm', cfg.files.subject, cfg.files.slice)
    
    if cfg.params.mask =='gaussian1d' and reduction_factor==4:
        mask = get_mask(torch.zeros([1, 1, 640, 320]),320, 1,type='gaussian1d',acc_factor=cfg.params.reduction_factor,center_fraction=cfg.params.center_fraction)
        mask = mask.cpu().detach().numpy()[0][0]
        save_folder = os.path.join(cfg.paths.save_folder, 'gaussian1d',f'accel{cfg.params.reduction_factor}_norm', cfg.files.subject, cfg.files.slice)
    os.makedirs(f'{save_folder}/c_est', exist_ok=True)
    


    k_space, mask, undersampled_k_space  = process_and_undersample_k_space(k, mask,device)

    mrim = kspace_to_img_shifted_mc(torch.tensor(undersampled_k_space,dtype=torch.complex128))
    mrim = torch.sqrt(torch.sum(torch.abs(mrim)**2,dim=0))
    mrim_uk_max = torch.max(torch.abs(mrim))
    undersampled_k_space = undersampled_k_space/mrim_uk_max
    k_space = k_space /mrim_uk_max

    combined_img_target = kspace_to_img_shifted_mc(k_space)
    combined_img_target = torch.sqrt(torch.sum(torch.abs(combined_img_target)**2,dim=0))
    combined_img_target = crop_center(combined_img_target,320)
    combined_img_target = combined_img_target /torch.max(torch.abs(combined_img_target))
    combined_img_target = torch.abs(combined_img_target)
    combined_img_target = ideal_bandpass(combined_img_target.unsqueeze(0),cfg.params.cutoff_low_freq,False)
    combined_img_target = combined_img_target[0]
    combined_img_target = ideal_bandpass(combined_img_target.unsqueeze(0),cfg.params.cutoff_high_freq)
    combined_img_target = combined_img_target[0]

    mrim_test_ = kspace_to_img_shifted_mc(undersampled_k_space)
    mrim_test = torch.sqrt(torch.sum(torch.abs(mrim_test_)**2,dim=0))
    mrim_test_phase = torch.angle(torch.sum(mrim_test_,dim=0))
    
    maska = torch.zeros_like(mask)

    for i in range(acs_start,acs_end+1,1):
        maska[:,:,i]=1

    acs_k_space = undersampled_k_space*maska
    maska= maska.to(device)

    acs_image = kspace_to_img_shifted_mc(acs_k_space)

    acs_image = torch.cat([acs_image.real,acs_image.imag]).float().to(device)
    acs_image = acs_image / torch.max(torch.abs(acs_image))


    # Initialize network
    if model == 'joint_training':
        net = CCM(**kwargs).to(device)
        net2 = CSM(**kwargs).to(device)
        optimizer2 = optim.Adam(net2.parameters(), lr=cfg.hyper_params.lr)
    mambamodule=mambalayer().to(device)
    
    optimizer = optim.Adam(net.parameters(), lr=cfg.hyper_params.lr)
    optimizer3 = optim.Adam(mambamodule.parameters(), lr=cfg.hyper_params.aksmlr)
    
    initial_input = torch.complex(mrim_test*torch.cos(mrim_test_phase), mrim_test*torch.sin(mrim_test_phase))
    nomalized_initial_input = initial_input / torch.max(torch.abs(mrim_test))
    
    #setting
    width, height = k_space[0,:,:].shape
    best_loss = 1
    kspace_repository_r = torch.zeros_like(undersampled_k_space, dtype=torch.float32)
    kspace_repository_i = torch.zeros_like(undersampled_k_space, dtype=torch.float32)
    
    writer = SummaryWriter(log_dir=f'{save_folder}/runs')

    with train_timing(log, epochs, save_folder):
        for epoch in range(epochs):
            global predicted_c_c_c_c
            global predicted_k_space_k
            global predicted_c_c_c

            if epoch == 0:
                forward_kwargs={
                    'net': net,
                    'net2': net2,
                    'u_k': acs_image,
                    'coords': torch.cat([nomalized_initial_input.real.unsqueeze(0),nomalized_initial_input.imag.unsqueeze(0)],dim=0).to(device), #
                    'width': width,
                    'height': height,
                    'step': epoch,
                    }
            
            
            predicted_c_c, predicted_cmap = train(**forward_kwargs)
            predicted_cmap = dss(predicted_cmap)
            mr_image = coil_unfold(predicted_c_c, predicted_cmap)
            predicted_c_c_c_c = predicted_c_c.detach().clone()
            predicted_c_c_s  =predicted_c_c.clone()
            predicted_c_c = crop_center(predicted_c_c,320)
            predicted_k_space = img_to_kspace_shifted_mc(mr_image)
            
            loss4 = total_variation_loss_cc_jh(torch.abs(predicted_c_c_s)) 
            loss3 = total_variation_loss_cmap_jh(torch.abs(predicted_cmap))
            loss2 = cc_loss(predicted_cmap)
            loss1 = k_loss_l1(undersampled_k_space, predicted_k_space, mask)
            loss6=loss1.clone().detach()
            net1_loss = loss1 +loss4
            net2_loss = loss6+loss2+loss3

            net1_loss.backward(retain_graph=True)
            net2_loss.backward()
            
            optimizer.step()
            optimizer2.step()
            optimizer.zero_grad()
            optimizer2.zero_grad()
            
                
            predicted_k_spaces = predicted_k_space.detach().clone() 
            predicted_k_space_att,att_score = mambamodule(torch.cat([predicted_k_spaces.real.unsqueeze(0).detach(),predicted_k_spaces.imag.unsqueeze(0).detach()],dim=0).float().to(device))
            att_score = torch.sigmoid(att_score)

            try:
                att_score = torch.complex(att_score[0,:16], att_score[0,16:])
            except:
                att_score = torch.complex(att_score[0,:20], att_score[0,20:])
            
            predicted_k_space_t = att_score*predicted_k_spaces
            mr = kspace_to_img_shifted_mc(predicted_k_space_t)
            ccimage = coil_combine(mr,predicted_cmap.detach())
            ccimage = ccimage / torch.max(torch.abs(ccimage))
            aksm_loss = total_variation_loss_cc_jh(ccimage)
            
            optimizer3.zero_grad()
            aksm_loss.backward()
            optimizer3.step()

            if epoch==0:
                top_values, top_indices_r = torch.topk(predicted_k_space_att.detach(), k.shape[1],dim=-2)
            else:
                top_values, top_indices_r = torch.topk(predicted_k_space_att.detach(), cfg.params.selective_top_k,dim=-2)

            for i in range(predicted_k_spaces.size(0)):
                kspace_repository_r[i].scatter_(-2, top_indices_r[i], predicted_k_spaces.real[i].gather(-2, top_indices_r[i]).float())
                kspace_repository_i[i].scatter_(-2, top_indices_r[i], predicted_k_spaces.imag[i].gather(-2, top_indices_r[i]).float())
            predicted_k_space_final = torch.complex(kspace_repository_r,kspace_repository_i)
            

            if epoch==0 or(epoch+1) % cfg.params.csm_update_iteration == 0 :
                random_rows, random_cols,random_chas = random_2d_indices(predicted_c_c_s, cfg.params.kspace_masking,epoch,k.shape[0])
                predicted_k_space_k = predicted_k_space.detach().clone()
                predicted_k_space_k[random_chas,random_rows, random_cols] = 0
                predicted_k_space_k = predicted_k_space_k/mrim_uk_max
                predicted_k_space_k = predicted_k_space_k*(1-mask) + undersampled_k_space*mask
                predicted_c_c_c = kspace_to_img_shifted_mc(predicted_k_space_k)
                predicted_c_c_c = predicted_c_c_c /torch.max(torch.abs(predicted_c_c_c_c))
                predicted_c_c_c_c = coil_combine(predicted_c_c_c, predicted_cmap)
                predicted_c_c_c_c = predicted_c_c_c_c/torch.max(torch.abs(predicted_c_c_c_c))

            else:
                predicted_k_space_k = predicted_k_space.detach().clone() 
                predicted_c_c_c_c = coil_combine(predicted_c_c_c, predicted_cmap)
                predicted_c_c_c_c = predicted_c_c_c_c/torch.max(torch.abs(predicted_c_c_c_c))

            if epoch !=0 and (epoch+1) % 1 == 0: #

                forward_kwargs={
                    'net': net,
                    'net2': net2,
                    'u_k': torch.cat([predicted_c_c_c.real.detach(),predicted_c_c_c.imag.detach()],dim=0).float().to(device),
                    'coords': torch.cat([predicted_c_c_c_c.real.unsqueeze(0).detach(),predicted_c_c_c_c.imag.unsqueeze(0).detach()],dim=0).float().to(device),
                    'width': width,
                    'height': height,
                    'step': epoch,
                    }



if __name__ == "__main__":
    
    seed = 2027
    deterministic = True
    x, y, box_width, box_height = 50, 50, 100, 100
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(mode=True)
    main()

