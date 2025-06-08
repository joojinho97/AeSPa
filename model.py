import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch import nn
import math
from mamba_ssm import Mamba
import random
import os
import pdb
seed=2027

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch,init_type='default'):
        super(DoubleConv, self).__init__()
        if init_type == 'default':
            torch.manual_seed(seed)
            # now calls kaiming_uniform_()
        self.conv1=nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1=nn.BatchNorm2d(out_ch)
        self.lrelu1=nn.LeakyReLU(inplace=True)
        self.conv2=nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2=nn.BatchNorm2d(out_ch)
        self.lrelu2=nn.LeakyReLU(inplace=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),

            nn.LeakyReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),

            nn.LeakyReLU(inplace=False),
        )
        
        self.conv1.reset_parameters()
        self.bn1.reset_parameters()
        self.conv2.reset_parameters()
        self.bn2.reset_parameters()
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        x3 = self.lrelu1(x2)
        x4 = self.conv2(x3)
        x5 = self.bn2(x4)
        x6 = self.lrelu2(x5)

        return x6

class Up(nn.Module):
    def __init__(self, in_ch, out_ch,init_type='default'):
        super(Up, self).__init__()
        if init_type == 'default':
            torch.manual_seed(seed)
            # now calls kaiming_uniform_()
        self.up_scale = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.up_scale.reset_parameters()
    def forward(self, x1, x2):
        x2 = self.up_scale(x2)

        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]

        #x2_padded = F.pad(x2.clone(), [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        #x = torch.cat([x2_padded, x1.clone()], dim=1)
        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x
    
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:  # bias가 None이 아닌지 확인
            m.bias.data.fill_(0.01)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:  # bias가 None이 아닌지 확인
            m.bias.data.fill_(0.01)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)

class DownLayer(nn.Module):
    def __init__(self, in_ch, out_ch,init_type='default'):
        super(DownLayer, self).__init__()
        if init_type == 'default':
            torch.manual_seed(seed)
            # now calls kaiming_uniform_()
        self.pool = nn.MaxPool2d(2, stride=2, padding=0)
        self.conv = DoubleConv(in_ch, out_ch)
  
    def forward(self, x):
        x = self.conv(self.pool(x))
        return x


class UpLayer(nn.Module):
    def __init__(self, in_ch, out_ch,init_type='default'):
        super(UpLayer, self).__init__()
        if init_type == 'default':
            torch.manual_seed(seed)
            # now calls kaiming_uniform_()
        self.up = Up(in_ch, out_ch)
        self.conv = DoubleConv(in_ch, out_ch)
        
        

    def forward(self, x1, x2):
        a = self.up(x1, x2)
        x = self.conv(a)
        return x


 

class CCM(nn.Module):
    def __init__(self, n=8,init_type='default', **kwargs):
        super(CCM, self).__init__()
        if init_type == 'default':
            torch.manual_seed(seed)

        self.n = n

        self.stride=(2,2)
        self.filter=(2,2)
        self.num=64
        self.conv1 = DoubleConv(2, self.num)
        self.down1 = DownLayer(self.num, self.num*2)
        self.down2 = DownLayer(self.num*2, self.num*4)
        self.down3 = DownLayer(self.num*4, self.num*8)
        self.down4 = DownLayer(self.num*8, self.num*16)
        self.up1 = UpLayer(self.num*16, self.num*8)
        self.up2 = UpLayer(self.num*8, self.num*4)
        self.up3 = UpLayer(self.num*4, self.num*2)
        self.up4 = UpLayer(self.num*2, self.num)
        self.last_conv = nn.Conv2d(self.num, 2, 1)   # 32차원 하나 넣어보기
        self.ln = nn.LayerNorm(self.num*16)
        self.mamba = Mamba(
                        d_model=self.num*16,
                        d_state=16,  
                        d_conv=4,    
                        expand=2,   
                    )
        self.ln.reset_parameters()
        self.last_conv.reset_parameters()
        
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data,std=0.02)
                
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data,std=0.02)
            
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.normal_(m.weight.data,std=0.02)
                
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data,std=0.02)


    def forward(self, coords):
        try:
            x1 = self.conv1(torch.cat([coords.view(1,2,640,372)], dim=-1))
        except:
            x1 = self.conv1(torch.cat([coords.view(1,2,640,320)], dim=-1))

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        for i in range(1):
            middle_feature = x5
            B, C= middle_feature.shape[:2]
            n_tokens = middle_feature.shape[2:].numel()
            img_dims = middle_feature.shape[2:]
            middle_feature_flat = middle_feature.view(B, C, n_tokens).transpose(-1, -2)
            middle_feature_flat = self.ln(middle_feature_flat)
            out = self.mamba(middle_feature_flat)
            out = out.transpose(-1, -2).view(B, C, *img_dims)
            x5 = out
        
        x1_up = self.up1(x4, x5)
        x2_up = self.up2(x3, x1_up)
        x3_up = self.up3(x2, x2_up)
        x4_up = self.up4(x1, x3_up)
        output = self.last_conv(x4_up)
        return output

class CSM(nn.Module):
    def __init__(self, n=8, init_type='default',**kwargs):
        super(CSM, self).__init__()
        if init_type == 'default':
            torch.manual_seed(seed)
        self.n = n

        self.stride=(2,2)
        self.filter=(2,2)
        self.num=64

        self.num_ch = kwargs['k']*2
        self.conv1 = DoubleConv(self.num_ch, self.num)
        self.down1 = DownLayer(self.num, self.num*2)
        self.down2 = DownLayer(self.num*2, self.num*4)
        self.down3 = DownLayer(self.num*4, self.num*8)
        self.down4 = DownLayer(self.num*8, self.num*16)
        self.up1 = UpLayer(self.num*16, self.num*8)
        self.up2 = UpLayer(self.num*8, self.num*4)
        self.up3 = UpLayer(self.num*4, self.num*2)
        self.up4 = UpLayer(self.num*2, self.num)
        self.last_conv = nn.Conv2d(self.num, self.num_ch, 1)   # 32차원 하나 넣어보기
        self.ln = nn.LayerNorm(self.num*16)
        self.softmax = torch.nn.Softmax(dim=1)
        self.mamba = Mamba(
                        d_model=self.num*16,
                        d_state=16,  
                        d_conv=4,    
                        expand=2,   
                    )
        self.ln.reset_parameters()
        
        self.last_conv.reset_parameters()
       
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data,std=0.02)
                
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data,std=0.02)
                
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.normal_(m.weight.data,std=0.02)
                
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data,std=0.02)
    

    def forward(self, coords):

        try:
            x1 = self.conv1(torch.cat([coords.view(1,30,640,372)], dim=-1))
        except:
            x1 = self.conv1(torch.cat([coords.view(1,self.num_ch,640,320)], dim=-1))

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        for i in range(1):
            middle_feature = x5
            B, C= middle_feature.shape[:2]
            n_tokens = middle_feature.shape[2:].numel()
            img_dims = middle_feature.shape[2:]
            middle_feature_flat = middle_feature.view(B, C, n_tokens).transpose(-1, -2)
            middle_feature_flat = self.ln(middle_feature_flat)
            
            out = self.mamba(middle_feature_flat)
            
            out = out.transpose(-1, -2).view(B, C, *img_dims)
            x5 = out
        
        x1_up = self.up1(x4, x5)
        x2_up = self.up2(x3, x1_up)
        x3_up = self.up3(x2, x2_up)
        x4_up = self.up4(x1, x3_up)
        output = self.last_conv(x4_up)

        return output
    
class mambalayer(nn.Module):
        def __init__(self, n=8, **kwargs):
            super(mambalayer, self).__init__()
            self.mamba = Mamba(
                        d_model=640,
                        d_state=64,  
                        d_conv=640,    
                        expand=1,   
                    )
            self.ln = torch.nn.LayerNorm(640)


        def forward(self, coords):
            for i in range(1):
                try:
                    middle_feature = coords.view(1,32,640,320).permute(0,2,3,1)
                except:
                    middle_feature = coords.view(1,40,640,320).permute(0,2,3,1)
                B, C= middle_feature.shape[:2]
         
                n_tokens = middle_feature.shape[2:].numel()
                img_dims = middle_feature.shape[2:]
                middle_feature_flat = middle_feature.contiguous().view(B, C, n_tokens).transpose(-1, -2)
                middle_feature_flat = self.ln(middle_feature_flat)
                
                out = self.mamba(middle_feature_flat)
                
                out = out.transpose(-1, -2).view(B, C, *img_dims)
                coords  = out
                
            coords = coords.permute(0,3,1,2)
            try:
                output = coords[0][:16]+coords[0][16:]
            except:
                output = coords[0][:20]+coords[0][20:]
            return output,coords
