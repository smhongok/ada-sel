import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import fastmri
from fastmri.data import transforms

from fastmri.models.unet import Unet

class NormUnet(nn.Module):
    """
    Normalized U-Net model.

    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        x = x.reshape(b, c * h * w)

        mean = x.mean(dim=1).view(b, 1, 1, 1)
        std = x.std(dim=1).view(b, 1, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if not x.shape[-1] == 2:
        #     raise ValueError("Last dimension must be 2 for complex.")

        # get shapes for unet and normalize
        # x = self.complex_to_chan_dim(x)
        # x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        x = self.unet(x)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        # x = self.unnorm(x, mean, std)
        # x = self.chan_complex_to_last_dim(x)

        return x





class ReviseUnet(nn.Module):

    def __init__(self, in_chans, out_chans, first_channels, useFc = False):
        super().__init__()
        self.bottom_param = first_channels * 16
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.first_block = ConvBlock(in_chans, self.bottom_param//16) 
        self.down1 = Down(self.bottom_param//16, self.bottom_param//8) 
        self.down2 = Down(self.bottom_param//8, self.bottom_param//4) 
        self.down3 = Down(self.bottom_param//4, self.bottom_param//2) 
        self.down4 = Down(self.bottom_param//2, self.bottom_param) 
        self.up1 = Up(self.bottom_param, self.bottom_param//2) 
        self.up2 = Up(self.bottom_param//2, self.bottom_param//4)
        self.up3 = Up(self.bottom_param//4, self.bottom_param//8) 
        self.up4 = Up(self.bottom_param//8, self.bottom_param//16) 
        self.last_block = nn.Conv2d(self.bottom_param//16, out_chans, kernel_size = 1)
        if useFc : 
            self.linear = nn.Sequential(
                nn.Linear(self.bottom_param*9,self.bottom_param*9), 
                nn.InstanceNorm1d(self.bottom_param*9),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4)
            )
            self.convTrans = nn.ConvTranspose2d(self.bottom_param,self.bottom_param,kernel_size=12,stride=6)
            self.maxpool = nn.MaxPool2d(stride=8,kernel_size=8)
        self.useFc = useFc

    def norm(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h * w)
        mean = x.mean(dim=2).view(b, c, 1, 1)
        std = x.std(dim=2).view(b, c, 1, 1)
        x = x.view(b, c, h, w)
        return (x - mean) / std, torch.mean(mean, dim=1).view(b, 1, 1, 1), torch.mean(std, dim=1).view(b, 1, 1, 1)

    def unnorm(self, x, mean, std):
        return x * std + mean

    def forward(self, input):
        # input, mean, std = self.norm(input)
        
        d1 = self.first_block(input) 
        d2 = self.down1(d1) 
        d3 = self.down2(d2) 
        d4 = self.down3(d3) 
        m0 = self.down4(d4)
        if self.useFc:
            m1 = self.maxpool(m0)
            m2 = m1.view((-1,self.bottom_param*3*3))
            m3 = self.linear(m2)
            m4 = m3.view((-1,self.bottom_param,3,3))
            m5 = self.convTrans(m4)
            u1 = self.up1(m5, d4) 
            u2 = self.up2(u1, d3) 
            u3 = self.up3(u2, d2)
            u4 = self.up4(u3, d1) 
            output = self.last_block(u4)
            # output = self.unnorm(output, mean, std)
        else:
            u1 = self.up1(m0, d4) 
            u2 = self.up2(u1, d3) 
            u3 = self.up3(u2, d2)
            u4 = self.up4(u3, d1) 
            output = self.last_block(u4)
            # output = self.unnorm(output, mean, std)
            
        return output

        # return torch.squeeze(output, dim=1) # (N, 384, 384) if out_chans is 1

class ConvBlock(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)

class Down(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_chans, out_chans)
        )

    def forward(self, x):
        return self.layers(x)

class Up(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.up = nn.ConvTranspose2d(in_chans, in_chans // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_chans, out_chans)

    def forward(self, x, concat_input):
        x = self.up(x)
        concat_output = torch.cat([concat_input, x], dim=1)
        return self.conv(concat_output)