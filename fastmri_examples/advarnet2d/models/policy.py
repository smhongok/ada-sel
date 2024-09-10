"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import functools
import operator
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import fastmri
from fastmri.models.unet import Unet
from models.unet import NormUnet, ReviseUnet

class LOUPEPolicy2D(nn.Module):
    """
    LOUPE policy model.
    """

    def __init__(
        self,
        budget: int,
        crop_size: Tuple[int, int] = (128, 128),        
        use_softplus: bool = True,
        slope: float = 2, # 10
        sampler_detach_mask: bool = False,
        straight_through_slope: float = 12, # 10
        fix_sign_leakage: bool = True,
        st_clamp: bool = False,
    ):
        super().__init__()
        self.use_softplus = use_softplus
        self.slope = slope
        self.straight_through_slope = straight_through_slope
        self.fix_sign_leakage = fix_sign_leakage
        self.st_clamp = st_clamp

        if use_softplus:
            # Softplus will be applied
            self.sampler = nn.Parameter(
                torch.normal(
                    torch.ones((1, crop_size[0], crop_size[1])), torch.ones((1, crop_size[0], crop_size[1])) / 10
                )
            )
        else:
            # Sigmoid will be applied
            self.sampler = nn.Parameter(torch.zeros((1, crop_size[0], crop_size[1])))

        self.binarizer = ThresholdSigmoidMask2D.apply
        self.budget = budget
        self.sampler_detach_mask = sampler_detach_mask

    def forward(self, kspace: torch.Tensor, mask: torch.Tensor):
        B, M, H, W, C = kspace.shape  # batch, coils, height, width, complex
        # Reshape to [B, W]
        sampler_output = self.sampler.expand(B, H, W)
        mask = mask.reshape(B, H, W)
        
        if self.use_softplus:
            # Softplus to make positive
            sampler_output = F.softplus(sampler_output, beta=self.slope)
            # Make sure max probability is 1, but ignore already sampled rows for this normalisation, since
            #  those get masked out later anyway.
            prob_mask = sampler_output / torch.max(
                (~mask*sampler_output).reshape(B, -1), dim=1
            )
        else:
            # Sigmoid to make positive
            prob_mask = torch.sigmoid(self.slope * sampler_output)
            
        # Mask out already sampled rows
        prob_mask = prob_mask * ~mask
        assert len(prob_mask.shape) == 3
        
        # Take out zero (masked) probabilities, since we don't want to include
        # those in the normalisation
        nonzero_idcs = (mask == 0).nonzero(as_tuple=True)
        probs_to_norm = prob_mask[nonzero_idcs].reshape(B, -1)            
        
        # Rescale probabilities to desired sparsity.
        normed_probs = self.rescale_probs(probs_to_norm) # B x (H x W)
        # Reassign to original array
        prob_mask[nonzero_idcs] = normed_probs.flatten()
        
        # Binarize the mask
        bin_mask = self.binarizer(
            prob_mask, self.straight_through_slope, self.st_clamp
        )
        
        return bin_mask, prob_mask

    def do_acquisition(
        self,
        kspace: torch.Tensor,
        mask: torch.Tensor,
    ):
        B, M, H, W, C = kspace.shape  # batch, coils, height, width, complex
        
        # BCHW --> BW --> B1HW1
        acquisitions, prob_mask = self(kspace, mask)
        acquisitions = acquisitions.reshape(B, 1, H, W, 1)
        prob_mask = prob_mask.reshape(B, 1, H, W, 1)

        acquisitions = mask + acquisitions # B1HW1        
        masked_kspace = acquisitions * kspace # BMHWC
        
        if self.sampler_detach_mask:
            acquisitions = acquisitions.detach()
            
        # Note that since masked_kspace = mask * kspace, this kspace_pred will leak sign information.
        if self.fix_sign_leakage:
            fix_sign_leakage_mask = torch.where(
                torch.bitwise_and(kspace < 0.0, acquisitions == 0.0), -1.0, 1.0
            )
            masked_kspace = masked_kspace * fix_sign_leakage_mask
        
        return acquisitions, masked_kspace, prob_mask

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
            dim=1, keepdim=True
        )

    def rescale_probs(self, batch_x: torch.Tensor):
        """
        Rescale Probability Map
        given a prob map x, rescales it so that it obtains the desired sparsity,
        specified by self.budget and the image size.

        if mean(x) > sparsity, then rescaling is easy: x' = x * sparsity / mean(x)
        if mean(x) < sparsity, one can basically do the same thing by rescaling
                                (1-x) appropriately, then taking 1 minus the result.
        """
        batch_size, W = batch_x.shape
        sparsity = self.budget / W
        ret = []
        for i in range(batch_size):
            x = batch_x[i : i + 1]
            xbar = torch.mean(x)
            r = sparsity / (xbar)
            beta = (1 - sparsity) / (1 - xbar)

            # compute adjustment
            le = torch.le(r, 1).float()
            ret.append(le * x * r + (1 - le) * (1 - (1 - x) * beta))

        return torch.cat(ret, dim=0)

class StraightThroughPolicy2D(nn.Module):
    """
    Policy model for active acquisition.
    """

    def __init__(
        self,
        budget: int,
        crop_size: Tuple[int, int] = (128, 128),
        slope: float = 2, # 10
        sampler_detach_mask: bool = False,
        use_softplus: bool = False,
        straight_through_slope: float = 12, # 10
        fix_sign_leakage: bool = True,
        st_clamp: bool = False,
        fc_size: int = 256,
        drop_prob: float = 0.0,
        num_fc_layers: int = 3,
        activation: str = "leakyrelu",
    ):
        super().__init__()
        
        # self.sampler = Unet(
        #     chans = 16,
        #     num_pool_layers = 4,
        #     in_chans = 2,
        #     out_chans = 1
        # )
        
        # self.sampler = NormUnet(
        #     chans = 32,
        #     num_pools = 4,
        #     in_chans = 2,
        #     out_chans = 1
        # )
        
        self.sampler = ReviseUnet(
            in_chans = 2,
            out_chans = 1,
            first_channels = 32,
        )

        self.binarizer = ThresholdSigmoidMask2D.apply
        self.slope = slope
        self.straight_through_slope = straight_through_slope
        self.budget = budget
        self.sampler_detach_mask = sampler_detach_mask
        self.use_softplus = use_softplus
        self.fix_sign_leakage = fix_sign_leakage
        self.st_clamp = st_clamp
        self.fc_size = fc_size
        self.drop_prob = drop_prob
        self.num_fc_layers = num_fc_layers
        self.activation = activation

    def forward(self, kspace_pred: torch.Tensor, mask: torch.Tensor):
        B, C, H, W = kspace_pred.shape

        sampler_output = self.sampler(kspace_pred) # 1 channel prob mask image (B x 1 x H x W)
        sampler_output = sampler_output.reshape(B, H, W)
        mask = mask.reshape(B, H, W)
        
        if self.use_softplus:
            # Softplus to make positive
            sampler_output = F.softplus(sampler_output, beta=self.slope)
            # Make sure max probability is 1, but ignore already sampled rows for this normalisation, since
            #  those get masked out later anyway.
            prob_mask = sampler_output / torch.max(
                (~mask*sampler_output).reshape(B, -1), dim=1
            )
        else:
            prob_mask = torch.sigmoid(self.slope * sampler_output)\
            
        # Mask out already sampled rows
        prob_mask = prob_mask * ~mask
        assert len(prob_mask.shape) == 3
        
        # Take out zero (masked) probabilities, since we don't want to include
        # those in the normalisation
        nonzero_idcs = (mask == 0).nonzero(as_tuple=True)
        probs_to_norm = prob_mask[nonzero_idcs].reshape(B, -1)            
        
        # Rescale probabilities to desired sparsity.
        normed_probs = self.rescale_probs(probs_to_norm) # B x (H x W)
        # Reassign to original array
        prob_mask[nonzero_idcs] = normed_probs.flatten()

        # Binarize the mask
        bin_mask = self.binarizer(
            prob_mask, self.straight_through_slope, self.st_clamp
        )
        
        return bin_mask, prob_mask

    def do_acquisition(
        self,
        kspace: torch.Tensor,
        kspace_pred: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ):
        B, M, H, W, C = kspace.shape  # batch, coils, height, width, complex
        # BMHWC --> BHWC --> BCHW
        current_recon = (
            self.sens_reduce(kspace_pred, sens_maps).squeeze(1).permute(0, 3, 1, 2)
        )
        
        # BCHW --> BW --> B1HW1
        acquisitions, prob_mask = self(current_recon, mask)
        acquisitions = acquisitions.reshape(B, 1, H, W, 1)
        prob_mask = prob_mask.reshape(B, 1, H, W, 1)

        mask = mask + acquisitions # B1HW1        
        masked_kspace = mask * kspace # BMHWC
        
        if self.sampler_detach_mask:
            mask = mask.detach()
            
        # Note that since masked_kspace = mask * kspace, this kspace_pred will leak sign information.
        if self.fix_sign_leakage:
            fix_sign_leakage_mask = torch.where(
                torch.bitwise_and(kspace < 0.0, mask == 0.0), -1.0, 1.0
            )
            masked_kspace = masked_kspace * fix_sign_leakage_mask
            
        return mask, masked_kspace, prob_mask

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
            dim=1, keepdim=True
        )

    def rescale_probs(self, batch_x: torch.Tensor):
        """
        Rescale Probability Map
        given a prob map x, rescales it so that it obtains the desired sparsity,
        specified by self.budget and the image size.

        if mean(x) > sparsity, then rescaling is easy: x' = x * sparsity / mean(x)
        if mean(x) < sparsity, one can basically do the same thing by rescaling
                                (1-x) appropriately, then taking 1 minus the result.
        """

        batch_size, W = batch_x.shape
        sparsity = self.budget / W
        ret = []
        for i in range(batch_size):
            x = batch_x[i : i + 1]
            xbar = torch.mean(x)
            r = sparsity / (xbar)
            beta = (1 - sparsity) / (1 - xbar)

            # compute adjustment
            le = torch.le(r, 1).float()
            ret.append(le * x * r + (1 - le) * (1 - (1 - x) * beta))

        return torch.cat(ret, dim=0)

class ThresholdSigmoidMask2D(Function):
    def __init__(self):
        """
        Straight through estimator.
        The forward step stochastically binarizes the probability mask.
        The backward step estimate the non differentiable > operator using sigmoid with large slope (10).
        """
        super(ThresholdSigmoidMask2D, self).__init__()

    @staticmethod
    def forward(ctx, inputs, slope, clamp):
        batch_size, H, W = inputs.shape
        probs = []
        results = []

        for i in range(batch_size):
            flatten_x = inputs[i].reshape(-1)
            count = 0
            while True:
                prob = flatten_x.new(flatten_x.size()).uniform_()
                result = (flatten_x > prob).float()
                if torch.isclose(torch.mean(result), torch.mean(flatten_x), atol=1e-3):
                    break
                count += 1
                if count > 1000:
                    print(torch.mean(prob), torch.mean(result), torch.mean(flatten_x))
                    raise RuntimeError(
                        "Rejection sampled exceeded number of tries. Probably this means all "
                        "sampling probabilities are 1 or 0 for some reason, leading to divide "
                        "by zero in rescale_probs()."
                    )
            probs.append(prob)
            results.append(result)
            
        results = torch.cat(results, dim=0).reshape(batch_size, H, W)
        probs = torch.cat(probs, dim=0).reshape(batch_size, H, W)

        slope = torch.tensor(slope, requires_grad=False)
        ctx.clamp = clamp
        ctx.save_for_backward(inputs, probs, slope)
        return results

    @staticmethod
    def backward(ctx, grad_output):
        input, prob, slope = ctx.saved_tensors
        if ctx.clamp:
            grad_output = F.hardtanh(grad_output)
        # derivative of sigmoid function
        current_grad = (
            slope
            * torch.exp(-slope * (input - prob))
            / torch.pow((torch.exp(-slope * (input - prob)) + 1), 2)
        )
        return current_grad * grad_output, None, None