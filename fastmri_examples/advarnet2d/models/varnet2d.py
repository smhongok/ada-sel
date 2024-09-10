"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from collections import defaultdict
from typing import Optional, Tuple

import torch
import torch.nn as nn

import fastmri
from transforms import batched_mask2d_center

from fastmri.models.policy import LOUPEPolicy, StraightThroughPolicy
from fastmri.models.varnet import NormUnet

class Sensitivity2DModel(nn.Module):
    """
    Model for learning sensitivity estimation from 2D k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        num_sense_lines: Optional[Tuple[int, int]] = None,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
            num_sense_lines: Number of low-frequency lines to use for sensitivity map
                computation, must be even or `None`. Default `None` will automatically
                compute the number from masks. Default behaviour may cause some slices to
                use more low-frequency lines than others, when used in conjunction with
                e.g. the EquispacedMaskFunc defaults.
        """
        super().__init__()

        self.num_sense_lines = num_sense_lines
        self.norm_unet = NormUnet(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
        )

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape

        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size

        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        return x / fastmri.rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def get_pad_and_num_low_freqs(
        self, mask: torch.Tensor, num_sense_lines: Optional[Tuple[int, int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # get low frequency line locations and mask them out
        squeezed_mask = mask[:, 0, :, :, 0].int()
        cent = squeezed_mask.shape[1] // 2, squeezed_mask.shape[2] // 2
        # running argmin returns the first non-zero
        left_row = torch.argmin(squeezed_mask[:, :cent[0], cent[1]].flip(1), dim=1)
        right_row = torch.argmin(squeezed_mask[:, cent[0]:, cent[1]], dim=1)
        left_col = torch.argmin(squeezed_mask[:, cent[0], :cent[1]].flip(1), dim=1)
        right_col = torch.argmin(squeezed_mask[:, cent[0], cent[1]:], dim=1)
        num_low_freqs_row = torch.max(
            2 * torch.min(left_row, right_row), torch.ones_like(left_row)
        )  # force a symmetric center unless 1
        num_low_freqs_col = torch.max(
            2 * torch.min(left_col, right_col), torch.ones_like(left_col)
        )  # force a symmetric center unless 1
        num_low_freqs = torch.stack((num_low_freqs_row, num_low_freqs_col), dim=1)

        if self.num_sense_lines is not None:  # Use pre-specified number instead
            if (num_low_freqs[:, 0] < self.num_sense_lines[0]).all() or (num_low_freqs[:, 0] < self.num_sense_lines[1]).all():
                raise RuntimeError(
                    "`num_sense_lines` cannot be greater than the actual number of "
                    "low-frequency lines in the mask: {}".format(num_low_freqs)
                )
            num_low_freqs[:, 0] = num_sense_lines[0] * torch.ones(
                mask.shape[0], dtype=mask.dtype, device=mask.device
            )
            num_low_freqs[:, 1] = num_sense_lines[1] * torch.ones(
                mask.shape[1], dtype=mask.dtype, device=mask.device
            )

        pad_row = (mask.shape[-3] - num_low_freqs[:, 0] + 1) // 2
        pad_col = (mask.shape[-2] - num_low_freqs[:, 1] + 1) // 2
        pad = torch.stack((pad_row, pad_col), dim=1)
        
        return pad, num_low_freqs

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        #print("mk", torch.mean(masked_kspace), torch.std(masked_kspace))
        pad, num_low_freqs = self.get_pad_and_num_low_freqs(mask, self.num_sense_lines)
        x = batched_mask2d_center(masked_kspace, pad, pad + num_low_freqs)
        #print("x", torch.mean(x), torch.std(x))
        #assert False
        # convert to image space
        x = fastmri.ifft2c(x)
        x, b = self.chans_to_batch_dim(x)
        # NOTE: Channel dimensions have been converted to batch dimensions, so this
        #  acts like a UNet that treats every coil as a separate image!
        # estimate sensitivities
        x = self.norm_unet(x)
        x = self.batch_chans_to_chan_dim(x, b)
        x = self.divide_root_sum_of_squares(x)

        return x

class VarNet2D(nn.Module):
    """
    A full variational network model for 2D kspace mask.
    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        num_sense_lines: Optional[Tuple[int, int]] = None,
        hard_dc: bool = False,
        dc_mode: str = "simul",
        sparse_dc_gradients: bool = True,
        default_root_dir: str = "./results",
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            num_sense_lines: Number of low-frequency lines to use for
                sensitivity map computation, must be even or `None`. Default
                `None` will automatically compute the number from masks.
                Default behaviour may cause some slices to use more
                low-frequency lines than others, when used in conjunction with
                e.g. the EquispacedMaskFunc defaults.
            hard_dc: Whether to do hard DC layers instead of soft (learned).
            dc_mode: str, whether to do DC before ('first'), after ('last') or
                simultaneously ('simul') with Refinement step. Default 'simul'.
            sparse_dc_gradients: Whether to sparsify the gradients in DC by
                using torch.where() with the mask: this essentially removes
                gradients for the policy on unsampled rows. This should change
                nothing for the non-active VarNet.
        """
        super().__init__()

        self.num_sense_lines = num_sense_lines
        self.hard_dc = hard_dc
        self.dc_mode = dc_mode

        self.sparse_dc_gradients = sparse_dc_gradients

        self.sens_net = Sensitivity2DModel(
            sens_chans, sens_pools, num_sense_lines=num_sense_lines
        )

        self.cascades = nn.ModuleList(
            [
                VarNet2DBlock(
                    NormUnet(chans, pools),
                    hard_dc=hard_dc,
                    dc_mode=dc_mode,
                    sparse_dc_gradients=sparse_dc_gradients,
                )
                for _ in range(num_cascades)
            ]
        )

    def forward(
        self, kspace: torch.Tensor, masked_kspace: torch.Tensor, mask: torch.Tensor
    ):

        extra_outputs = defaultdict(list)
        sens_maps = self.sens_net(masked_kspace, mask)
        
        extra_outputs["sense"].append(sens_maps.detach().cpu())
        extra_outputs["masks"].append(mask.detach().cpu())
        # Store current reconstruction
        current_recon = fastmri.complex_abs(
            self.sens_reduce(masked_kspace, sens_maps)
        ).squeeze(1)
        extra_outputs["recons"].append(current_recon.detach().cpu())

        zero_fill_rss = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(masked_kspace.detach())), dim=1)
        extra_outputs["zero_fill_rss"].append(zero_fill_rss.detach())
        
        extra_outputs["acquire_mask"].append(mask[0,0,:,:,0].detach().cpu())
        
        kspace_pred = masked_kspace.clone()
        for cascade in self.cascades:
            kspace_pred = cascade(
                kspace_pred, masked_kspace, mask, sens_maps, kspace=kspace
            )

            # Store current reconstruction
            current_recon = fastmri.complex_abs(
                self.sens_reduce(masked_kspace, sens_maps)
            ).squeeze(1)

            extra_outputs["recons"].append(current_recon.detach().cpu())

        # Could presumably do complex_abs(complex_rss()) instead and get same result?
        output = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)
        return output, extra_outputs

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
            dim=1, keepdim=True
        )

class VarNet2DBlock(nn.Module):
    """
    Model block for adaptive end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(
        self,
        model: nn.Module,
        inter_sens: bool = True,
        hard_dc: bool = False,
        dc_mode: str = "simul",
        sparse_dc_gradients: bool = True,
    ):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
            inter_sens: boolean, whether to do reduction and expansion using
                estimated sensitivity maps.
            hard_dc: boolean, whether to do hard DC layer instead of soft.
            dc_mode: str, whether to do DC before ('first'), after ('last') or
                simultaneously ('simul') with Refinement step. Default 'simul'.
            sparse_dc_gradients: Whether to sparsify the gradients in DC by
                using torch.where() with the mask: this essentially removes
                gradients for the policy on unsampled rows.
        """
        super().__init__()

        self.model = model
        self.inter_sens = inter_sens
        self.hard_dc = hard_dc
        self.dc_mode = dc_mode
        self.sparse_dc_gradients = sparse_dc_gradients

        if dc_mode not in ["first", "last", "simul"]:
            raise ValueError(
                "`dc_mode` must be one of 'first', 'last', or 'simul'. "
                "Not {}".format(dc_mode)
            )

        if hard_dc:
            self.dc_weight = 1
        else:
            self.dc_weight = nn.Parameter(torch.ones(1))  # type: ignore

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
            dim=1, keepdim=True
        )

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
        kspace: Optional[torch.Tensor],
    ) -> torch.Tensor:
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)

        if self.dc_mode == "first":
            # DC before Refinement, this directly puts kspace rows from ref_kspace
            #  into current_kspace if dc_weight = 1.
            if self.sparse_dc_gradients:
                current_kspace = (
                    current_kspace
                    - torch.where(mask.byte(), current_kspace - ref_kspace, zero)
                    * self.dc_weight
                )
            else:
                # Values in current_kspace that should be replaced by actual sampled
                # information
                dc_kspace = current_kspace * mask
                # don't need to multiply ref_kspace by mask because ref_kspace is 0
                # where mask is 0
                current_kspace = (
                    current_kspace - (dc_kspace - ref_kspace) * self.dc_weight
                )
        model_term = self.sens_expand(
            self.model(self.sens_reduce(current_kspace, sens_maps)),
            sens_maps,
        )

        if self.dc_mode == "first":
            return current_kspace - model_term
        elif self.dc_mode == "simul":
            # Default implementation: simultaneous DC and Refinement
            if self.sparse_dc_gradients:
                soft_dc = (
                    torch.where(mask.byte(), current_kspace - ref_kspace, zero)
                    * self.dc_weight
                )
            else:
                # Values in current_kspace that should be replaced by actual sampled
                # information
                dc_kspace = current_kspace * mask
                soft_dc = (dc_kspace - ref_kspace) * self.dc_weight
            return current_kspace - soft_dc - model_term
        elif self.dc_mode == "last":
            combined_kspace = current_kspace - model_term

            if self.sparse_dc_gradients:
                combined_kspace = (
                    combined_kspace
                    - torch.where(mask.byte(), combined_kspace - ref_kspace, zero)
                    * self.dc_weight
                )
            else:
                # Values in combined_kspace that should be replaced by actual sampled
                # information
                dc_kspace = combined_kspace * mask
                # don't need to multiply ref_kspace by mask because ref_kspace is 0
                # where mask is 0
                combined_kspace = (
                    combined_kspace - (dc_kspace - ref_kspace) * self.dc_weight
                )

            return combined_kspace
        else:
            raise ValueError(
                "`dc_mode` must be one of 'first', 'last', or 'simul'. "
                "Not {}".format(self.dc_mode)
            )
