"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser
from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from utils.utils import create_dir_and_save_image, createDirectory, create_dir_and_save_npy
import time

import fastmri
from fastmri import evaluate
from fastmri.data import transforms
from fastmri.data.transforms import to_tensor

import sys
from models.advarnet2d import AdVarNet2D
from models.varnet2d import VarNet2D
from transforms import VarNet2DSample
from fastmri.pl_modules.mri_module import MriModule

from .metrics import DistributedArraySum, DistributedMetricSum
from losses import L2Loss, L1Loss

import csv

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")

class AdVarNet2DModule(MriModule):
    """
    VarNet training module.

    This can be used to train variational networks from the paper:

    A. Sriram et al. End-to-end variational networks for accelerated MRI
    reconstruction. In International Conference on Medical Image Computing and
    Computer-Assisted Intervention, 2020.

    which was inspired by the earlier paper:

    K. Hammernik et al. Learning a variational network for reconstruction of
    accelerated MRI data. Magnetic Resonance inMedicine, 79(6):3055–3071, 2018.
    """

    def __init__(
        self,
        loupe_mask: bool = False,
        num_cascades: int = 12,
        pools: int = 4,
        chans: int = 18,
        sens_pools: int = 4,
        sens_chans: int = 8,
        lr: float = 0.0003,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        num_sense_lines: Tuple[int, int] = None,
        hard_dc: bool = False,
        dc_mode: str = "simul",
        sparse_dc_gradients: bool = True,
        use_softplus: bool = False,
        crop_size: Tuple[int, int] = (128, 128),
        budget: int = 3072,
        loss_type: str = "ssim",
        default_root_dir: str = "./results",
        lamda: float = 1.0,
        slope: float = 10.0,
        vmap_target_path: str = None,
        **kwargs,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            chans: Number of channels for cascade U-Net.
            sens_pools: Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            sens_chans: Number of channels for sensitivity map U-Net.
            lr: Learning rate.
            lr_step_size: Learning rate step size.
            lr_gamma: Learning rate gamma decay.
            weight_decay: Parameter for penalizing weights norm.
            num_sense_lines: Number of low-frequency lines to use for sensitivity map
                computation, must be even or `None`. Default `None` will automatically
                compute the number from masks. Default behaviour may cause some slices to
                use more low-frequency lines than others, when used in conjunction with
                e.g. the EquispacedMaskFunc defaults.
            hard_dc: Whether to do hard DC layers instead of soft (learned).
            dc_mode: str, whether to do DC before ('first'), after ('last') or
                simultaneously ('simul') with Refinement step. Default 'simul'.
            sparse_dc_gradients: Whether to sparsify the gradients in DC by using torch.where()
                with the mask: this essentially removes gradients for the policy on unsampled rows. This should
                change nothing for the non-active VarNet.
        """
        super().__init__()
        self.save_hyperparameters()

        self.loupe_mask = loupe_mask
        self.num_cascades = num_cascades
        self.pools = pools
        self.chans = chans
        self.sens_pools = sens_pools
        self.sens_chans = sens_chans
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.num_sense_lines = num_sense_lines
        self.hard_dc = hard_dc
        self.dc_mode = dc_mode
        self.sparse_dc_gradients = sparse_dc_gradients
        self.use_softplus = use_softplus
        self.crop_size = crop_size
        self.budget = budget
        self.loss_type = loss_type
        self.default_root_dir = default_root_dir
        self.lamda = lamda
        self.slope = slope

        # logging functions
        self.NMSE = DistributedMetricSum()
        self.SSIM = DistributedMetricSum()
        self.PSNR = DistributedMetricSum()
        self.varNMSE = DistributedMetricSum()
        self.varSSIM = DistributedMetricSum()
        self.varPSNR = DistributedMetricSum()
        self.ValLoss = DistributedMetricSum()
        self.ValReconLoss = DistributedMetricSum()
        self.ValSamplingLoss = DistributedMetricSum()
        self.ValProbRegLoss = DistributedMetricSum()
        self.TotExamples = DistributedMetricSum()
        self.TotSliceExamples = DistributedMetricSum()
        self.ValMargDist = DistributedArraySum()
        self.ValCondEnt = DistributedMetricSum()

        self.TrainNMSE = DistributedMetricSum()
        self.TrainSSIM = DistributedMetricSum()
        self.TrainPSNR = DistributedMetricSum()
        self.TrainvarNMSE = DistributedMetricSum()
        self.TrainvarSSIM = DistributedMetricSum()
        self.TrainvarPSNR = DistributedMetricSum()
        self.TrainLoss = DistributedMetricSum()
        self.TrainReconLoss = DistributedMetricSum()
        self.TrainSamplingLoss = DistributedMetricSum()
        self.TrainProbRegLoss = DistributedMetricSum()
        self.TrainTotExamples = DistributedMetricSum()
        self.TrainTotSliceExamples = DistributedMetricSum()
        self.TrainMargDist = DistributedArraySum()
        self.TrainCondEnt = DistributedMetricSum()
        self.varnet2d = AdVarNet2D(
            loupe_mask=self.loupe_mask,
            num_cascades=self.num_cascades,
            sens_chans=self.sens_chans,
            sens_pools=self.sens_pools,
            chans=self.chans,
            pools=self.pools,
            num_sense_lines=self.num_sense_lines,
            hard_dc=self.hard_dc,
            dc_mode=self.dc_mode,
            sparse_dc_gradients=self.sparse_dc_gradients,
            use_softplus=self.use_softplus,
            crop_size=self.crop_size,
            budget=self.budget,
            default_root_dir=self.default_root_dir,
            slope = self.slope,
        )

        assert self.loss_type  == "ssim" or self.loss_type  == "l2" or self.loss_type  == "l1"
        if self.loss_type == "ssim" :
            self.loss = fastmri.SSIMLoss()
        elif self.loss_type == "l2" :
            self.loss = L2Loss(multiple=1e+9)
        elif self.loss_type == "l1" :
            self.loss = L1Loss(multiple=1e+5)
        self.prob_reg_loss = L1Loss(multiple=1e+1)
            
        # self.min_validation_loss = 1e+6
        
        if vmap_target_path is not None :
            self.vmap_target = torch.Tensor(np.load(vmap_target_path))
            self.vmap_target = self.vmap_target / self.vmap_target.sum() * self.budget
            create_dir_and_save_image(self.vmap_target / self.vmap_target.max(), self.default_root_dir + "/masks", f'vmap.png')
        else :
            self.vmap_target = None

    def forward(self, kspace, masked_kspace, mask):
        return self.varnet2d(kspace, masked_kspace, mask)

    def training_step(self, batch, batch_idx):
        output, extra_outputs = self(batch.kspace, batch.masked_kspace, batch.mask)
        
        target, output = transforms.center_crop_to_smallest(batch.target, output)
        
        # NOTE: Using max value here... 
        
        if self.loss_type == "ssim" :
            recon_loss = self.loss(output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value)
            sampling_loss = self.loss(extra_outputs["zero_fill_rss"][0].unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value)
        elif self.loss_type == "l2" or self.loss_type == "l1":
            recon_loss = self.loss(output.unsqueeze(1), target.unsqueeze(1))
            sampling_loss = self.loss(extra_outputs["zero_fill_rss"][0].unsqueeze(1), target.unsqueeze(1))
            
        if self.vmap_target is not None :
            self.vmap_target = self.vmap_target.to(extra_outputs["prob_mask"][0].get_device()) # convert device to cuda:0
            prob_reg_loss = self.prob_reg_loss(extra_outputs["prob_mask"][0].unsqueeze(0), self.vmap_target.unsqueeze(0))
        else :
            prob_reg_loss = torch.zeros_like(recon_loss)
        
        loss = recon_loss + self.lamda * prob_reg_loss
        
        create_dir_and_save_image(extra_outputs["acquire_mask"][0], self.default_root_dir + "/masks", f'e{self.current_epoch}_s{self.global_step}_mask.png')
        create_dir_and_save_image(extra_outputs["prob_mask"][0].detach().cpu(), self.default_root_dir + "/probs", f'e{self.current_epoch}_s{self.global_step}_prob.png')
        
        self.log("train_loss", loss)        
        self.log("train_recon_loss", recon_loss)
        self.log("train_sampling_loss", sampling_loss)
        self.log("train_prob_reg_loss", prob_reg_loss)
        
        # Return same stuff as on validation step here
        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output,
            "target": target,
            "loss": loss,
            "sampling_loss": sampling_loss,
            "recon_loss": recon_loss,
            "prob_reg_loss": prob_reg_loss,
            "extra_outputs": extra_outputs,
        }

    def training_step_end(self, train_logs):
        # check inputs
        for k in (
            "batch_idx",
            "fname",
            "slice_num",
            "max_value",
            "output",
            "target",
            "loss",
            "sampling_loss",
            "recon_loss",
            "prob_reg_loss",
            "extra_outputs",
        ):
            if k not in train_logs.keys():
                raise RuntimeError(
                    f"Expected key {k} in dict returned by training_step."
                )
        if train_logs["output"].ndim == 2:
            train_logs["output"] = train_logs["output"].unsqueeze(0)
        elif train_logs["output"].ndim != 3:
            raise RuntimeError("Unexpected output size from training_step.")
        if train_logs["target"].ndim == 2:
            train_logs["target"] = train_logs["target"].unsqueeze(0)
        elif train_logs["target"].ndim != 3:
            raise RuntimeError("Unexpected output size from training_step.")

        # compute evaluation metrics
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()
        for i, fname in enumerate(train_logs["fname"]):
            slice_num = int(train_logs["slice_num"][i].cpu())
            maxval = train_logs["max_value"][i].cpu().numpy()
            output = train_logs["output"][i].detach().cpu().numpy()
            target = train_logs["target"][i].cpu().numpy()

            mse_vals[fname][slice_num] = torch.tensor(
                evaluate.mse(target, output)
            ).view(1)
            target_norms[fname][slice_num] = torch.tensor(
                evaluate.mse(target, np.zeros_like(target))
            ).view(1)
            ssim_vals[fname][slice_num] = torch.tensor(
                evaluate.ssim(target[None, ...], output[None, ...], maxval=maxval)
            ).view(1)
            max_vals[fname] = maxval

        return {
            "loss": train_logs["loss"],
            "recon_loss": train_logs["recon_loss"],
            "sampling_loss": train_logs["sampling_loss"],
            "prob_reg_loss": train_logs["prob_reg_loss"],
            "mse_vals": mse_vals,
            "target_norms": target_norms,
            "ssim_vals": ssim_vals,
            "max_vals": max_vals,
        }

    def validation_step(self, batch, batch_idx):
        batch: VarNet2DSample
        output, extra_outputs = self.forward(
            batch.kspace, batch.masked_kspace, batch.mask
        )        
        target, output = transforms.center_crop_to_smallest(batch.target, output)            
        
        if self.loss_type == "ssim" :
            recon_loss = self.loss(output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value)
            sampling_loss = self.loss(extra_outputs["zero_fill_rss"][0].unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value)
        elif self.loss_type == "l2" or self.loss_type == "l1":
            recon_loss = self.loss(output.unsqueeze(1), target.unsqueeze(1))
            sampling_loss = self.loss(extra_outputs["zero_fill_rss"][0].unsqueeze(1), target.unsqueeze(1))
            
        if self.vmap_target is not None :
            self.vmap_target = self.vmap_target.to(extra_outputs["prob_mask"][0].get_device()) # convert device to cuda:0
            prob_reg_loss = self.prob_reg_loss(extra_outputs["prob_mask"][0].unsqueeze(0), self.vmap_target.unsqueeze(0))
        else :
            prob_reg_loss = torch.zeros_like(recon_loss)
        
        loss = recon_loss + self.lamda * prob_reg_loss
        
        create_dir_and_save_npy(extra_outputs["prob_mask"][0].detach().cpu().numpy(), self.default_root_dir + "/probs_npy", f'e{self.current_epoch}_prob.npy')

        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output,
            "target": target,
            "val_loss": loss,
            "val_recon_loss": recon_loss,
            "val_sampling_loss": sampling_loss,
            "val_prob_reg_loss": prob_reg_loss,
            "extra_outputs": extra_outputs,
        }

    def validation_step_end(self, val_logs):
        # check inputs
        for k in (
            "batch_idx",
            "fname",
            "slice_num",
            "max_value",
            "output",
            "target",
            "val_loss",
            "val_recon_loss",
            "val_sampling_loss",
            "val_prob_reg_loss",
        ):
            if k not in val_logs.keys():
                raise RuntimeError(
                    f"Expected key {k} in dict returned by validation_step."
                )
        if val_logs["output"].ndim == 2:
            val_logs["output"] = val_logs["output"].unsqueeze(0)
        elif val_logs["output"].ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")
        if val_logs["target"].ndim == 2:
            val_logs["target"] = val_logs["target"].unsqueeze(0)
        elif val_logs["target"].ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")

        # pick a set of images to log if we don't have one already
        if self.val_log_indices is None:
            self.val_log_indices = list(
                np.random.RandomState(seed=42).choice(
                    a = np.arange(4, len(self.trainer.val_dataloaders[0])),
                    size = self.num_log_images
                )
            )
            self.val_log_indices.extend(np.arange(4))
        
        # compute evaluation metrics
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()
        for i, fname in enumerate(val_logs["fname"]):
            slice_num = int(val_logs["slice_num"][i].cpu())
            maxval = val_logs["max_value"][i].cpu().numpy()
            output = val_logs["output"][i].cpu().numpy()
            target = val_logs["target"][i].cpu().numpy()

            mse_vals[fname][slice_num] = torch.tensor(
                evaluate.mse(target, output)
            ).view(1)
            target_norms[fname][slice_num] = torch.tensor(
                evaluate.mse(target, np.zeros_like(target))
            ).view(1)
            ssim_vals[fname][slice_num] = torch.tensor(
                evaluate.ssim(target[None, ...], output[None, ...], maxval=maxval)
            ).view(1)
            max_vals[fname] = maxval
        
        return {
            "val_loss": val_logs["val_loss"],
            "val_recon_loss": val_logs["val_recon_loss"],
            "val_sampling_loss": val_logs["val_sampling_loss"],
            "val_prob_reg_loss": val_logs["val_prob_reg_loss"],
            "mse_vals": mse_vals,
            "target_norms": target_norms,
            "ssim_vals": ssim_vals,
            "max_vals": max_vals,
        }

    def training_epoch_end(self, train_logs):
        losses = []
        recon_losses = []
        sampling_losses = []
        prob_reg_losses = []
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()

        # use dict updates to handle duplicate slices
        for train_log in train_logs:
            losses.append(train_log["loss"].data.view(-1))
            recon_losses.append(train_log["recon_loss"].data.view(-1))
            sampling_losses.append(train_log["sampling_loss"].data.view(-1))
            prob_reg_losses.append(train_log["prob_reg_loss"].data.view(-1))

            for k in train_log["mse_vals"].keys():
                mse_vals[k].update(train_log["mse_vals"][k])
            for k in train_log["target_norms"].keys():
                target_norms[k].update(train_log["target_norms"][k])
            for k in train_log["ssim_vals"].keys():
                ssim_vals[k].update(train_log["ssim_vals"][k])
            for k in train_log["max_vals"]:
                max_vals[k] = train_log["max_vals"][k]

        # check to make sure we have all files in all metrics
        assert (
            mse_vals.keys()
            == target_norms.keys()
            == ssim_vals.keys()
            == max_vals.keys()
        )

        # apply means across image volumes
        metrics = {"nmse": 0, "ssim": 0, "psnr": 0}
        nmse_list = []
        ssim_list = []
        psnr_list = []
        local_examples = 0
        for fname in mse_vals.keys():
            local_examples = local_examples + 1
            mse_val = torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals[fname].items()])
            )
            target_norm = torch.mean(
                torch.cat([v.view(-1) for _, v in target_norms[fname].items()])
            )
            nmse = mse_val / target_norm
            metrics["nmse"] = metrics["nmse"] + nmse
            nmse_list.append(nmse)
            psnr = 20 * torch.log10(
                    torch.tensor(
                        max_vals[fname], dtype=mse_val.dtype, device=mse_val.device
                    )
                ) - 10 * torch.log10(mse_val)
            metrics["psnr"] = metrics["psnr"] + psnr
            psnr_list.append(psnr)
            ssim = torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()])
            )
            metrics["ssim"] = metrics["ssim"] + ssim
            ssim_list.append(ssim)

        # reduce across ddp via sum
        metrics["nmse"] = self.TrainNMSE(metrics["nmse"])
        metrics["ssim"] = self.TrainSSIM(metrics["ssim"])
        metrics["psnr"] = self.TrainPSNR(metrics["psnr"])
        metrics["varnmse"] = self.TrainvarNMSE(np.var(nmse_list))
        metrics["varssim"] = self.TrainvarSSIM(np.var(ssim_list))
        metrics["varpsnr"] = self.TrainvarPSNR(np.var(psnr_list))
        tot_examples = self.TrainTotExamples(torch.tensor(local_examples))
        train_loss = self.TrainLoss(torch.sum(torch.cat(losses)))
        train_recon_loss = self.TrainReconLoss(torch.sum(torch.cat(recon_losses)))
        train_sampling_loss = self.TrainSamplingLoss(torch.sum(torch.cat(sampling_losses)))
        train_prob_reg_loss = self.TrainProbRegLoss(torch.sum(torch.cat(prob_reg_losses)))
        tot_slice_examples = self.TrainTotSliceExamples(
            torch.tensor(len(losses), dtype=torch.float)
        )

        self.log("training_loss", train_loss / tot_slice_examples, prog_bar=True)
        self.log("training_recon_loss", train_recon_loss / tot_slice_examples)
        self.log("training_sampling_loss", train_sampling_loss / tot_slice_examples)
        self.log("training_prob_reg_loss", train_prob_reg_loss / tot_slice_examples)
        for metric, value in metrics.items():
            self.log(f"train_metrics/{metric}", value / tot_examples)

    def validation_epoch_end(self, val_logs):
        # aggregate losses
        losses = []
        recon_losses = []
        sampling_losses = []
        prob_reg_losses = []
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()

        # use dict updates to handle duplicate slices
        for val_log in val_logs:
            losses.append(val_log["val_loss"].view(-1))
            recon_losses.append(val_log["val_recon_loss"].data.view(-1))
            sampling_losses.append(val_log["val_sampling_loss"].data.view(-1))
            prob_reg_losses.append(val_log["val_prob_reg_loss"].data.view(-1))

            for k in val_log["mse_vals"].keys():
                mse_vals[k].update(val_log["mse_vals"][k])
            for k in val_log["target_norms"].keys():
                target_norms[k].update(val_log["target_norms"][k])
            for k in val_log["ssim_vals"].keys():
                ssim_vals[k].update(val_log["ssim_vals"][k])
            for k in val_log["max_vals"]:
                max_vals[k] = val_log["max_vals"][k]

        # check to make sure we have all files in all metrics
        assert (
            mse_vals.keys()
            == target_norms.keys()
            == ssim_vals.keys()
            == max_vals.keys()
        )

        # apply means across image volumes
        metrics = {"nmse": 0, "ssim": 0, "psnr": 0}
        nmse_list = []
        ssim_list = []
        psnr_list = []
        local_examples = 0
        for fname in mse_vals.keys():
            local_examples = local_examples + 1
            mse_val = torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals[fname].items()])
            )
            target_norm = torch.mean(
                torch.cat([v.view(-1) for _, v in target_norms[fname].items()])
            )
            nmse = mse_val / target_norm
            metrics["nmse"] = metrics["nmse"] + nmse
            nmse_list.append(nmse)
            psnr = 20 * torch.log10(
                    torch.tensor(
                        max_vals[fname], dtype=mse_val.dtype, device=mse_val.device
                    )
                ) - 10 * torch.log10(mse_val)
            metrics["psnr"] = metrics["psnr"] + psnr
            psnr_list.append(psnr)
            ssim = torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()])
            )
            metrics["ssim"] = metrics["ssim"] + ssim
            ssim_list.append(ssim)
        
        # reduce across ddp via sum
        metrics["nmse"] = self.NMSE(metrics["nmse"])
        metrics["ssim"] = self.SSIM(metrics["ssim"])
        metrics["psnr"] = self.PSNR(metrics["psnr"])
        metrics["varnmse"] = self.varNMSE(np.var(nmse_list))
        metrics["varssim"] = self.varSSIM(np.var(ssim_list))
        metrics["varpsnr"] = self.varPSNR(np.var(psnr_list))
        tot_examples = self.TotExamples(torch.tensor(local_examples))
        val_loss = self.ValLoss(torch.sum(torch.cat(losses)))
        val_recon_loss = self.ValReconLoss(torch.sum(torch.cat(recon_losses)))
        val_sampling_loss = self.ValSamplingLoss(torch.sum(torch.cat(sampling_losses)))
        val_prob_reg_loss = self.ValProbRegLoss(torch.sum(torch.cat(prob_reg_losses)))
        tot_slice_examples = self.TotSliceExamples(
            torch.tensor(len(losses), dtype=torch.float)
        )

        self.log("validation_loss", val_loss / tot_slice_examples, prog_bar=True)
        self.log("validation_recon_loss", val_recon_loss / tot_slice_examples)
        self.log("validation_sampling_loss", val_sampling_loss / tot_slice_examples)
        self.log("validation_prob_reg_loss", val_prob_reg_loss / tot_slice_examples)
        for metric, value in metrics.items():
            self.log(f"val_metrics/{metric}", value / tot_examples)
        
        save_dir = self.default_root_dir + "/metrics"
        createDirectory(save_dir)
        save_path = save_dir + f"/metrics_epoch={self.current_epoch}.csv"
        f = open(save_path,'w', newline='')
        wr = csv.writer(f)
        wr.writerow(["file_name", "slice_num", "ssim", "psnr", "nmse"])
        
        for fname in mse_vals.keys():
            for slice_num in mse_vals[fname].keys():
                mse_val = mse_vals[fname][slice_num].view(-1)
                target_norm = target_norms[fname][slice_num].view(-1)
                nmse = mse_val / target_norm
                
                psnr = 20 * torch.log10(
                        torch.tensor(
                            max_vals[fname], dtype=mse_val.dtype, device=mse_val.device
                        )
                    ) - 10 * torch.log10(mse_val)
                
                ssim = ssim_vals[fname][slice_num].view(-1)
                
                wr.writerow([fname, slice_num, ssim.item(), psnr.item(), nmse.item()])

    def test_step(self, batch, batch_idx):
        batch: VarNet2DSample
        output, extra_outputs = self.forward(
            batch.kspace, batch.masked_kspace, batch.mask
        )
        target, output = transforms.center_crop_to_smallest(batch.target, output)

        if self.loss_type == "ssim" :
            recon_loss = self.loss(output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value)
            sampling_loss = self.loss(extra_outputs["zero_fill_rss"][0].unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value)
            loss = recon_loss
        elif self.loss_type == "l2" or self.loss_type == "l1":
            recon_loss = self.loss(output.unsqueeze(1), target.unsqueeze(1))
            sampling_loss = self.loss(extra_outputs["zero_fill_rss"][0].unsqueeze(1), target.unsqueeze(1))
            loss = recon_loss
        
        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output,
            "target": target,
            "val_loss": loss,
            "val_recon_loss": recon_loss,
            "val_sampling_loss": sampling_loss,
            "extra_outputs": extra_outputs,
        }
        
    def test_step_end(self, test_logs):
        # check inputs
        for k in (
            "batch_idx",
            "fname",
            "slice_num",
            "max_value",
            "output",
            "target",
            "val_loss",
            "val_recon_loss",
            "val_sampling_loss",
        ):
            if k not in test_logs.keys():
                raise RuntimeError(
                    f"Expected key {k} in dict returned by validation_step."
                )
        if test_logs["output"].ndim == 2:
            test_logs["output"] = test_logs["output"].unsqueeze(0)
        elif test_logs["output"].ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")
        if test_logs["target"].ndim == 2:
            test_logs["target"] = test_logs["target"].unsqueeze(0)
        elif test_logs["target"].ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")

        # log images to tensorboard :
        for i in range(test_logs["target"].shape[0]) :
            file_name = test_logs["fname"][i]
            slice_num = test_logs["slice_num"][i]
            target = test_logs["target"][i].unsqueeze(0)
            output = test_logs["output"][i].unsqueeze(0)
            error = torch.abs(target - output)
            # error = error / error.max()
            error = torch.clip((error / target.max()) * 5, max=1.0)
            output = output / output.max()
            target = target / target.max()
            
            save_dir = self.default_root_dir + "/final_output_imgs"
            create_dir_and_save_image(target, save_dir, f"{file_name}_s{slice_num}_target.png")
            create_dir_and_save_image(output, save_dir, f"{file_name}_s{slice_num}_output.png")
            create_dir_and_save_image(error, save_dir, f"{file_name}_s{slice_num}_error.png")

        # compute evaluation metrics
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()
        for i, fname in enumerate(test_logs["fname"]):
            slice_num = int(test_logs["slice_num"][i].cpu())
            maxval = test_logs["max_value"][i].cpu().numpy()
            output = test_logs["output"][i].cpu().numpy()
            target = test_logs["target"][i].cpu().numpy()

            mse_vals[fname][slice_num] = torch.tensor(
                evaluate.mse(target, output)
            ).view(1)
            target_norms[fname][slice_num] = torch.tensor(
                evaluate.mse(target, np.zeros_like(target))
            ).view(1)
            ssim_vals[fname][slice_num] = torch.tensor(
                evaluate.ssim(target[None, ...], output[None, ...], maxval=maxval)
            ).view(1)
            max_vals[fname] = maxval

        return {
            "val_loss": test_logs["val_loss"],
            "val_recon_loss": test_logs["val_recon_loss"],
            "val_sampling_loss": test_logs["val_sampling_loss"],
            "mse_vals": mse_vals,
            "target_norms": target_norms,
            "ssim_vals": ssim_vals,
            "max_vals": max_vals,
        }
        
    def test_epoch_end(self, test_logs):
        # aggregate losses
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()

        # use dict updates to handle duplicate slices
        for test_log in test_logs:
            for k in test_log["mse_vals"].keys():
                mse_vals[k].update(test_log["mse_vals"][k])
            for k in test_log["target_norms"].keys():
                target_norms[k].update(test_log["target_norms"][k])
            for k in test_log["ssim_vals"].keys():
                ssim_vals[k].update(test_log["ssim_vals"][k])
            for k in test_log["max_vals"]:
                max_vals[k] = test_log["max_vals"][k]
        
        save_path = self.default_root_dir + f"/metrics_final.csv"
        f = open(save_path,'w', newline='')
        wr = csv.writer(f)
        wr.writerow(["file_name", "slice_num", "ssim", "psnr", "nmse"])
        
        for fname in mse_vals.keys():
            for slice_num in mse_vals[fname].keys():
                mse_val = mse_vals[fname][slice_num].view(-1)
                target_norm = target_norms[fname][slice_num].view(-1)
                nmse = mse_val / target_norm
                
                psnr = 20 * torch.log10(
                        torch.tensor(
                            max_vals[fname], dtype=mse_val.dtype, device=mse_val.device
                        )
                    ) - 10 * torch.log10(mse_val)
                
                ssim = ssim_vals[fname][slice_num].view(-1)
                
                wr.writerow([fname, slice_num, ssim.item(), psnr.item(), nmse.item()])

    def configure_optimizers(self):
        # This needs to be a class attribute for storing of gradients workaround
        self.optim = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            self.optim, self.lr_step_size, self.lr_gamma
        )

        return [self.optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # param overwrites

        # network params
        parser.add_argument(
            "--num_cascades",
            default=12,
            type=int,
            help="Number of VarNet cascades",
        )
        parser.add_argument(
            "--pools",
            default=4,
            type=int,
            help="Number of U-Net pooling layers in VarNet blocks",
        )
        parser.add_argument(
            "--chans",
            default=18,
            type=int,
            help="Number of channels for U-Net in VarNet blocks",
        )
        parser.add_argument(
            "--sens_pools",
            default=4,
            type=int,
            help="Number of pooling layers for sense map estimation U-Net in VarNet",
        )
        parser.add_argument(
            "--sens_chans",
            default=8,
            type=float,
            help="Number of channels for sense map estimation U-Net in VarNet",
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.0003, type=float, help="Adam learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma",
            default=0.1,
            type=float,
            help="Extent to which step size should be decreased",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )
        parser.add_argument(
            "--use_softplus",
            default=False,
            type=str2bool,
            help="Use Softplus for Policy",
        )

        return parser
