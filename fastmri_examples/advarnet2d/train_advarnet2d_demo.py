"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import pathlib
from argparse import ArgumentParser

import pytorch_lightning as pl

from fastmri.data.mri_data import fetch_dir
from fastmri.pl_modules import FastMriDataModule
from transforms import VarNet2DDataTransform
from pl_modules import AdVarNet2DModule, VarNet2DModule
from subsample import create_mask2d_for_mask_type, create_budget_for_acquisition, calc_num_sense_lines

from typing import Tuple

from pytorch_lightning.loggers import TensorBoardLogger

def int2none(v):
    if v is None:
        return v
    if v.lower() == "none":
        return None
    else:
        return int(v)
    
def float2none(v):
    if v is None:
        return v
    if v.lower() == "none":
        return None
    else:
        return float(v)
    
def str2none(v):
    if v is None:
        return v
    if v.lower() == "none":
        return None
    else:
        return str(v)

def cli_main(args):
    pl.seed_everything(args.seed)
    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    budget = create_budget_for_acquisition(
        args.crop_size[0], args.center_fractions[0], args.accelerations[0]
    ) # calculate budget of HF sampling mask
    num_sense_lines = calc_num_sense_lines(
        args.crop_size[0], args.center_fractions[0]
    ) # calculate number of sense_lines in Tuple[int, int]
    mask = create_mask2d_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations, args.vmap_target_path, budget, args.power_vmap, args.sortvmap_fraction
    ) # 2d mask function for specific mask_type
    
    # use random masks for train transform, fixed masks for val transform, test transform
    train_transform = VarNet2DDataTransform(mask_func=mask, use_seed=False)
    val_transform = VarNet2DDataTransform(mask_func=mask)
    test_transform = VarNet2DDataTransform(mask_func=mask)
    # ptl data module - this handles data loaders
    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
    )

    # ------------
    # model
    # ------------
    if args.mask_type == "adaptive" or args.mask_type == "loupe" :      
        model = AdVarNet2DModule(
            loupe_mask=(args.mask_type=="loupe"),
            num_cascades=args.num_cascades,
            pools=args.pools,
            chans=args.chans,
            sens_pools=args.sens_pools,
            sens_chans=args.sens_chans,
            lr=args.lr,
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma,
            weight_decay=args.weight_decay,
            use_softplus=args.use_softplus,
            budget=budget,
            crop_size=args.crop_size[0],
            loss_type=args.loss_type,
            default_root_dir=args.default_root_dir,
            lamda=args.lamda,
            slope=args.slope,
            vmap_target_path=args.vmap_target_path,
            num_sense_lines = num_sense_lines,
        )
    else : 
        model = VarNet2DModule(
            num_cascades=args.num_cascades,
            pools=args.pools,
            chans=args.chans,
            sens_pools=args.sens_pools,
            sens_chans=args.sens_chans,
            lr=args.lr,
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma,
            weight_decay=args.weight_decay,
            loss_type=args.loss_type,
            default_root_dir=args.default_root_dir,
            vmap_target_path=args.vmap_target_path,
            num_sense_lines = num_sense_lines,
        )
            

    # ------------
    # trainer
    # ------------
    logger = TensorBoardLogger(save_dir=args.default_root_dir + "/lightning_logs" )
    trainer = pl.Trainer.from_argparse_args(args, logger=logger)

    # ------------
    # run
    # ------------
    if args.mode == "train":
        trainer.fit(model, datamodule=data_module)
    elif args.mode == "test":
        trainer.test(model, datamodule=data_module)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")


def build_args():
    parser = ArgumentParser()

    # basic args
    path_config = pathlib.Path("../../fastmri_dirs.yaml")
    backend = "ddp"
    num_gpus = 2 if backend == "ddp" else 1
    batch_size = 1

    # set defaults based on optional directory config
    data_path = fetch_dir("knee_path", path_config)
    default_root_dir = fetch_dir("log_path", path_config) / "varnet" / "varnet_demo"

    # client arguments
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test"),
        type=str,
        help="Operation mode",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int2none,
        help="Random seed to use. `None` for no seed.",
    )
    parser.add_argument(
        "--loss_type",
        choices=("ssim", "l2", "l1"),
        default="ssim",
        type=str,
        help="Type of loss function",
    )
    parser.add_argument(
        "--lamda",
        default=0.0,
        type=float,
        help="ratio between output_loss and prob_reg_loss",
    )

    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced_fraction", "adaptive", "loupe", "designate"),
        default="designate",
        type=str,
        help="Type of k-space mask function",
    )
    parser.add_argument(
        "--vmap_target_path",
        default=None,
        type=str2none,
        help="path fo variable density map target, used in designte mask_type",
    )
    parser.add_argument(
        "--power_vmap",
        default=1.0,
        type=float,
        help="power of variable density map while sampling from designated variable density map",
    )
    parser.add_argument(
        "--sortvmap_fraction",
        default=0.0,
        type=float,
        help="fraction of sorted variable density map while sampling from designated variable density map",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.125],
        type=float,
        help="fraction of center lines to use in LF sampling mask",
    )
    parser.add_argument(
        "--crop_size",
        nargs="+",
        default=[(320, 320)],
        type=Tuple[int, int],
        help="Input image size (crop size)",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[8],
        type=int,
        help="Acceleration rates to use for masks",
    )
    parser.add_argument(
        "--slope",
        default=10.0,
        type=float,
        help="slope in StraightThroughPolicy",
    )

    # data config
    parser = FastMriDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        data_path=data_path,  # path to fastMRI data
        mask_type="designate",  # VarNet uses equispaced mask
        challenge="multicoil",  # only multicoil implemented for VarNet
        batch_size=batch_size,  # number of samples per batch
        test_path=None,  # path for test split, overwrites data_path
        test_split="val", # default : "test"
    )

    # module config
    parser = AdVarNet2DModule.add_model_specific_args(parser)
    parser.set_defaults(
        num_cascades=5,  # number of unrolled iterations
        pools=3,  # number of pooling layers for U-Net
        chans=18,  # number of top-level channels for U-Net
        sens_pools=3,  # number of pooling layers for sense est. U-Net
        sens_chans=8,  # number of top-level channels for sense est. U-Net
        lr=0.001,  # Adam learning rate
        lr_step_size=40,  # epoch at which to decrease learning rate
        lr_gamma=0.1,  # extent to which to decrease learning rate
        weight_decay=0.0,  # weight regularization strength
        use_softplus=False,
    )

    # trainer config
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        gpus=num_gpus,  # number of gpus to use
        replace_sampler_ddp=False,  # this is necessary for volume dispatch during val
        strategy=backend,  # what distributed version to use
        seed=42,  # random seed
        deterministic=True,  # makes things slower, but deterministic
        default_root_dir=default_root_dir,  # directory for logs and checkpoints
        max_epochs=50,  # max number of epochs
    )

    args = parser.parse_args()

    # configure checkpointing in checkpoint_dir
    checkpoint_dir = pathlib.Path(args.default_root_dir) / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=pathlib.Path(args.default_root_dir) / "checkpoints",
            save_top_k=True,
            verbose=True,
            monitor="validation_loss",
            mode="min",
        )
    ]

    # set default checkpoint if one exists in our checkpoint directory
    if args.resume_from_checkpoint is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.resume_from_checkpoint = str(ckpt_list[-1])

    return args


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)


if __name__ == "__main__":
    run_cli()
