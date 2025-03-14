# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory
from timm.utils.model_ema import ModelEma

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae
import models_bmae

from engine_pretrain import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=256,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--num_bootstrap", default=4, type=int)
    parser.add_argument("--enable_ema", action="store_true", help="Enable exponential moving average")
    parser.set_defaults(enable_ema=False)
    parser.add_argument("--ema_decay", default=0.9999, type=float, help="Decay factor for EMA")
    parser.add_argument("--use_new_feature_predictor", action="store_true", help="Use new feature predictor")
    parser.set_defaults(use_new_feature_predictor=False)
    parser.add_argument("--feature_class", default="latent", type=str, help="Feature class to predict")  # latent or hog
    parser.add_argument("--target_layer_index", default=-1, type=int, help="Target layer index for feature prediction")

    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters
    parser.add_argument(
        "--model", default="bmae_deit_tiny_patch4", type=str, metavar="MODEL", help="Name of model to train"
    )

    parser.add_argument("--input_size", default=32, type=int, help="images input size")

    parser.add_argument("--mask_ratio", default=0.75, type=float, help="Masking ratio (percentage of removed patches).")

    parser.add_argument(
        "--norm_pix_loss", action="store_true", help="Use (per-patch) normalized pixels as targets for computing loss"
    )
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)")

    parser.add_argument("--lr", type=float, default=None, metavar="LR", help="learning rate (absolute lr)")
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--min_lr", type=float, default=0.0, metavar="LR", help="lower lr bound for cyclic schedulers that hit 0"
    )

    parser.add_argument("--warmup_epochs", type=int, default=10, metavar="N", help="epochs to warmup LR")

    # Dataset parameters
    parser.add_argument("--data_path", default="./datasets01/cifar10/", type=str, help="dataset path")

    parser.add_argument("--output_dir", default="./output/pretrain_dir", help="path where to save, empty for no saving")
    parser.add_argument(
        "--save_checkpoint", default="./checkpoints/pretrain", help="path where to save, empty for no saving"
    )
    parser.add_argument("--log_dir", default="./log/pretrain_dir", help="path where to tensorboard log")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset_train = datasets.CIFAR10(args.data_path, transform=transform_train, train=True, download=True)
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if args.enable_ema:
        print("Enable EMA")
        args.num_bootstrap = 1  # At most one mode at the same time
        # args.use_new_feature_predictor = True

    # define the model
    if args.model.startswith("bmae"):
        model_init = models_bmae.__dict__[args.model]
    elif args.model.startswith("mae"):
        model_init = models_mae.__dict__[args.model]
    else:
        raise ValueError("Unknown model type: {}".format(args.model))
    # model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    model = misc.create_model(model_init, args, bootstrap=False, device=device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    model_ema = None
    if args.enable_ema:
        model_ema = ModelEma(model=model, decay=args.ema_decay, device=device)

    print(f"Start training for {args.epochs} epochs in {args.num_bootstrap} bootstraps")
    epoches_per_bootstrap = args.epochs // args.num_bootstrap
    trained_models = []

    start_time = time.time()
    if args.num_bootstrap > 1:
        print("Bootstrap training: Train Regular MAE (MAE-1)")
    elif args.enable_ema:
        print("Train MAE with EMA")

    for epoch in range(args.start_epoch, epoches_per_bootstrap):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            model_ema=model_ema,
            args=args,
        )

        if args.output_dir and ((epoch + 1) % epoches_per_bootstrap == 0 or epoch + 1 == args.epochs):
            # misc.save_model(
            #     args=args,
            #     model=model,
            #     model_without_ddp=model_without_ddp,
            #     optimizer=optimizer,
            #     loss_scaler=loss_scaler,
            #     epoch=epoch,
            #     bootstrap_idx=1,
            # )
            trained_models.append(model)

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    args.acc = 0
    if args.save_checkpoint:
        if args.enable_ema or args.num_bootstrap == 1:
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
                bootstrap_idx=1,
            )
        # misc.save_model(
        #     args=args,
        #     model=model,
        #     model_without_ddp=model_without_ddp,
        #     optimizer=optimizer,
        #     loss_scaler=loss_scaler,
        #     epoch=epoch,
        #     bootstrap_idx=1,
        # )

    bootstrapped_start_epoch = epoches_per_bootstrap
    for epoch in range(epoches_per_bootstrap, args.epochs):
        if epoch % epoches_per_bootstrap == 0:
            bootstrap_idx = epoch // epoches_per_bootstrap + 1
            bootstrapped_start_epoch = epoch
            bootstrapped_model = misc.create_model(
                model_init, args, bootstrap=True, target_encoder=trained_models[-1], device=device
            )
            model_without_ddp = bootstrapped_model
            print(f"Bootstrap {bootstrap_idx} training: Train MAE-{bootstrap_idx}")

            print("Model = %s" % str(model_without_ddp))

            param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
            optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
            loss_scaler = NativeScaler()
            print(optimizer)
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            bootstrapped_model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
            # bootstrap_idx=bootstrap_idx,
        )

        if args.output_dir and ((epoch + 1) % epoches_per_bootstrap == 0 or epoch + 1 == args.epochs):
            # misc.save_model(
            #     args=args,
            #     model=model,
            #     model_without_ddp=model_without_ddp,
            #     optimizer=optimizer,
            #     loss_scaler=loss_scaler,
            #     epoch=epoch,
            #     bootstrap_idx=bootstrap_idx,
            # )
            trained_models.append(bootstrapped_model)

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    if args.save_checkpoint:
        misc.save_model(
            args=args,
            model=model,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            bootstrap_idx=1,
        )


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.output_dir = os.path.join(args.output_dir, current_time)
        if args.log_dir:
            args.log_dir = os.path.join(args.log_dir, current_time)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        args_text = json.dumps(args.__dict__, indent=2)
        with open(os.path.join(args.output_dir, "args.txt"), "w") as f:
            f.write(args_text)
    if args.save_checkpoint:
        Path(args.save_checkpoint).mkdir(parents=True, exist_ok=True)
    main(args)
