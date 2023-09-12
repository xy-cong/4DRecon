import os
import sys
import shutil
import pickle
import argparse

import numpy as np
import os.path as osp
# pip install psbody==0.5.0

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import OneCycleLR
from utils.scheduler_utils import MultiplicativeLRSchedule, adjust_learning_rate
from torch.cuda.amp import GradScaler



from utils import utils, mesh_sampling
from utils.writer import Writer
from utils.scheduler_utils import *
from train_eval_unit import train_one_epoch, test_opt_one_epoch, recon_from_lat_vecs, interp_from_lat_vecs, analysis_one_epoch, train_debug_DeepSDF
import datasets
from pyutils import *

from loguru import logger
from omegaconf import OmegaConf


from lightning import Fabric, seed_everything


# Global setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_arap_weight(epoch, loss_config):
    arap_loss_begin = loss_config.sdf_arap_begin_epoch
    arap_loss_end   = loss_config.sdf_arap_end_epoch
    if epoch <= arap_loss_begin:
        return 1e-5
    elif epoch <= arap_loss_end:
        # 在2000到6000 epoch之间线性增长
        return 1e-5 + (epoch - arap_loss_begin) * (loss_config.sdf_asap_weight - 1e-5) / (arap_loss_end - arap_loss_begin)
    else:
        return loss_config.sdf_asap_weight

def update_config_each_epoch(config, epoch):
    config.update({
        'use_sdf_asap': epoch >= config.loss.sdf_arap_begin_epoch,
        'use_topo_loss': epoch >= config.loss.topology_PD_loss_begin_epoch,
        'use_data_term': True,
        'use_surface_normal': config.dataset.get('surface_normal', False),
        'sdf_asap_weight_epoch': get_arap_weight(epoch, config.loss),
        'use_gradient': config.loss.get('gradient_loss', False),
        'use_inter': config.loss.get('inter_constraint_loss', False),
    })
    return config

def main(config):
    #### set up and deterministic
    torch.manual_seed(config.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    #### setup directory
    # import ipdb; ipdb.set_trace()
    config.config_path = os.path.normpath(config.config_path)
    config_path_sep = config.config_path.split(os.sep)
    assert(config_path_sep[0] == 'config' and config_path_sep[-1][-5:] == '.yaml')
    config_path_sep[-1] = config_path_sep[-1][:-5]
    exp_name = '/'.join(config_path_sep[1:])
    exp_dir = f"{config.work_dir}/{exp_name}"
    log_dir = get_directory( f"{exp_dir}/log" )
    ckpt_train_dir = get_directory( f"{log_dir}/ckpt_train/{config.rep}" )
    ckpt_test_dir  = get_directory( f"{log_dir}/ckpt_test/{config.rep}" )
    logger.add(f"{log_dir}/{config.mode}_{config.rep}_log.log", format="{message}", level="DEBUG")
    logger.info(config)
    shutil.copy2(config.config_path, log_dir)

    #### load dataset
    # import ipdb; ipdb.set_trace()
    MeshSdfDataset = load_module(f"datasets.{config.dataset.module_name}", config.dataset.class_name)
    logger.info(f"load dataset: datasets.{config.dataset.module_name}.{config.dataset.class_name}")
    train_mesh_sdf_dataset = MeshSdfDataset(mode='train', rep=config.rep, **config.dataset)
    test_mesh_sdf_dataset  = MeshSdfDataset(mode='test',  rep=config.rep, **config.dataset)
    
    train_loader = DataLoader(train_mesh_sdf_dataset, 
                              batch_size=config.optimization[config.rep].batch_size, 
                              shuffle=config.dataset.shuffle, pin_memory=True, num_workers=config.num_workers)
    test_loader  = DataLoader(test_mesh_sdf_dataset,
                              batch_size=config.optimization[config.rep].batch_size,
                              shuffle=True, pin_memory=True, num_workers=config.num_workers)


    Net = load_module("models."+config.model[config.rep]["module_name"], 
                      config.model[config.rep]["class_name"])
    model = Net(config=config)
    # # TODO: add distributed learning
    # if config.distributed:
    #     model = torch.nn.DataParallel(model)
    model = model.to(device)
    logger.info(model)

    if config.optimization[config.rep].schedular == 'OneCycleLR':
        # import ipdb; ipdb.set_trace()
        schedular_cfg = config.optimization[config.rep].OneCycleLR

        optimizer_train = torch.optim.Adam([
                { "params": model.parameters(), "lr": schedular_cfg.init_lr}
            ])
        
        scheduler_train = OneCycleLR(
            optimizer_train, max_lr=schedular_cfg.max_lr, epochs=config.optimization[config.rep].num_epochs, steps_per_epoch=len(train_loader), pct_start=schedular_cfg.cyc_frac,
            anneal_strategy='cos', cycle_momentum=True, base_momentum=schedular_cfg.min_momentum, max_momentum=schedular_cfg.max_momentum,
            div_factor=(schedular_cfg.max_lr / schedular_cfg.init_lr), final_div_factor=(schedular_cfg.max_lr / schedular_cfg.fin_lr),
            three_phase=True
            )
    elif config.optimization[config.rep].schedular == 'MultiplicativeLRSchedule':
        schedular_cfg = config.optimization[config.rep].MultiplicativeLRSchedule
        
        scheduler_train = MultiplicativeLRSchedule(
            lr_group_init=[schedular_cfg.lr], gammas=schedular_cfg.gammas, milestones=schedular_cfg.milestones
            )
        optimizer_train = torch.optim.Adam([
                { "params": model.parameters(), "lr": schedular_cfg.lr}
            ])
    elif config.optimization[config.rep].schedular == 'Step':
        lat_vecs = torch.nn.Embedding(len(train_mesh_sdf_dataset), config.latent_dim, max_norm=1.0).to(device)
        torch.nn.init.normal_(
            lat_vecs.weight.data,
            0.0,
            1.0 / np.sqrt(config.latent_dim),
        )  
        scheduler_train = []
        scheduler_train.append(
            StepLearningRateSchedule(
                    initial = 0.0005,
                    interval = 1000,
                    factor = 0.5
                )
            )
        scheduler_train.append(
            StepLearningRateSchedule(
                    initial = 0.0001,
                    interval = 1000,
                    factor = 0.5
                )
            )
        optimizer_train = torch.optim.Adam(
            [
                {
                    "params": model.parameters(),
                    "lr": scheduler_train[0].get_learning_rate(0),
                },
                {
                    "params": lat_vecs.parameters(),
                    "lr": scheduler_train[1].get_learning_rate(0),
                },
            ]
        )
    else:
        raise NotImplementedError

    
    ####  setup writer and state_info
    writer = Writer(log_dir, config.mode, config.rep)

    state_info = {} # store epoch, basic training info, loss & so on.
    state_info['device'] = device
    state_info['len_train_loader'] = len(train_loader)
    state_info['len_test_loader']  = len(test_loader)
    state_info['schedular'] = config.optimization[config.rep].schedular 
    # state_info['arap_loss_OOM_cnt'] = 0


    #### train and eval
    if config.mode == 'train':
        logger.info('-' * 30 + ' train ' + '-' * 30 )
        scaler = GradScaler()

        start_epoch = 0
        end_epoch = config.optimization[config.rep].num_epochs

        if config.epoch_continue is not None:
            start_epoch = 1 + writer.load_checkpoint(f"{ckpt_train_dir}/checkpoint_{config.epoch_continue:04d}.pt",
                                                     model, optimizer_train)
            logger.info(f"continue to train from previous epoch = {start_epoch}")
            for _ in range(start_epoch*len(train_loader)):
                lr_group = adjust_learning_rate(config.optimization[config.rep].schedular, scheduler_train, optimizer_train, start_epoch)
                state_info.update({'lr': lr_group})
                # print(lr_group)

        
        if config.distributed:
            print("Torch CUDA available?", torch.cuda.is_available())
            torch.set_float32_matmul_precision("medium")
            fabric = Fabric(
                        accelerator="cuda", precision="bf16-mixed",
                        devices=4, strategy="ddp"  # ddp / fsdp
                    )
            fabric.launch()

            model, optimizer_train, scheduler_train = fabric.setup(model, optimizer_train, scheduler_train)
            # model = fabric.setup_module(model)
            # optimizer_train = fabric.setup_optimizers(optimizer_train)
            # scheduler_train = fabric.setup_optimizers(scheduler_train)
            train_loader = fabric.setup_dataloaders(train_loader)
        else:
            # model = model.to(device)
            fabric = None

        for epoch in range(start_epoch, end_epoch):
            logger.info(f"epoch = {epoch}")
            # lr = OneCycleLR_adjust_learning_rate(scheduler_train, optimizer_train)
            # lr_group = scheduler_train.get_last_lr()
            # lr_group = adjust_learning_rate(config.optimization[config.rep].schedular, scheduler_train, optimizer_train, epoch)
            
            state_info.update( {'epoch': epoch} )
            np.random.seed() # Reset numpy seed. REF: https://github.com/pytorch/pytorch/issues/5059

            config = update_config_each_epoch(config, epoch)
            if config.optimization[config.rep].schedular == 'Step':
                adjust_learning_rate('Step', scheduler_train, optimizer_train, epoch)
                state_info.update({'lr': [schedule.get_learning_rate(epoch) for schedule in scheduler_train]})
                train_debug_DeepSDF(state_info, config, lat_vecs, train_loader, model, optimizer_train, scheduler_train, scaler, writer)
            else:
                train_one_epoch(state_info, config, fabric, train_loader, model, optimizer_train, scheduler_train, scaler, writer)
                lat_vecs = None
            # save checkpoint
            if (epoch + 1) % config.log.save_epoch_interval == 0: 
                writer.save_checkpoint(f"{ckpt_train_dir}/checkpoint_{epoch:04d}.pt", epoch, model, optimizer_train, lat_vecs)
            if (epoch + 1) % config.log.save_latest_epoch_interval == 0: 
                writer.save_checkpoint(f"{ckpt_train_dir}/checkpoint_latest.pt", epoch, model, optimizer_train, lat_vecs)
        
    elif config.mode == 'interp':
        logger.info('>' * 30 + ' interp ' + '>' * 30)
        model = model.to(device)
        if config.split == 'train':
            recon_lat_vecs = torch.nn.Embedding(len(train_mesh_sdf_dataset), config.latent_dim).to(device)
            mesh_sdf_dataset = train_mesh_sdf_dataset
        elif config.split == 'test':
            recon_lat_vecs = torch.nn.Embedding(len(test_mesh_sdf_dataset), config.latent_dim).to(device)
            mesh_sdf_dataset = test_mesh_sdf_dataset
        else:
            raise ValueError("split is train or test")
        lat_vecs = None

        # import ipdb; ipdb.set_trace()

        results_dir = get_directory( f"{exp_dir}/results/{config.split}/interp_{config.rep}/{config.epoch_continue:04d}" )

        writer.load_checkpoint(f"{log_dir}/ckpt_{config.split}/{config.rep}/checkpoint_{config.epoch_continue:04d}.pt", model)

        interp_from_lat_vecs(config, results_dir, model, lat_vecs, mesh_sdf_dataset)
        
    else:
        raise NotImplementedError



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path", required=True, help='config file path')
    parser.add_argument("--continue_from", dest="epoch_continue", type=int, help='epoch of loaded ckpt, so checkpoint_{epoch:04d}.pt is loaded')
    parser.add_argument("--distributed", action='store_true', help='currently use nn.DataParallel')
    parser.add_argument("--batch_size", type=int, default=None, help='if specified, it will override batch_size in the config')
    parser.add_argument("--mode", type=str, required=True, help='train, test, recon, interp')
    parser.add_argument("--split", type=str, default='test', help='{train, test}, use train or test dataset')
    parser.add_argument("--rep", type=str, required=True, help='{mesh, sdf, all}, representation to train/test/recon')
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    OmegaConf.resolve(config)
    update_config_from_args(config, args)

    assert (config.rep in ['sdf', 'mesh']), ("currently joint training not implemented yet")
    if config.batch_size is not None:
        config.optimization[config.rep].batch_size = config.batch_size
    
    main(config)