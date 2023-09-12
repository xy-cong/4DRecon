import os
import time
import torch

from pyutils import *
from loguru import logger
from tensorboardX import SummaryWriter

class Writer():
    def __init__(self, log_dir, mode=None, rep=None):
        self.log_dir = log_dir
        if mode in ['train', 'test']:
            self.summary_writer = SummaryWriter(get_directory(f"{log_dir}/summary/{mode}/{rep}"))


    def log_state_info(self, state_info):
        state_info_str = f"it={state_info['b']:04d} "
        # import ipdb; ipdb.set_trace() 
        for k, v in state_info.items():
            if 'loss' in k:
                state_info_str = f"{state_info_str}{k}={v:.6f} "
        state_info_str = f"{state_info_str}| "
        for k, v in state_info.items():
            if 'weight' in k:
                state_info_str = f"{state_info_str}{k}={v:.6f} "
        # for k, v in state_info.items():
        #     if 'err' in k:
        #         state_info_str = f"{state_info_str}{k}={v:.6f} "
        state_info_str = f"{state_info_str}|"
        for k, v in state_info.items():
            if 'lr' in k:
                for lr in v:
                    state_info_str = f"{state_info_str}{k}={lr:.6f} "
            if 'schedular' in k:
                state_info_str = f"{state_info_str}{v} "
        logger.info(state_info_str)


    def log_summary(self, state_info, global_step, mode):
        for k, v in state_info.items():
            if 'loss' in k:
                self.summary_writer.add_scalar(f'{mode}/{k}', v, global_step)
            if 'weight' in k:
                self.summary_writer.add_scalar(f'{mode}/{k}', v, global_step)
        for i, lr in enumerate(state_info['lr']):
            self.summary_writer.add_scalar(f'{mode}/lr{i}', lr, global_step)

    def log_summary_epoch(self, state_info, mode):
        epoch = state_info['epoch']
        for k, v in state_info.items():
            if 'loss' in k:
                self.summary_writer.add_scalar(f'{mode}/{k}', v, epoch)
            if 'weight' in k:
                self.summary_writer.add_scalar(f'{mode}/{k}', v, epoch)
        for i, lr in enumerate(state_info['lr']):
            self.summary_writer.add_scalar(f'{mode}/lr{i}', lr, epoch)


    def save_checkpoint(self, ckpt_path, epoch, model, optimizer, lat_vec=None):
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'lat_vec': lat_vec.state_dict() if lat_vec is not None else None,
                'optimizer_state_dict': optimizer.state_dict(),
            },
            ckpt_path
        )
        logger.info(ckpt_path)


    def load_checkpoint(self, ckpt_path, model=None, optimizer=None):
        # in-place load
        ckpt = torch.load(ckpt_path)
        if model is not None:
            logger.info(f"load model from ${ckpt_path}")
            model.load_state_dict(ckpt["model_state_dict"])
        if optimizer is not None:
            logger.info(f"load optimizer from ${ckpt_path}")
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info("loaded!")
        return ckpt["epoch"]

