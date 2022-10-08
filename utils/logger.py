import os

import numpy as np
import torch

import datetime
import subprocess

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf

import wandb

# get base folder name from experiment_name, date, and run ID
def get_base_folder_name(cwd=".", experiment_name="."):
    dt_now = datetime.date.today()
    
    count = 0
    while(True):
        base_folder_name = "{}/results/{}/{}/run_{}".format(cwd, experiment_name, dt_now, count)
        if not os.path.exists(base_folder_name):
            print("base_folder_name: {}".format(base_folder_name))
            break
        else:
            count += 1
    run_name = "{}/{}/run_{}".format(experiment_name, dt_now, count)
    os.makedirs(base_folder_name, exist_ok=True)
    return base_folder_name, run_name

# get git hash
def get_git_hash():
    cmd = "git rev-parse --short HEAD"
    hash = subprocess.check_output(cmd.split()).strip().decode('utf-8')
    return hash

# initialize config
def init_cfg(cfg, results_dir):
    hash = get_git_hash()
    cfg.main.git_hash = hash

    # Overshooting distance cannot be greater than chunk size
    cfg.rssm.overshooting_distance = min(cfg.train.chunk_size, cfg.rssm.overshooting_distance)

    cfg.main.log_dir = results_dir

    print(' ' * 5 + 'Options')
    for k, v in cfg.items():
        print(' ' * 5 + k)
        for k2, v2 in v.items():
            print(' ' * 10 + k2 + ': ' + str(v2))
    
    # save config to result folder
    file_name_cfg = '{}/hydra_config.yaml'.format(results_dir)
    OmegaConf.save(cfg, file_name_cfg)

    return cfg, file_name_cfg

# trans config format from hydra format to dict
def cfg2dict(cfg):
    if type(cfg) == DictConfig:
        cfg_dict = dict()
        for key in cfg.keys():
            cfg_dict[key] = cfg2dict(cfg[key])
        return cfg_dict
    elif type(cfg) == ListConfig:
        return list(cfg)
    else:
        return cfg

# stop logger (if use logger)
def stop_logger(cfg):
    if cfg.main.wandb:
        wandb.finish()

# init config and logger, and set seed and device
def setup_experiment(cfg):
    print("Setup experiment")
    if cfg.main.experiment_name == None:
        print("Please set experiment_name")
        quit()
    cwd = hydra.utils.get_original_cwd()
    results_dir, run_name = get_base_folder_name(cwd, cfg.main.experiment_name)

    # init config
    cfg, file_name_cfg = init_cfg(cfg, results_dir)
    
    # init logger
    if cfg.main.wandb:
        wandb.init(name=run_name, project=cfg.env.env_config.env_name, config=cfg2dict(cfg), tags=cfg.main.tags)
        wandb.save(file_name_cfg, base_path=results_dir)

    # set seed
    np.random.seed(cfg.main.seed)
    torch.manual_seed(cfg.main.seed)

    # set device
    if torch.cuda.is_available() and not cfg.main.disable_cuda:
        print("using {}".format(cfg.main.device))
        device = torch.device(cfg.main.device)
        torch.cuda.manual_seed(cfg.main.seed)
    else:
        print("using CPU")
        device = torch.device('cpu')

    return cwd, results_dir, device