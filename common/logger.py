import os

import numpy as np
import torch

import mlflow

from omegaconf import DictConfig, ListConfig, OmegaConf
import datetime
import subprocess

import hydra
import wandb

def get_base_folder_name(cwd=".", experiment_name=".", task="."):
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

def get_git_hash():
    cmd = "git rev-parse --short HEAD"
    hash = subprocess.check_output(cmd.split()).strip().decode('utf-8')
    return hash

def init_cfg(cfg, results_dir):
    hash = get_git_hash()
    cfg.main.git_hash = hash

    # Overshooting distance cannot be greater than chunk size
    cfg.model.overshooting_distance = min(cfg.train.chunk_size, cfg.model.overshooting_distance)

    cfg.main.log_dir = results_dir

    print(' ' * 5 + 'Options')
    for k, v in cfg.items():
        print(' ' * 5 + k)
        for k2, v2 in v.items():
            print(' ' * 10 + k2 + ': ' + str(v2))
    
    
    file_name_cfg = '{}/hydra_config.yaml'.format(results_dir)
    OmegaConf.save(cfg, file_name_cfg)

    return cfg, file_name_cfg

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

def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)

def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f'{parent_name}.{k}', v)
            else:
                mlflow.log_param(f'{parent_name}.{k}', v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f'{parent_name}.{i}', v)

def init_mlflow(cfg, cwd):
    # ---------- ML Flow setting ----------
    # mlrunsディレクトリ指定
    tracking_uri = cwd+'/mlruns'    # パス
    mlflow.set_tracking_uri(tracking_uri)
    # experiment指定
    mlflow.set_experiment(cfg.main.experiment_name)

    mlflow.start_run()

    # sava params to mlflow
    log_params_from_omegaconf_dict(cfg)

def end_logger(cfg):
    if cfg.main.wandb:
        wandb.finish()

    if cfg.main.mlflow:
        mlflow.log_artifact(os.path.join(os.getcwd(), '.hydra/config.yaml'))
        mlflow.log_artifact(os.path.join(os.getcwd(), '.hydra/hydra.yaml'))
        mlflow.log_artifact(os.path.join(os.getcwd(), '.hydra/overrides.yaml'))
        mlflow.log_artifact(os.path.join(os.getcwd(), 'main.log'))

        mlflow.end_run()

def setup(cfg):
    print("Setup")
    if cfg.main.experiment_name == None:
        print("Please set experiment_name")
        quit()
    cwd = hydra.utils.get_original_cwd()
    results_dir, run_name = get_base_folder_name(cwd, cfg.main.experiment_name, cfg.env.env_config.env_name)

    # init config
    cfg, file_name_cfg = init_cfg(cfg, results_dir)
    
    # init logger
    if cfg.main.wandb:
        wandb.init(name=run_name, project=cfg.env.env_config.env_name, config=cfg2dict(cfg), tags=cfg.main.tags)
        wandb.save(file_name_cfg, base_path=results_dir)

    if cfg.main.mlflow:
        init_mlflow(cfg, cwd)

    np.random.seed(cfg.main.seed)
    torch.manual_seed(cfg.main.seed)
    if torch.cuda.is_available() and not cfg.main.disable_cuda:
        print("using {}".format(cfg.main.device))
        device = torch.device(cfg.main.device)
        torch.cuda.manual_seed(cfg.main.seed)
    else:
        print("using CPU")
        device = torch.device('cpu')

    return cwd, results_dir, device