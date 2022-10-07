#Please do this !!!
#export HYDRA_FULL_ERROR=1
import sys
import os
from pathlib import Path
module_path = os.path.join(Path().resolve(), '../../../../..')
sys.path.append(module_path)
os.environ['PYTHONPATH'] = module_path

import hydra
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf

from algos.Imitation_Learning.Behavioral_Cloning_mse_CNN.train import run

import torch
torch.backends.cudnn.benchmark = True

import copy
import glob

def setting_demonstration(cfg, n_expert=5, n_data=100):
    _cfg = copy.deepcopy(cfg)
    cwd = hydra.utils.get_original_cwd()

    taskset_name = _cfg.env.taskset_name
    env_name = _cfg.env.env_config.env_name
    folder_name_expert = cwd+"/../../../../../dataset/{}/{}/Expert_demonstration/train/optimal/*".format(taskset_name, env_name)
    folder_names_expert = glob.glob(folder_name_expert)
    folder_names_expert.sort()

    _cfg.train.experience_replay = folder_names_expert[:n_expert]
    _cfg.train.validation_data = folder_names_expert[-1:]
    _cfg.train.n_episode = None

    _cfg.main.experiment_name += "-{}_experts".format(n_data*n_expert)
    return _cfg

def setting_policy(cfg, seed, n_expert=5, n_data=100, lr=1e-3):
    _cfg = copy.deepcopy(cfg)
    _cfg = setting_demonstration(_cfg, n_expert, n_data)
    
    _cfg.model.actor_learning_rate = lr
    _cfg.model.value_learning_rate = lr

    _cfg = setting_seed(_cfg, seed)
    return _cfg

def setting_seed(cfg, seed):
    _cfg = copy.deepcopy(cfg)
    _cfg.main.seed = seed
    _cfg.main.experiment_name = _cfg.main.experiment_name+"-seed_{}".format(_cfg.main.seed)
    return _cfg

def set_experiment_name(cfg, experiment_name):
    _cfg = copy.deepcopy(cfg)
    _cfg.main.experiment_name = copy.deepcopy(experiment_name)
    return _cfg

def set_tags(cfg, tags):
    _cfg = copy.deepcopy(cfg)
    _cfg.main.tags = copy.deepcopy(tags)
    return _cfg

@hydra.main(config_path="config", config_name="config")
def main(cfg : DictConfig) -> None:
    experiment_name = "BC_mse-CNN"
    tags = ["IL", "BC", "BC_mse-CNN"]
    
    seeds = [0,1,2,3,4]
    for seed in seeds:
        for n_expert in [5,4,3,2,1]:
            n_expert_policy = n_expert
            _cfg = copy.deepcopy(cfg)
            _cfg = set_experiment_name(_cfg, experiment_name)
            _cfg = set_tags(_cfg, tags)
            _cfg = setting_policy(_cfg, seed, n_expert_policy)
            run(_cfg)


if __name__=="__main__":
    main()
