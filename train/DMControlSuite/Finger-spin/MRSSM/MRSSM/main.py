#Please do this !!!
#export HYDRA_FULL_ERROR=1
import sys
import os
from pathlib import Path
module_path = os.path.join(Path().resolve(), '../../../../..')
sys.path.append(module_path)
os.environ['PYTHONPATH'] = module_path

import hydra
from omegaconf import DictConfig, OmegaConf

import copy
import glob

from algos.MRSSM.MRSSM.train import run

import torch
torch.backends.cudnn.benchmark = True

def setting_demonstration(cfg, n_expert=5, n_uniform=5, n_data=100):
    _cfg = copy.deepcopy(cfg)
    cwd = hydra.utils.get_original_cwd()

    taskset_name = _cfg.env.taskset_name
    env_name = _cfg.env.env_config.env_name
    folder_name_expert = cwd+"/../../../../../dataset/{}/{}/Expert_demonstration/train/optimal/*".format(taskset_name, env_name)
    folder_names_expert = glob.glob(folder_name_expert)
    folder_names_expert.sort()

    folder_name_uniform = cwd+"/../../../../../dataset/{}/{}/demonstration_for_RSSM/train/uniform/*".format(taskset_name, env_name)
    folder_names_uniform = glob.glob(folder_name_uniform)
    folder_names_uniform.sort()

    _cfg.train.experience_replay = folder_names_expert[:n_expert] + folder_names_uniform[:n_uniform]
    _cfg.train.validation_data = folder_names_expert[-1:] + folder_names_uniform[-1:]
    _cfg.train.n_episode = None

    _cfg.main.experiment_name += "-{}_experts-{}_uniforms".format(n_data*n_expert, n_data*n_uniform)
    return _cfg

def setting_seed(cfg, seed):
    _cfg = copy.deepcopy(cfg)
    _cfg.main.seed = seed
    _cfg.main.experiment_name += "-seed_{}".format(seed)
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
def main(cfg_raw : DictConfig) -> None:
    experiment_name = "RSSM"
    tags = ["RSSM"]
    for seed in range(0,5):
        for n_expert in [0]:
            n_uniform = 10-n_expert
            _cfg = copy.deepcopy(cfg_raw)
            _cfg = set_experiment_name(_cfg, experiment_name)
            _cfg = set_tags(_cfg, tags)
            _cfg = setting_demonstration(_cfg, n_uniform=n_uniform, n_expert=n_expert)
            _cfg = setting_seed(_cfg, seed)
            run(_cfg)

if __name__=="__main__":
    main()
