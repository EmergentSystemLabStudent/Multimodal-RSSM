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
    for seed in range(0,1):
        _cfg = copy.deepcopy(cfg_raw)
        _cfg = set_experiment_name(_cfg, experiment_name)
        _cfg = set_tags(_cfg, tags)
        _cfg = setting_seed(_cfg, seed)
        run(_cfg)

if __name__=="__main__":
    main()
