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

from algos.Reinforcement_Learning.Dreamer.train import run

import torch
torch.backends.cudnn.benchmark = True


@hydra.main(config_path="config", config_name="config")
def main(cfg : DictConfig) -> None:
    run(cfg)

if __name__=="__main__":
    main()