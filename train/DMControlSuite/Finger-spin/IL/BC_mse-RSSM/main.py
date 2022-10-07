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

import copy
import glob

from algos.Imitation_Learning.Behavioral_Cloning_mse.train import run

import torch
torch.backends.cudnn.benchmark = True

def load_rssm_cfg(cfg):
    _cfg = copy.deepcopy(cfg)
    # cwd = hydra.utils.get_original_cwd()
    rssm_model_path = os.path.dirname(os.path.join("../../..", cfg.train.rssm.model_path))
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize(config_path=rssm_model_path):
        rssm_cfg = compose(config_name="hydra_config")
    _cfg.env                         = rssm_cfg.env

    _cfg.model.observation_names_enc = rssm_cfg.model.observation_names_enc
    _cfg.model.observation_names_rec = rssm_cfg.model.observation_names_rec
    _cfg.model.condition_names       = rssm_cfg.model.condition_names
    _cfg.model.predict_reward        = rssm_cfg.model.predict_reward
    _cfg.model.multimodal            = rssm_cfg.model.multimodal
    _cfg.model.multimodal_params     = rssm_cfg.model.multimodal_params
    _cfg.model.activation_function   = rssm_cfg.model.activation_function
    _cfg.model.embedding_size        = rssm_cfg.model.embedding_size
    _cfg.model.hidden_size           = rssm_cfg.model.hidden_size
    _cfg.model.belief_size           = rssm_cfg.model.belief_size
    _cfg.model.state_size            = rssm_cfg.model.state_size
    _cfg.model.normalization         = rssm_cfg.model.normalization
    
    return _cfg

def setting_rssm(cfg, name, path, fix_flag=True):
    _cfg = copy.deepcopy(cfg)

    _cfg.train.rssm.load = True
    _cfg.train.rssm.model_path = path
    _cfg = load_rssm_cfg(_cfg)
    
    _cfg.main.experiment_name += "-"+name

    _cfg.train.rssm.fix = fix_flag
    if fix_flag:
        _cfg.main.experiment_name += "-fix"
        _cfg.main.tags += ["fix-RSSM"]
    else:
        _cfg.main.experiment_name += "-fine_tuning"
        _cfg.main.tags += ["fine_tune-RSSM"]
    return _cfg

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

def setting_seed(cfg, seed):
    _cfg = copy.deepcopy(cfg)
    _cfg.main.seed = seed
    _cfg.main.experiment_name += "-seed_{}".format(seed)
    return _cfg

def setting_policy(cfg, horizon, seed, n_expert=5, n_data=100, lr=1e-3):
    _cfg = copy.deepcopy(cfg)
    
    _cfg.model.planning_horizon = horizon
    _cfg.main.experiment_name += "-T_{}".format(horizon)

    _cfg = setting_demonstration(_cfg, n_expert, n_data)

    _cfg.model.actor_learning_rate = lr
    _cfg.model.value_learning_rate = lr

    _cfg = setting_seed(_cfg, seed)
    return _cfg

def setting_non_rssm(cfg, lr=1e-3):
    _cfg = copy.deepcopy(cfg)
    _cfg.main.experiment_name += "-non_RSSM"
    _cfg.main.tags += ["Non-RSSM"]

    _cfg.train.rssm.load = False
    _cfg.train.rssm.model_path = None
    _cfg.train.rssm.fix = False
    _cfg.model.model_learning_rate = lr
    
    return _cfg

def set_experiment_name(cfg, experiment_name):
    _cfg = copy.deepcopy(cfg)
    _cfg.main.experiment_name = copy.deepcopy(experiment_name)
    return _cfg

def set_tags(cfg, tags):
    _cfg = copy.deepcopy(cfg)
    _cfg.main.tags = copy.deepcopy(tags)
    return _cfg

def get_name_and_path(seed, itr, n_expert=5, n_uniform=5, n_data=100):
    name = "RSSM-{}_experts-{}_uniforms-seed_{}".format(n_data*n_expert, n_data*n_uniform, seed)
    # if seed < 1:
    #     date = "09-16"
    # else:
    #     date = "09-17"
    date = "10-04"
    path = "../../MRSSM/MRSSM/results/{}/2022-{}/run_1/models_{}.pth".format(name, date, itr)
    return name, path

@hydra.main(config_path="config", config_name="config")
def main(cfg : DictConfig) -> None:
    experiment_name = "BC_mse-RSSM"
    tags = ["IL", "BC", "BC-RSSM", "BC_mse-RSSM"]

    horizon = [2]
    # seeds = [0,1,2,3,4]
    seeds = [0]
    itrs = [50_000]
    load_RSSM = True
    lr = 1e-4
    for seed in seeds:
        for T in horizon:
            for fix_flag in [True]:
                for itr in itrs:
                    for n_expert in [1,2]:
                        n_expert_rssm = 0
                        n_uniform_rssm = 10
                        n_expert_policy = n_expert
                        _cfg = copy.deepcopy(cfg)
                        _cfg = set_experiment_name(_cfg, experiment_name)
                        _cfg = set_tags(_cfg, tags)
                        _cfg = setting_policy(_cfg, horizon=T, seed=seed, n_expert=n_expert_policy,lr=lr)
                        if load_RSSM:
                            name, path = get_name_and_path(seed, itr, n_expert=n_expert_rssm, n_uniform=n_uniform_rssm)
                            _cfg = setting_rssm(_cfg, name=name, path=path, fix_flag=fix_flag)
                        else:
                            _cfg = setting_non_rssm(_cfg)
                        run(_cfg)

    # for seed in seeds:
    #     for T in horizon:
    #         for fix_flag in [True]:
    #             for itr in itrs:
    #                 for n_expert in [5]:
    #                     n_expert_rssm = n_expert
    #                     n_expert_policy = n_expert
    #                     _cfg = copy.deepcopy(cfg)
    #                     _cfg = set_experiment_name(_cfg, experiment_name)
    #                     _cfg = set_tags(_cfg, tags)
    #                     _cfg = setting_policy(_cfg, horizon=T, seed=seed, n_expert=n_expert_policy)
    #                     if load_RSSM:
    #                         name, path = get_name_and_path(seed, itr, n_expert=n_expert_rssm)
    #                         _cfg = setting_rssm(_cfg, name=name, path=path, fix_flag=fix_flag)
    #                     else:
    #                         _cfg = setting_non_rssm(_cfg)
    #                     run(_cfg)

if __name__=="__main__":
    main()
