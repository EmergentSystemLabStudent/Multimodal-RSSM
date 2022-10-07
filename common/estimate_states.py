#Please do this !!!
#export HYDRA_FULL_ERROR=1
import sys
import os
from pathlib import Path
sys.path.append(os.path.join(Path().resolve(), '../../../..'))



import torch
import numpy as np

import glob
from tqdm import tqdm

from algos.base.memory import ExperienceReplay_Multimodal
from omegaconf import ListConfig

def get_center_idx(n_crop):
    return int((int(np.sqrt(n_crop-1)/2)+1)**2-1)

def tensor2numpy_state(state):
    for key in state.keys():
        if "expert" in key:
            for k in state[key].keys():
                state[key][k] = state[key][k].detach().cpu().numpy()    
        else:
            state[key] = state[key].detach().cpu().numpy()
        
    return state


def clip_obs(obs, t_start=0, t_end=-1):
    output = dict()
    for key in obs.keys():
        output[key] = obs[key][t_start:t_end]
    return output

def estimate_state(model, observations, actions, rewards, nonterminals):
    state = model.estimate_state(observations, actions, rewards, nonterminals)
    return state

def expand_time_0(tensor):
    _tensor = torch.zeros_like(tensor[:1])
    return torch.vstack([_tensor, tensor])

def get_states(cfg, D, model, device):
    crop_idx = get_center_idx(cfg.train.n_crop)
    states = dict()
    for epi_idx in tqdm(range(D.episodes)):
        observations, actions, rewards, nonterminals = D.episode_data(idx=epi_idx, crop_idx=crop_idx)
        
        actions = expand_time_0(actions)
        nonterminals = expand_time_0(nonterminals)
        state = estimate_state(model, observations, actions[:-1], rewards, nonterminals[:-1])
        states[D.file_names[epi_idx]] = tensor2numpy_state(state)
        del state, observations, actions, rewards, nonterminals
        torch.cuda.empty_cache()
    return states

def load_dataset(cfg, cwd, experience_replay, D):
    dataset_dir = os.path.join(cwd, experience_replay)
    if os.path.exists(dataset_dir):    
        print("load dataset from {}".format(dataset_dir))
        if os.path.isdir(dataset_dir):
            D.load_dataset(dataset_dir, n_episode=cfg.train.n_episode, n_ep_per_data=cfg.train.n_episode_per_data)
        else:
            dataset = torch.load(dataset_dir)
            D.convert_dataset(dataset)

def run(cfg, model_class, model_folder, model_idx=-1):
    cfg_device = "cuda:0"
    cfg.main.device = cfg_device
    print(' ' * 26 + 'Options')
    for k, v in cfg.items():
        print(' ' * 26 + k + ': ' + str(v))

    cwd = "."
    device = torch.device(cfg.main.device)
    cfg.train.pca_scales = [0]
    cfg.train.noise_scales = [0]
    observation_names = list(set(cfg.model.observation_names_enc+cfg.model.observation_names_rec))
    D = ExperienceReplay_Multimodal(size=cfg.train.experience_size,
                                            max_episode_length=cfg.env.max_episode_length,
                                            observation_names=observation_names,
                                            observation_shapes=cfg.env.observation_shapes,
                                            n_crop=cfg.train.n_crop,
                                            noise_scales=cfg.train.noise_scales,
                                            pca_scales=cfg.train.pca_scales,
                                            dh_base=cfg.train.dh_base,
                                            dw_base=cfg.train.dw_base,
                                            action_name=cfg.env.action_name,
                                            action_size=cfg.env.action_size,
                                            bit_depth=cfg.env.bit_depth,
                                            device=device,
                                            taskset_name=cfg.env.taskset_name)
    if type(cfg.train.experience_replay) == str:
        load_dataset(cfg, cwd, cfg.train.experience_replay, D)
    elif type(cfg.train.experience_replay) == ListConfig:
        for experience_replay in cfg.train.experience_replay+cfg.train.validation_data:
            load_dataset(cfg, cwd, experience_replay, D)
    

    model = model_class(cfg, device)
    model_paths = glob.glob(os.path.join(model_folder, '*.pth'))

    model_path = model_paths[model_idx]
    print("model_path: {}".format(model_path))
    model.load_model(model_path)
    model.eval()
    
    states = get_states(cfg, D, model, device)

    save_file_name = model_path.replace(".pth",".npy").replace("/models_","/states_models_")
    print("save to {}".format(save_file_name))
    np.save(save_file_name, states)


