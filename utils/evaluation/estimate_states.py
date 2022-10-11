import torch
import numpy as np

import glob
from tqdm import tqdm

from utils.processing.image_processing import normalize_image
from utils.replay_buffer.data_augment import augment_image_data

from algos.MRSSM.MRSSM.train import get_dataset_loader

def tensor2numpy_state(state):
    for key in state.keys():
        if "expert" in key:
            for k in state[key].keys():
                state[key][k] = state[key][k].detach().cpu().numpy()    
        else:
            state[key] = state[key].detach().cpu().numpy()
        
    return state

def estimate_state(model, observations, actions, rewards, nonterminals):
    state = model.estimate_state(observations, actions, rewards, nonterminals)
    return state

def get_all_data(D):
    observations = dict()
    for name in D.observation_names:
        observations[name] = D.observations[name][:D.idx]
    actions = D.actions[:D.idx]
    rewards = D.rewards[:D.idx]
    nonterminals = D.nonterminals[:D.idx]
    return observations, actions, rewards, nonterminals
    
def get_episode_data(D, epi_idx, crop_idx=None, pca_rand=None):
    idx_done, _ = np.where(D.nonterminals[:D.idx] == 0)
    idx_done = np.hstack([[0], idx_done + 1])
    idx_start = idx_done[epi_idx]
    idx_end = idx_done[epi_idx + 1]
    _observations, _actions, _rewards, _nonterminals = get_all_data(D)

    observations = dict()
    for name in _observations.keys():
        observations[name] = _observations[name][idx_start:idx_end].to(device=D.device).unsqueeze(1).to(torch.float32)
        if "image" in name:
            if "bin" in name:
                observations[name], _ = augment_image_data(
                    observations[name], name, D.n_crop, D.dh_base, D.dw_base, crop_idx=crop_idx)
            else:
                observations[name], pca_rand = augment_image_data(
                    observations[name], name, D.n_crop, D.dh_base, D.dw_base, D.noise_scales, pca_rand, D.lambd_eigen_values[name], D.p_eigen_vectors[name], D.pca_scales, crop_idx=crop_idx)
                observations[name] = normalize_image(observations[name], torch.tensor(D.bit_depth))

    actions = _actions[idx_start:idx_end].to(device=D.device).unsqueeze(1)
    rewards = _rewards[idx_start:idx_end].to(D.device).unsqueeze(1)
    nonterminals = _nonterminals[idx_start:idx_end].to(D.device).unsqueeze(1)

    return observations, actions, rewards, nonterminals

def get_states(D, model, device, crop_idx=0, pca_rand=None):
    states = dict()
    for epi_idx in tqdm(range(D.episodes)):
        observations, actions, rewards, nonterminals = get_episode_data(D, epi_idx=epi_idx, crop_idx=crop_idx, pca_rand=pca_rand)
        
        _observations = model._clip_obs(observations, idx_start=1)
        state = estimate_state(model, _observations, actions[:-1], rewards, nonterminals[:-1])
        states[D.file_names[epi_idx]] = tensor2numpy_state(state)
        del state, observations, actions, rewards, nonterminals
        torch.cuda.empty_cache()
    return states


def run(cfg, cwd, device, model_class, model_path):
    # load train data
    D = get_dataset_loader(cfg, cwd, device, cfg.train.train_data_path)

    # load model
    model = model_class(cfg, device)
    model.load_model(model_path)
    model.eval()
    print("model_path: {}".format(model_path))
    
    # get states
    states = get_states(D, model, device)

    # save states
    save_file_name = model_path.replace(".pth",".npy").replace("/models_","/states_models_")
    print("save to {}".format(save_file_name))
    np.save(save_file_name, states)


