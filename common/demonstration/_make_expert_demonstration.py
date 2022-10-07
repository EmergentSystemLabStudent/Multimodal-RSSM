import sys
import os
from pathlib import Path

sys.path.append(os.path.join(Path().resolve(), '../../../..'))

import numpy as np
import torch
from algos.Reinforcement_Learning.Dreamer.algo import RL

from tqdm import tqdm

import hydra
from omegaconf import DictConfig

from common.env import MultimodalControlSuiteEnv


def run(cfg, n_test_episode=None, output_dir="expert_demonstrations", seed_base=2000, itr=1000):
    cfg.main.wandb = False
    cfg.main.mlflow = False
    device = torch.device(cfg.main.device)

    env = MultimodalControlSuiteEnv(cfg.env.env_config, cfg.env.symbolic_env, cfg.main.seed,
            cfg.env.env_config.horizon, cfg.env.action_repeat, cfg.env.bit_depth, cfg.env.info_names,
            noise_scale=cfg.env.noise_scale, add_noise=cfg.env.add_noise, noisy_background=cfg.env.noisy_background)
    cfg.env.action_size = env.action_size

    model = RL(cfg, device)
    model.load_model('{}/models_{}.pth'.format(cfg.main.log_dir, itr))

    
    cwd = hydra.utils.get_original_cwd()
    save_dir = "{}/{}".format(cwd,output_dir)
    os.makedirs(save_dir, exist_ok=True)

    if n_test_episode == None:
        n_test_episode = cfg.main.test_episodes
    cfg.main.test_episodes = 1
    for i in tqdm(range(n_test_episode)):
        seed = seed_base + i
        test_episode = model.get_test_episode(cfg, device, seed=seed)

        observations = dict()
        for name in test_episode['observation'][0].keys():
            observations[name] = []
        for t in range(len(test_episode['observation'])):
            for name in test_episode['observation'][t].keys():
                observations[name].append(test_episode['observation'][t][name])
        for name in test_episode['observation'][0].keys():
            observations[name] = np.array(observations[name])
        test_episode_info = dict()
        for name in test_episode.keys():
            if name == 'observation':
                for k in observations.keys():
                    test_episode_info[k] = observations[k]
            else:
                test_episode_info[name] = np.array(test_episode[name])

        demonstration = dict()
        for name in test_episode_info.keys():
            demonstration[name] = test_episode_info[name][:, 0]
        
        np.save('{}/seed_{}.npy'.format(save_dir, seed), demonstration, allow_pickle=True)

@hydra.main(config_path=".", config_name="config")
def main(cfg : DictConfig) -> None:
    run(cfg)

if __name__=="__main__":
    main()