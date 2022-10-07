import sys
import os
from pathlib import Path

sys.path.append(os.path.join(Path().resolve(), '../../../..'))

import numpy as np
import torch

from tqdm import tqdm

import hydra

from common.env import MultimodalEnvBatcher, MultimodalControlSuiteEnv

import ray

def observation2np(observation):
    output = dict()
    for name in observation.keys():
        output[name] = observation[name].detach().cpu().numpy()
    return output

@ray.remote
def get_uniform_action_episode(cfg, device="cpu", seed=2001):
    # Initialise parallelised test environments
    env_kwargs = dict(env_config=cfg.env.env_config, 
                      symbolic=cfg.env.symbolic_env, 
                      seed=seed, 
                      episode_horizon=cfg.env.env_config.horizon,
                      action_repeat=cfg.env.action_repeat, 
                      bit_depth=cfg.env.bit_depth, 
                      info_names=cfg.env.info_names)
    test_envs = MultimodalEnvBatcher(MultimodalControlSuiteEnv, (), env_kwargs, cfg.main.test_episodes)

    episode_info = dict()
    with torch.no_grad():
        observations = []
        rewards = []
        dones = []
        actions = []

        observation, total_rewards = test_envs.reset(), np.zeros((cfg.main.test_episodes, ))
        action = torch.zeros(cfg.main.test_episodes, test_envs.action_size, device=device)

        observations.append(observation2np(observation))
        
        for t in range(cfg.env.env_config.horizon // cfg.env.action_repeat):
            action = test_envs.sample_random_action()
            next_observation, reward, done = test_envs.step(action.cpu() if isinstance(test_envs, MultimodalEnvBatcher) else action[0].cpu())  # Perform environment step (action repeats handled internally)
            total_rewards += reward.numpy()
            
            observation = next_observation
            observations.append(observation2np(observation))    
            rewards.append(reward.detach().cpu().numpy())
            dones.append(done.detach().cpu().numpy())
            actions.append(action.detach().cpu().numpy())

            if done.sum().item() == cfg.main.test_episodes:
                break

        episode_info['observation'] = observations
        episode_info['reward'] = rewards
        episode_info['done'] = dones
        episode_info['action'] = actions
        episode_info['seed'] = seed

    return episode_info

# def run(cfg):
#     device = torch.device(cfg.main.device)

#     env = MultimodalControlSuiteEnv(cfg.env.env_name, cfg.env.symbolic_env, cfg.main.seed,
#             cfg.env.env_config.horizon, cfg.env.action_repeat, cfg.env.bit_depth, cfg.env.info_names,
#             noise_scale=cfg.env.noise_scale, add_noise=cfg.env.add_noise, noisy_background=cfg.env.noisy_background)
#     cfg.env.action_size = env.action_size
    
#     cwd = hydra.utils.get_original_cwd()
#     save_dir = "{}/uniform".format(cwd)
#     os.makedirs(save_dir, exist_ok=True)

#     seed_base = cfg.main.seed
#     n_test_episode = cfg.main.test_episodes
#     cfg.main.test_episodes = 1
#     for i in tqdm(range(n_test_episode)):
#         seed = seed_base + i
#         test_episode = get_uniform_action_episode(cfg, seed=seed)

#         observations = dict()
#         for name in test_episode['observation'][0].keys():
#             observations[name] = []
#         for t in range(len(test_episode['observation'])):
#             for name in test_episode['observation'][t].keys():
#                 observations[name].append(test_episode['observation'][t][name])
#         for name in test_episode['observation'][0].keys():
#             observations[name] = np.array(observations[name])
#         test_episode_info = dict()
#         for name in test_episode.keys():
#             if name == 'observation':
#                 for k in observations.keys():
#                     test_episode_info[k] = observations[k]
#             else:
#                 test_episode_info[name] = np.array(test_episode[name])

#         demonstration = dict()
#         for name in test_episode_info.keys():
#             demonstration[name] = test_episode_info[name][:, 0]
        
#         np.save('{}/seed_{}.npy'.format(save_dir, seed), demonstration, allow_pickle=True)


def run(cfg, cwd=".", n_test_episode=100, output_dir="uniform", seed_base=2000, itr=1000, folder_name=".", n_process=5):
    save_dir = "{}/{}".format(cwd,output_dir)
    os.makedirs(save_dir, exist_ok=True)
    print("save to {}".format(save_dir))
    
    cfg.main.test_episodes = 1
    total_rewards = []
    
    ray.init(num_cpus=n_process)
    work_in_progresses = [get_uniform_action_episode.remote(cfg, seed=seed) for seed in range(seed_base, seed_base+n_test_episode)]
    
    for i in tqdm(range(n_test_episode)):
        finished, work_in_progresses = ray.wait(work_in_progresses, num_returns=1)
        
        test_episode = ray.get(finished[0])
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
            if name == "seed":
                demonstration[name] = test_episode_info[name]
            else:
                demonstration[name] = test_episode_info[name][:, 0]

        seed = test_episode['seed']
        np.save('{}/seed_{}.npy'.format(save_dir, seed), demonstration, allow_pickle=True)
    ray.shutdown()