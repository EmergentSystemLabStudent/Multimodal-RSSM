import os

import torch

import numpy as np
from tqdm import tqdm

from algos.Imitation_Learning.KL_minimizing_prior.algo import IL
from algos.base.base import get_test_episode_ray

import ray

    
def run(cfg, cwd=".", n_test_episode=100, output_dir="expert_demonstrations", seed_base=2000, itr=1000, folder_name=".", n_process=5):
    save_dir = "{}/{}".format(cwd,output_dir)
    os.makedirs(save_dir, exist_ok=True)
    print("save to {}".format(save_dir))
    
    device = torch.device(cfg.main.device)

    model_path = cfg.main.log_dir+"/models_{}.pth".format(itr)

    IL_model = IL(cfg, device)
    IL_model.load_model(model_path)

    cfg.main.test_episodes = 1
    total_rewards = []
    
    ray.init(num_cpus=n_process)
    work_in_progresses = [get_test_episode_ray.remote(IL_model, cfg, device, seed, output_states=False) for seed in range(seed_base, seed_base+n_test_episode)]
    
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