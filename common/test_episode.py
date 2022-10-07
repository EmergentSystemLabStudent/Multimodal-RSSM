import os

import torch

import numpy as np
from tqdm import tqdm


from common.env import MultimodalControlSuiteEnv
from algos.Imitation_Learning.KL_minimizing_prior.algo import IL
from algos.base.base import get_test_episode_ray

import ray

    
def run(cfg, model_path, n_test_episode=100, seed=3000, folder_name=".", save_episode=False, n_process=5):
    device = torch.device(cfg.main.device)

    
    if cfg.env.action_size == None:
        env = MultimodalControlSuiteEnv(cfg.env.env_config, cfg.env.symbolic_env, cfg.main.seed,
                cfg.env.env_config.horizon, cfg.env.action_repeat, cfg.env.bit_depth, cfg.env.info_names,
                noise_scale=cfg.env.noise_scale, add_noise=cfg.env.add_noise, noisy_background=cfg.env.noisy_background)
        cfg.env.action_size = env.action_size

    IL_model = IL(cfg, device)
    IL_model.load_model(model_path)

    cfg.main.test_episodes = 1
    total_rewards = []
    
    ray.init(num_cpus=n_process)
    work_in_progresses = [get_test_episode_ray.remote(IL_model, cfg, device, seed) for seed in range(seed, seed+n_test_episode)]
    
    results = []
    total_rewards = []
    for i in tqdm(range(n_test_episode)):
        finished, work_in_progresses = ray.wait(work_in_progresses, num_returns=1)
        
        result = ray.get(finished[0])
        results.append(result)
        total_reward = np.sum(np.array(result['reward']), axis=0)
        total_rewards.append(total_reward)
        tqdm.write("{}, mean:{}".format(total_reward, np.mean(total_rewards)))

    ray.shutdown()
    total_rewards = np.array(total_rewards)
    print("mean:{}".format(np.mean(total_rewards)))
    save_folder = os.path.join(os.path.dirname(model_path), "results")
    os.makedirs(save_folder, exist_ok=True)
    np.save(os.path.join(save_folder, "total_rewards.npy"), total_rewards)
    
    if save_episode:
        test_episodes = dict()
        for name in results[0].keys():
            test_episodes[name] = np.stack([results[i][name] for i in range(n_test_episode)])
        torch.save(test_episodes, os.path.join(save_folder, "test_episode.pth"))
    