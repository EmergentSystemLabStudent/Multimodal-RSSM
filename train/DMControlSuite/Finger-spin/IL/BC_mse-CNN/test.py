#Please do this !!!
#export HYDRA_FULL_ERROR=1
import sys
import os
from pathlib import Path
# module_path = os.path.join(Path().resolve(), '../../../../..')
module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../../..')
sys.path.append(module_path)
os.environ['PYTHONPATH'] = module_path

from common.test_episode import run

from hydra import initialize, compose

import torch

import numpy as np
from tqdm import tqdm

from common.env import MultimodalControlSuiteEnv
from algos.Imitation_Learning.Behavioral_Cloning_mse_CNN.algo import IL
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

def multi_run(path, seed=3000):
    dirpath = os.path.join(os.path.dirname(__file__), path)
    files = os.listdir(dirpath)
    # files = os.listdir(path)
    foders_dir = [f for f in files if os.path.isdir(os.path.join(dirpath, f))]
    print(files, foders_dir)

    device = "cpu"
    # device = "cuda:0"
    save_episode = False
    n_test_episode = 100
    itr = 10_000

    for folder_dir in foders_dir:
        files_path = os.listdir(dirpath+"/"+folder_dir)
        if (not "results" in files_path) and ("hydra_config.yaml" in files_path):
            with initialize(path+"/"+folder_dir):
                cfg = compose(config_name="hydra_config")
            cfg.main.device = device
            cfg.main.experiment_name = "test-" + cfg.main.experiment_name
            cfg.main.wandb = False
            
            abspath = os.path.dirname(os.path.abspath(__file__))
            log_dir = (cfg.main.log_dir).replace("/home/docker/sharespace/MultimodalRSSM/train", abspath+"/../../../..")
            
            if "results" in os.listdir(log_dir):
                continue
            model_path = "{}/models_{}.pth".format(log_dir, itr)
            run(cfg, model_path=model_path, n_test_episode=n_test_episode, seed=seed, folder_name=folder_dir, save_episode=save_episode)

def main():
    multi_run(path="test")

if __name__=="__main__":
    main()
