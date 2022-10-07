import os

import torch
from tqdm import tqdm

from common.env import MultimodalControlSuiteEnv
from algos.base.memory import ExperienceReplay_Multimodal
from algos.Reinforcement_Learning.Dreamer.algo import RL

from common.logger import end_logger, setup
import wandb


def train(cfg, cwd, results_dir, device):
    # Initialise training environment and experience replay memory
    print("Initialise training environment and experience replay memory")
    env = MultimodalControlSuiteEnv(cfg.env.env_config, cfg.env.symbolic_env, cfg.main.seed,
            cfg.env.env_config.horizon, cfg.env.action_repeat, cfg.env.bit_depth, cfg.env.info_names,
            noise_scale=cfg.env.noise_scale, add_noise=cfg.env.add_noise, noisy_background=cfg.env.noisy_background)
    if cfg.env.action_size == None:
        cfg.env.action_size = env.action_size
    
    if cfg.train.experience_replay != '' and os.path.exists(cfg.train.experience_replay):
        D = torch.load(cfg.train.experience_replay)
    elif not cfg.main.test:
        observation_names = list(set(cfg.model.observation_names_enc+cfg.model.observation_names_rec))
        D = ExperienceReplay_Multimodal(size=cfg.train.experience_size, 
                                        observation_names=observation_names,
                                        observation_shapes=cfg.env.observation_shapes,
                                        action_name=cfg.env.action_name,
                                        action_size=cfg.env.action_size, 
                                        bit_depth=cfg.env.bit_depth, 
                                        device=device)
        # Initialise dataset D with S random seed episodes
        print("Initialise dataset D with {} random seed episodes".format(cfg.train.seed_episodes))
        total_step = 0
        for s in tqdm(range(1, cfg.train.seed_episodes + 1)):
            observation, done, t = env.reset(), False, 0
            while not done:
                action = env.sample_random_action()
                next_observation, reward, done = env.step(action)
                D.append(observation, action, reward, done)
                observation = next_observation
                t += 1
            total_step += t
        

    # Initialise model parameters randomly
    print("Initialise model parameters randomly")
    model = RL(cfg, device)
    
    # Training (and testing)
    for episode in tqdm(range(cfg.train.seed_episodes + 1, cfg.env.episodes + 1), total=cfg.env.episodes, initial=cfg.train.seed_episodes + 1, desc="Traning loop"):
        # Model fitting
        model.optimize(cfg, D)

        # Data collection
        train_reward, t = model.data_collection(cfg, env, D, device)
        total_step += t

        # Test model
        if episode % cfg.main.test_interval == 0:
            model.test_model(cfg, device, episode//cfg.main.test_interval)

        tqdm.write("episodes: {}, total_steps: {}, optimize_itr: {}, train_reward: {} ".format(
            episode, total_step, model.itr_optim, train_reward))
            
        if cfg.main.wandb:
            log_data = {
                "total_step":total_step,
                "episode":episode,
            }                
            wandb.log(data=log_data, step=model.itr_optim)

        # Checkpoint models
        if episode % cfg.main.checkpoint_interval == 0:
            model.save_model(results_dir, episode)
            if cfg.main.checkpoint_experience:
                # Warning: will fail with MemoryError with large memory sizes
                torch.save(D, os.path.join(results_dir, 'experience.pth'))


    # Close training environment
    env.close()



def run(cfg):
    # init logger and set seeds
    cwd, results_dir, device = setup(cfg)
    # ---------- train RSSM ----------
    train(cfg, cwd, results_dir, device)

    # ---------- ML Flow log ----------
    end_logger(cfg)