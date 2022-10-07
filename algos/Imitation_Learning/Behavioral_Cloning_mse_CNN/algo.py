import os
import numpy as np
from tqdm import tqdm

import torch
from torch.distributions import Normal
from torch import nn, optim
from torch.nn import functional as F

from common.env import MultimodalEnvBatcher, MultimodalControlSuiteEnv

from common.models.policy import ActorModel_Enc
from algos.base.base import Controller_base, Model_base

import wandb

class Controller(Controller_base):
    def __init__(self, 
                 cfg, 
                 device=torch.device("cpu"), 
                 horizon=1) -> None:
        super().__init__(cfg, IL, device, horizon)

class IL(Model_base):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        print("Behavioral Cloning")
        self.cfg = cfg
        self.device = device

        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.use_amp)
        self.itr_optim = 0

        self.init_models(device)
        self.init_param_list()
        self.init_optimizer()

        self.planner = self.actor_model
    
    def eval(self):
        self.actor_model.eval()
    
    def train(self):
        self.actor_model.train()
    
    def init_models(self, device):
        self.actor_model = ActorModel_Enc(name=self.cfg.model.observation_names_enc[0],
                                          observation_shapes=self.cfg.env.observation_shapes,
                                          embedding_size=dict(self.cfg.model.embedding_size), 
                                          activation_function=dict(self.cfg.model.activation_function),
                                          normalization=self.cfg.model.normalization,
                                          hidden_size=self.cfg.model.hidden_size,
                                          action_size=self.cfg.env.action_size, 
                                    ).to(device=device)
        
    def init_param_list(self):
        self.value_actor_param_list = list(self.actor_model.parameters())
        
    def init_optimizer(self):
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(
        ), lr=0 if self.cfg.model.learning_rate_schedule != 0 else self.cfg.model.actor_learning_rate, eps=self.cfg.model.adam_epsilon)
    
    def optimize(self,
                 D, 
                 ):
        self.itr_optim += 1
        # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
        observations, actions, rewards, nonterminals = D.sample(
                self.cfg.train.batch_size, self.cfg.train.chunk_size)  # Transitions start at time t = 0

        self.optimize_loss(observations,
                           actions,
                           self.itr_optim)

    def calc_BC_logprob_loss(self, observations, actions):
        with torch.cuda.amp.autocast(enabled=self.cfg.train.use_amp):
            logprobs = []
            for t in range(0, self.cfg.train.chunk_size):
                logprob = self.actor_model.get_log_prob(observations[self.cfg.model.observation_names_enc[0]][t], actions[t]).sum(dim=-1)
                logprobs.append(logprob)
            actor_loss = - torch.mean(torch.vstack(logprobs))
        return actor_loss

    def calc_BC_mse_loss(self, observations, actions):
        with torch.cuda.amp.autocast(enabled=self.cfg.train.use_amp):
            actor_loss = []
            for t in range(0, self.cfg.train.chunk_size):
                action_pred = self.actor_model.get_action(observations[self.cfg.model.observation_names_enc[0]][t])
                mse = F.mse_loss(action_pred, actions[t], reduction='none').sum(dim=-1).mean()
                actor_loss.append(mse)
            actor_loss = torch.mean(torch.vstack(actor_loss))
        return actor_loss

    def calc_loss(self,
                  observations,
                  actions,
                  ):
        return self.calc_BC_mse_loss(observations, actions)
    
    def optimize_loss(self, 
                      observations,
                      actions,
                      itr_optim):
        with torch.cuda.amp.autocast(enabled=self.cfg.train.use_amp):
            actor_loss = self.calc_loss(observations, actions)
        
        # Update model parameters
        self.actor_optimizer.zero_grad()

        self.scaler.scale(actor_loss).backward()
        nn.utils.clip_grad_norm_(self.actor_model.parameters(),
                                    self.cfg.model.grad_clip_norm, norm_type=2)
        self.scaler.step(self.actor_optimizer)
        self.scaler.update()

        # Log loss info
        loss_info = dict()
        loss_info["actor_loss"] = actor_loss.item()
        
        if self.cfg.main.wandb:
            for name in loss_info.keys():
                wandb.log(data={"{}/train".format(name):loss_info[name]}, step=itr_optim)
            frame = itr_optim * self.cfg.train.batch_size * self.cfg.train.chunk_size
            wandb.log(data={"frame":frame}, step=itr_optim)

        if self.cfg.main.wandb and (itr_optim % self.cfg.main.validation_interval == 0):
            with torch.cuda.amp.autocast(enabled=self.cfg.train.use_amp):
                bc_logprob_loss = self.calc_BC_logprob_loss(observations, actions)
                wandb.log(data={"BC_logprob_loss/train":bc_logprob_loss.item()}, step=itr_optim)
                bc_mse_loss = self.calc_BC_mse_loss(observations, actions)
                wandb.log(data={"BC_mse_loss/train":bc_mse_loss.item()}, step=itr_optim)
                
    def validation(self,
                    D,
                    ):
        self.eval()
        # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
        observations, actions, rewards, nonterminals = D.sample(
                self.cfg.train.batch_size, self.cfg.train.chunk_size)  # Transitions start at time t = 0

        with torch.cuda.amp.autocast(enabled=self.cfg.train.use_amp):
            actor_loss = self.calc_loss(observations, actions)
        
        # Log loss info
        loss_info = dict()
        loss_info["actor_loss"] = actor_loss.item()

        if self.cfg.main.wandb:
            for name in loss_info.keys():        
                wandb.log(data={"{}/validation".format(name):loss_info[name]}, step=self.itr_optim)
            
        self.train()
    
    def load_state_dict(self, state_dict):
        self.actor_model.load_state_dict(state_dict['actor_model'])
    
    def get_state_dict(self):
        state_dict = {'actor_model': self.actor_model.state_dict(),
                     }
        return state_dict

    def save_model(self, results_dir, itr):
        state_dict = self.get_state_dict()
        torch.save(state_dict, os.path.join(results_dir, 'models_%d.pth' % itr))

    def step_env_and_act(self, env, action, observation, explore=False):
        for key in observation.keys():
            if not torch.is_tensor(observation[key]):
                observation[key] = torch.tensor(observation[key], dtype=torch.float32)
            observation[key] = observation[key].to(device=self.device)
        
        # Get action from planner(q(s_t|o≤t,a<t), p)
        action = self.planner.get_action(observation[self.cfg.model.observation_names_enc[0]], det=not(explore))
        
        if explore:
            # Add gaussian exploration noise on top of the sampled action
            action = torch.clamp(
                Normal(action, self.cfg.train.action_noise).rsample(), -1, 1)
            # action = action + cfg.train.action_noise * torch.randn_like(action)  # Add exploration noise ε ~ p(ε) to the action
        next_observation, reward, done = env.step(action.cpu() if isinstance(
            env, MultimodalEnvBatcher) else action[0].cpu())  # Perform environment step (action repeats handled internally)
        return action, next_observation, reward, done


    def get_test_episode(self, cfg, device, seed=3000, vis_pbar=True, output_states=True):
        # Set models to eval mode
        self.eval()
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
            observations.append(self.observation2np(observation))
            pbar = range(cfg.env.env_config.horizon // cfg.env.action_repeat)
            if vis_pbar:
                pbar = tqdm(pbar, desc="Test model", leave=False)
            
            for t in pbar:
                action, next_observation, reward, done = self.step_env_and_act(
                    test_envs, action, observation)
                total_rewards += reward.numpy()
                observation = next_observation
                
                rewards.append(reward.detach().cpu().numpy())
                dones.append(done.detach().cpu().numpy())
                actions.append(action.detach().cpu().numpy())

                observations.append(self.observation2np(observation))
                
                if done.sum().item() == cfg.main.test_episodes:
                    if vis_pbar:
                        pbar.close()
                    break

            episode_info['observation'] = observations
            episode_info['reward'] = rewards
            episode_info['done'] = dones
            episode_info['action'] = actions
            episode_info['seed'] = seed
            

        # Set models to train mode
        self.train()
        # Close test environments
        test_envs.close()

        return episode_info