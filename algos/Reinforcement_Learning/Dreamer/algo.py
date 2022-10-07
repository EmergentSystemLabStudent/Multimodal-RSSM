
import os
from tqdm import tqdm

import torch
from torch import nn, optim
from common.utils import imagine_ahead, lambda_return, FreezeParameters
from common.models.policy import ValueModel, ActorModel
from algos.base.planner import MPCPlanner

from algos.base.base import Model_base
import wandb

class RL(Model_base):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)

        self.init_models(device)
        self.init_param_list()
        self.init_optimizer()

        if cfg.main.algo == "dreamer":
            print("DREAMER")
            self.planner = self.actor_model
        else:
            self.planner = MPCPlanner(cfg.env.action_size, cfg.model.planning_horizon, cfg.planner.optimisation_iters,
                                cfg.planner.candidates, cfg.planner.top_candidates, self.rssm.transition_model, self.rssm.reward_model)
        

    def init_models(self, device):
        self.actor_model = ActorModel(self.cfg.model.belief_size, self.cfg.model.state_size, self.cfg.model.hidden_size,
                                self.cfg.env.action_size, self.cfg.model.activation_function.dense).to(device=device)
        self.value_model = ValueModel(self.cfg.model.belief_size, self.cfg.model.state_size, self.cfg.model.hidden_size,
                                self.cfg.model.activation_function.dense).to(device=device)

    def init_param_list(self):
        self.value_actor_param_list = list(self.value_model.parameters()) \
            + list(self.actor_model.parameters())
        
    def init_optimizer(self):
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(
        ), lr=0 if self.cfg.model.learning_rate_schedule != 0 else self.cfg.model.actor_learning_rate, eps=self.cfg.model.adam_epsilon)
        self.value_optimizer = optim.Adam(self.value_model.parameters(
        ), lr=0 if self.cfg.model.learning_rate_schedule != 0 else self.cfg.model.value_learning_rate, eps=self.cfg.model.adam_epsilon)

    def optimize_loss(self, 
                      states, 
                      itr_optim
                      ):

        beliefs             = states["beliefs"]
        posterior_states    = states["posterior_states"]

        # Dreamer implementation: actor loss calculation and optimization
        with torch.no_grad():
            actor_states = posterior_states.detach()
            actor_beliefs = beliefs.detach()
        with FreezeParameters(self.rssm.model_modules):
            imagination_traj = imagine_ahead(
                actor_states, actor_beliefs, self.actor_model, self.rssm.transition_model, self.cfg.model.planning_horizon)
        imged_beliefs, imged_prior_states, imged_prior_means, imged_prior_std_devs = imagination_traj
        with FreezeParameters(self.rssm.model_modules + self.value_model.modules):
            imged_reward = self.rssm.reward_model(
                h_t=imged_beliefs, s_t=imged_prior_states)['loc']
            value_pred = self.value_model(
                h_t=imged_beliefs, s_t=imged_prior_states)['loc']
        returns = lambda_return(imged_reward, value_pred,
                                bootstrap=value_pred[-1], discount=self.cfg.model.discount, lambda_=self.cfg.model.disclam)
        actor_loss = -torch.mean(returns)
        # Update model parameters
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_model.parameters(),
                                self.cfg.model.grad_clip_norm, norm_type=2)
        self.actor_optimizer.step()

        # Dreamer implementation: value loss calculation and optimization
        with torch.no_grad():
            value_beliefs = imged_beliefs.detach()
            value_prior_states = imged_prior_states.detach()
            target_return = returns.detach()
        # detach the input tensor from the transition network.
        value_loss = - self.value_model.get_log_prob(value_beliefs, value_prior_states, target_return)
        value_loss = value_loss.mean(dim=(0, 1))
        # Update model parameters
        self.value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value_model.parameters(),
                                self.cfg.model.grad_clip_norm, norm_type=2)
        self.value_optimizer.step()

        # Log loss info
        loss_info = dict()
        loss_info["actor_loss"] = actor_loss.item()
        loss_info["value_loss"] = value_loss.item()
        
        if self.cfg.main.wandb:
            for name in loss_info.keys():
                wandb.log(data={"{}/train".format(name):loss_info[name]}, step=itr_optim)
            wandb.log(data={"itr":itr_optim}, step=itr_optim)
        
    def optimize_step(self, cfg, D, itr_optim):
        
        # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
        observations_raw, actions, rewards, nonterminals = D.sample(
            cfg.train.batch_size, cfg.train.chunk_size)  # Transitions start at time t = 0
        
        # observations = dict()
        # for key in self.cfg.model.observation_names_enc:
        #     observations[key] = observations_raw[key]
        # observations_target = self.clip_obs(observations, idx_start=1)
        observations_target = self.clip_obs(observations_raw, idx_start=1)

        states = self.rssm.estimate_state(observations_target, actions[:-1], rewards, nonterminals[:-1])
        
        self.rssm.optimize_loss(observations_target,
                                actions, 
                                rewards, 
                                nonterminals,
                                states,
                                itr_optim
                                )

        self.optimize_loss(states, 
                            itr_optim
                            )

    def optimize(self, cfg, D):
        for s in tqdm(range(cfg.train.collect_interval), leave=False):
            self.itr_optim += 1
            self.optimize_step(cfg, D, self.itr_optim)
    
    def get_state_dict(self):
        state_dict = {'actor_model': self.actor_model.state_dict(),
                      'value_model': self.value_model.state_dict(),
                      'rssm': self.rssm.get_state_dict(),
                     }
        return state_dict

