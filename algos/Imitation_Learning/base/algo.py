import torch
from torch import nn
from torch.nn import functional as F

from common.utils import imagine_ahead, FreezeParameters

from common.models.policy import ActorModel
from algos.base.base import Model_base

import wandb

class IL_base(Model_base):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
    
    def init_models(self, device):
        self.actor_model = ActorModel(self.cfg.model.belief_size, self.cfg.model.state_size, self.cfg.model.hidden_size,
                                self.cfg.env.action_size, self.cfg.model.activation_function.dense).to(device=device)
    
    def init_optimizer(self):
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(
        ), lr=0 if self.cfg.model.learning_rate_schedule != 0 else self.cfg.model.actor_learning_rate, eps=self.cfg.model.adam_epsilon)

    def get_state_dict(self):
        state_dict = {'actor_model': self.actor_model.state_dict(),
                      'rssm': self.rssm.get_state_dict(),
                     }
        return state_dict

    def calc_BC_logprob_loss(self, actions, nonterminals, states):
        with torch.cuda.amp.autocast(enabled=self.cfg.train.use_amp):
            beliefs             = states["beliefs"]
            posterior_states    = states["posterior_states"]

            with torch.no_grad():
                actor_states = posterior_states.detach()
                actor_beliefs = beliefs.detach()
            
            logprobs = []
            for t in range(0, self.cfg.train.chunk_size - 1):
                logprob = self.actor_model.pie.get_log_prob(s_t=actor_states[t], h_t=actor_beliefs[t], a_t=actions[t+1]).sum(dim=-1)
                logprobs.append(logprob)
            actor_loss = - torch.mean(torch.vstack(logprobs))
        return actor_loss

    def calc_BC_mse_loss(self, actions, nonterminals, states):
        with torch.cuda.amp.autocast(enabled=self.cfg.train.use_amp):
            beliefs             = states["beliefs"]
            posterior_states    = states["posterior_states"]

            with torch.no_grad():
                actor_states = posterior_states.detach()
                actor_beliefs = beliefs.detach()
            
            actor_loss = []
            for t in range(0, self.cfg.train.chunk_size - 1):
                action_pred = self.actor_model.get_action(state=actor_states[t], belief=actor_beliefs[t])
                mse = F.mse_loss(action_pred, actions[t+1], reduction='none').sum(dim=-1).mean()
                actor_loss.append(mse)
            actor_loss = torch.mean(torch.vstack(actor_loss))
        return actor_loss

    def optimize(self,
                 D, 
                 ):
        self.itr_optim += 1
        # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
        observations, actions, rewards, nonterminals = D.sample(
                self.cfg.train.batch_size, self.cfg.train.chunk_size)  # Transitions start at time t = 0

        observations_target = self.clip_obs(observations, idx_start=1)

        with torch.cuda.amp.autocast(enabled=self.cfg.train.use_amp):
            states = self.rssm.estimate_state(observations_target, actions[:-1], rewards, nonterminals[:-1])
        
        if not self.cfg.train.rssm.fix:
            self.rssm.optimize_loss(observations_target,
                                    actions, 
                                    rewards, 
                                    nonterminals,
                                    states,
                                    self.itr_optim)
        
        self.optimize_loss(actions, 
                            nonterminals,
                            states, 
                            self.itr_optim)

    def calc_loss(self,
                  actions,
                  nonterminals,
                  states):
        raise NotImplementedError
    
    def optimize_loss(self, 
                      actions, 
                      nonterminals, 
                      states, 
                      itr_optim):
        with torch.cuda.amp.autocast(enabled=self.cfg.train.use_amp):
            actor_loss = self.calc_loss(actions, nonterminals, states)
        
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
                bc_logprob_loss = self.calc_BC_logprob_loss(actions, nonterminals, states)
                wandb.log(data={"BC_logprob_loss/train":bc_logprob_loss.item()}, step=itr_optim)
                bc_mse_loss = self.calc_BC_mse_loss(actions, nonterminals, states)
                wandb.log(data={"BC_mse_loss/train":bc_mse_loss.item()}, step=itr_optim)
                
    def validation(self,
                    D,
                    ):
        self.eval()
        # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
        observations, actions, rewards, nonterminals = D.sample(
                self.cfg.train.batch_size, self.cfg.train.chunk_size)  # Transitions start at time t = 0

        observations_target = self.clip_obs(observations, idx_start=1)

        with torch.cuda.amp.autocast(enabled=self.cfg.train.use_amp):
            states = self.rssm.estimate_state(observations_target, actions[:-1], rewards, nonterminals[:-1])
            actor_loss = self.calc_loss(actions, nonterminals, states)
            if not self.cfg.train.rssm.fix:
                rssm_loss, rssm_loss_info = self.rssm._get_model_loss(observations_target, actions, rewards, nonterminals, states)

        # Log loss info
        loss_info = dict()
        loss_info["actor_loss"] = actor_loss.item()

        if self.cfg.main.wandb:
            for name in loss_info.keys():        
                wandb.log(data={"{}/validation".format(name):loss_info[name]}, step=self.itr_optim)
            if not self.cfg.train.rssm.fix:
                for name in rssm_loss_info.keys():
                    wandb.log(data={"{}/validation".format(name):rssm_loss_info[name]}, step=self.itr_optim)

        self.train()