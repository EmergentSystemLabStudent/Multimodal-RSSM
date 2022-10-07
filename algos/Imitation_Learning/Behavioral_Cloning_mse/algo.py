import torch
from torch import nn, optim

from common.models.policy import ValueModel, ActorModel
from algos.Imitation_Learning.base.algo import IL_base
from algos.base.base import Controller_base

import wandb

class Controller(Controller_base):
    def __init__(self, 
                 cfg, 
                 device=torch.device("cpu"), 
                 horizon=1) -> None:
        super().__init__(cfg, IL, device, horizon)

class IL(IL_base):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        print("Behavioral Cloning")
        
        self.init_models(device)
        self.init_param_list()
        self.init_optimizer()

        if cfg.main.algo == "MPC":
            print("MPC")
            from algos.base.planner import MPCPlanner
            self.planner = MPCPlanner(cfg.env.action_size, cfg.model.planning_horizon, cfg.planner.optimisation_iters,
                                cfg.planner.candidates, cfg.planner.top_candidates, self.rssm.transition_model, self.rssm.reward_model)
        else:
            print("Actor")
            self.planner = self.actor_model
        
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
    
    def calc_loss(self,
                  actions,
                  nonterminals,
                  states, 
                  ):
        return self.calc_BC_mse_loss(actions, nonterminals, states)        

    def get_state_dict(self):
        state_dict = {'actor_model': self.actor_model.state_dict(),
                      'value_model': self.value_model.state_dict(),
                      'rssm': self.rssm.get_state_dict(),
                     }
        return state_dict
