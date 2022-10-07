import torch

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
        print("Behavioral Cloning (MSE)")
        
        self.init_models(device)
        self.init_param_list()
        self.init_optimizer()

        self.planner = self.actor_model
        
    def calc_loss(self,
                  actions,
                  nonterminals,
                  states, 
                  ):
        return self.calc_BC_mse_loss(actions, nonterminals, states)
    