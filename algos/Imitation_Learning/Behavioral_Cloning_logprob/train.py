from .algo import IL
from algos.Imitation_Learning.base.train import run_base

def run(cfg):
    run_base(cfg, IL)
