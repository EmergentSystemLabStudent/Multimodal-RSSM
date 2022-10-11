#Please do this !!!
#export HYDRA_FULL_ERROR=1
import sys
import os
from pathlib import Path
module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../../..')
sys.path.append(module_path)
os.environ['PYTHONPATH'] = module_path

from hydra import initialize, compose

from algos.MRSSM.MRSSM.algo import build_RSSM
from utils.evaluation.estimate_states import run

def multi_run(path):
    dirpath = os.path.join(os.path.dirname(__file__), path)
    files = os.listdir(dirpath)
    foders_dir = [f for f in files if os.path.isdir(os.path.join(dirpath, f))]
    print(files, foders_dir)

    device = "cpu"
    # device = "cuda:0"
    itr = 10_000

    for folder_dir in foders_dir:
        files_path = os.listdir(dirpath+"/"+folder_dir)
        if (not "states" in files_path) and ("hydra_config.yaml" in files_path):
            with initialize(path+"/"+folder_dir):
                cfg = compose(config_name="hydra_config")
            cfg.main.device = device
            cfg.main.wandb = False
            
            log_dir = cfg.main.log_dir

            if "states" in os.listdir(log_dir):
                continue
            model_path = "{}/models_{}.pth".format(log_dir, itr)
            run(cfg, cwd=".", model_class=build_RSSM, device=device, model_path=model_path)

def main():
    # set configs like MRSSM/MRSSM/eval_targets/hoge/hydra_config.yaml
    # estimated states are saved to the model_path (loaded model folders).
    multi_run(path="eval_targets")

if __name__=="__main__":
    main()
