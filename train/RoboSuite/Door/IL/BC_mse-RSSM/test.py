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
