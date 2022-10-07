import os

from tqdm import tqdm

from algos.base.memory import ExperienceReplay_Multimodal, load_dataset
from algos.MRSSM.MRSSM.algo import build_RSSM

from common.logger import end_logger, setup

def train(cfg, cwd, results_dir, device):
    # Initialise training environment and experience replay memory
    print("Initialise training environment and experience replay memory")
    if cfg.env.action_size == None:
        from common.env import MultimodalControlSuiteEnv
        env = MultimodalControlSuiteEnv(cfg.env.env_config, cfg.env.symbolic_env, cfg.main.seed,
            cfg.env.env_config.horizon, cfg.env.action_repeat, cfg.env.bit_depth, cfg.env.info_names,
            noise_scale=cfg.env.noise_scale, add_noise=cfg.env.add_noise, noisy_background=cfg.env.noisy_background)
        cfg.env.action_size = env.action_size

    observation_names = list(set(cfg.model.observation_names_enc+cfg.model.observation_names_rec))
    D = ExperienceReplay_Multimodal(size=cfg.train.experience_size,
                                    observation_names=observation_names,
                                    observation_shapes=cfg.env.observation_shapes,
                                    n_crop=cfg.train.n_crop,
                                    noise_scales=cfg.train.noise_scales,
                                    pca_scales=cfg.train.pca_scales,
                                    dh_base=cfg.train.dh_base,
                                    dw_base=cfg.train.dw_base,
                                    action_name=cfg.env.action_name,
                                    action_size=cfg.env.action_size,
                                    bit_depth=cfg.env.bit_depth,
                                    device=device,
                                    taskset_name=cfg.env.taskset_name)
    load_dataset(cfg, cwd, D, cfg.train.experience_replay)

    D_val = ExperienceReplay_Multimodal(size=cfg.train.experience_size,
                                        observation_names=observation_names,
                                        observation_shapes=cfg.env.observation_shapes,
                                        n_crop=cfg.train.n_crop,
                                        noise_scales=cfg.train.noise_scales,
                                        pca_scales=cfg.train.pca_scales,
                                        dh_base=cfg.train.dh_base,
                                        dw_base=cfg.train.dw_base,
                                        action_name=cfg.env.action_name,
                                        action_size=cfg.env.action_size,
                                        bit_depth=cfg.env.bit_depth,
                                        device=device,
                                        taskset_name=cfg.env.taskset_name)
    load_dataset(cfg, cwd, D_val, cfg.train.validation_data)    

    # Initialise model parameters randomly
    print("Initialise model parameters randomly")
    model = build_RSSM(cfg, device)
    
    model_path = os.path.join(cwd, cfg.main.models)
    if cfg.main.models != '':
        if os.path.exists(model_path):
            model.load_model(model_path)
        else:
            print("model was not loaded.")
            raise NotImplementedError("{} is not exist".format(model_path))
    
    if cfg.train.rssm.load:
        if cfg.train.rssm.model_path==None:
            raise NotImplementedError
        model_path = os.path.join(cwd, cfg.train.rssm.model_path)
        if os.path.exists(model_path):
            print("load RSSM from {}".format(model_path))
            model.load_model(model_path)
        else:
            raise NotImplementedError("{} is not exist".format(model_path))
    
    for itr in tqdm(range(1, cfg.train.train_iteration+1), desc="train"):
        
        model.optimize(D)

        if itr % cfg.main.validation_interval == 0:
            model.validation(D_val)

        # Checkpoint models
        if itr % cfg.main.checkpoint_interval == 0:
            model.save_model(results_dir, itr)

def run(cfg):
    # init logger and set seeds
    cwd, results_dir, device = setup(cfg)
    # ---------- train RSSM ----------
    train(cfg, cwd, results_dir, device)

    # ---------- ML Flow log ----------
    end_logger(cfg)
