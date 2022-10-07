import os
import numpy as np
from tqdm import tqdm

import torch
from torch.distributions import Normal

from common.env import MultimodalEnvBatcher, MultimodalControlSuiteEnv
from common.models.encoder import bottle_tupele, bottle_tupele_multimodal
from common.models.policy import ValueModel, ActorModel
from algos.MRSSM.MRSSM.algo import build_RSSM
from algos.base.processing import preprocess_pose_seq, postprocess_pose

import copy
import wandb

import ray

class Controller_base:
    def __init__(self, 
                 cfg, 
                 policy_cls,
                 device=torch.device("cpu"), 
                 horizon=1) -> None:
        self.cfg = cfg
        self.device = device
        self.model = policy_cls(self.cfg, self.device)
        self.horizon = horizon
        T = horizon + 2
        self.beliefs = [torch.empty(0)] * T
        self.prior_states = [torch.empty(0)] * T
        self.posterior_states = [torch.empty(0)] * T

        self.next_init_state = None
        self.beliefs[0] = torch.zeros(1, self.cfg.model.belief_size, device=self.device)
        self.posterior_states[0] = torch.zeros(1, self.cfg.model.state_size, device=self.device)
        self.prev_action = torch.zeros(1, self.cfg.env.action_size, dtype=torch.float32, device=self.device)
        self.pose_prev = torch.zeros(1, *self.cfg.env.observation_shapes["pose_quat"], device=self.device)

    def __call__(self, observations, dummy_action):
        return self.get_action(observations)

    def load_model(self, model_path):
        self.model.load_model(model_path)

    def get_action(self, observations, det=True):
        observations = preprocess_pose_seq(observations, self.pose_prev)
        self.pose_prev = observations["pose_quat"].unsqueeze(0)

        if self.next_init_state is not None:
            self.beliefs[0], self.posterior_states[0] = self.next_init_state
        _state = self.posterior_states[0]

        obs_emb = self.model.rssm.encoder(observations)
        for name in obs_emb.keys():
            obs_emb[name] = obs_emb[name].unsqueeze(0)
        beliefs, _, _, _, posterior_states, posterior_mean, _, _, _ = self.model.rssm.transition_model(_state, self.prev_action.unsqueeze(0), self.beliefs[0], obs_emb, None, det)

        self.beliefs[1] = beliefs.squeeze(0)
        if det:
            self.posterior_states[1] = posterior_mean.squeeze(0)
        else:
            self.posterior_states[1] = posterior_states.squeeze(0)

        self.next_init_state = (self.beliefs[1], self.posterior_states[1])

        action = self.model.planner.get_action(belief=self.beliefs[1], state=self.posterior_states[1], det=True)
        

        self.prev_action = action.detach().clone()#.unsqueeze(0)
        _pose_pred = postprocess_pose(self.cfg.env.action_name, action)
        print("action {}".format(_pose_pred))
        action_size = _pose_pred.size(1)
        if "d_pose_quat" in self.cfg.env.action_name:
            pose_pred = copy.deepcopy(observations["pose_quat"])
            pose_pred[:,:action_size] += _pose_pred
            # pose_pred[:,2] += _pose_pred[:,2] # one-hole-drilling用の一時的なやつ　すぐ直す
        else:
            pose_pred = copy.deepcopy(observations["pose_quat"])
            pose_pred[:,:action_size] = _pose_pred

        print("pose_quat:{}".format(observations["pose_quat"]))
        print("d_pose:{}".format(pose_pred-observations["pose_quat"]))
        return pose_pred

@ray.remote
def get_test_episode_ray(model, cfg, device, seed, vis_pbar=False, output_states=True):
    result = model.get_test_episode(cfg=cfg, device=device, seed=seed, vis_pbar=vis_pbar, output_states=output_states)
    return result

class Model_base:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

        self.rssm = build_RSSM(cfg, device)
        if cfg.model.multimodal:
            self.bottle_tupele = bottle_tupele_multimodal
        else:
            self.bottle_tupele = bottle_tupele

        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.use_amp)
        self.itr_optim = 0
    
    def eval(self):
        self.rssm.eval()
        self.actor_model.eval()
        self.value_model.eval()

    def train(self):
        self.rssm.train()
        self.actor_model.train()
        self.value_model.train()

    def init_models(self, device):
        raise NotImplementedError
        self.actor_model = ActorModel(self.cfg.model.belief_size, self.cfg.model.state_size, self.cfg.model.hidden_size,
                                self.cfg.env.action_size, self.cfg.model.activation_function.dense).to(device=device)
        self.value_model = ValueModel(self.cfg.model.belief_size, self.cfg.model.state_size, self.cfg.model.hidden_size,
                                self.cfg.model.activation_function.dense).to(device=device)


    def init_param_list(self):
        raise NotImplementedError
        
    def init_optimizer(self):
        raise NotImplementedError
    
    def clip_obs(self, observations, idx_start=0, idx_end=None):
        output = dict()
        for k in observations.keys():
            output[k] = observations[k][idx_start:idx_end]
        return output

    def estimate_state(self, observations, actions, rewards, nonterminals):
        return self.rssm.estimate_state(observations, actions, rewards, nonterminals)

    def optimize_loss(self, 
                      actions, 
                      states, 
                      itr):
        raise NotImplementedError
    
    def optimize_step(self, cfg, D, step):
        raise NotImplementedError
        
    def optimize(self, cfg, D, step):
        raise NotImplementedError
    
    def validation(self,
                 D, 
                 ):
        raise NotImplementedError

    def load_model_dicts(self, model_path):
        print("load model_dicts from {}".format(model_path))
        return torch.load(model_path, map_location=torch.device(self.device))

    def load_rssm(self, model_path):
        model_dicts = self.load_model_dicts(model_path)
        try:
            self.rssm.load_state_dict(model_dicts)
        except:
            self.rssm.load_state_dict(model_dicts["rssm"])
        self.rssm._init_optimizer()
    
    def load_state_dict(self, state_dict):
        self.actor_model.load_state_dict(state_dict['actor_model'])
        self.value_model.load_state_dict(state_dict['value_model'])
        self.rssm.load_state_dict(state_dict["rssm"])

    def load_model(self, model_path):
        model_dicts = self.load_model_dicts(model_path)
        self.load_state_dict(model_dicts)
        self.init_optimizer()

    def get_state_dict(self):
        raise NotImplementedError

    def save_model(self, results_dir, itr):
        state_dict = self.get_state_dict()
        torch.save(state_dict, os.path.join(results_dir, 'models_%d.pth' % itr))

    def update_belief(self, belief, posterior_state, action, observation):
        for key in observation.keys():
            if not torch.is_tensor(observation[key]):
                observation[key] = torch.tensor(observation[key], dtype=torch.float32)
            observation[key] = observation[key].unsqueeze(dim=0).to(device=self.device)
        
        # Infer belief over current state q(s_t|o≤t,a<t) from the history
        obs_emb = self.bottle_tupele(self.rssm.encoder, observation)
        belief, _, _, _, posterior_state, _, _, _, _ = self.rssm.transition_model(posterior_state, action.unsqueeze(
            dim=0), belief, obs_emb)  # Action and observation need extra time dimension
        belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(
            dim=0)  # Remove time dimension from belief/state
        return belief, posterior_state

    def update_belief_and_act(self, env, belief, posterior_state, action, observation, explore=False):
        belief, posterior_state = self.update_belief(belief, posterior_state, action, observation)

        # Get action from planner(q(s_t|o≤t,a<t), p)
        action = self.planner.get_action(belief=belief, state=posterior_state, det=not(explore))
        
        if explore:
            # Add gaussian exploration noise on top of the sampled action
            action = torch.clamp(
                Normal(action, self.cfg.train.action_noise).rsample(), -1, 1)
            # action = action + cfg.train.action_noise * torch.randn_like(action)  # Add exploration noise ε ~ p(ε) to the action
        next_observation, reward, done = env.step(action.cpu() if isinstance(
            env, MultimodalEnvBatcher) else action[0].cpu())  # Perform environment step (action repeats handled internally)
        return belief, posterior_state, action, next_observation, reward, done

    def observation2np(self, observation):
        output = dict()
        for name in observation.keys():
            output[name] = observation[name].detach().cpu().numpy()
        return output

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
            beliefs = []
            posterior_states = []
            observation, total_rewards = test_envs.reset(), np.zeros((cfg.main.test_episodes, ))
            belief, posterior_state, action = torch.zeros(cfg.main.test_episodes, cfg.model.belief_size, device=device), torch.zeros(
                cfg.main.test_episodes, cfg.model.state_size, device=device), torch.zeros(cfg.main.test_episodes, test_envs.action_size, device=device)
            observations.append(self.observation2np(observation))
            beliefs.append(belief)
            posterior_states.append(posterior_state)
            pbar = range(cfg.env.env_config.horizon // cfg.env.action_repeat)
            if vis_pbar:
                pbar = tqdm(pbar, desc="Test model", leave=False)
            
            for t in pbar:
                belief, posterior_state, action, next_observation, reward, done = self.update_belief_and_act(
                    test_envs, belief, posterior_state, action, observation)
                total_rewards += reward.numpy()
                observation = next_observation
                
                rewards.append(reward.detach().cpu().numpy())
                dones.append(done.detach().cpu().numpy())
                actions.append(action.detach().cpu().numpy())

                observations.append(self.observation2np(observation))
                beliefs.append(belief)
                posterior_states.append(posterior_state)

                if done.sum().item() == cfg.main.test_episodes:
                    if vis_pbar:
                        pbar.close()
                    break

            episode_info['observation'] = observations
            episode_info['reward'] = rewards
            episode_info['done'] = dones
            episode_info['action'] = actions
            episode_info['seed'] = seed
            if output_states:
                episode_info['beliefs'] = beliefs
                episode_info['posterior_states'] = posterior_states
            

        # Set models to train mode
        self.train()
        # Close test environments
        test_envs.close()

        return episode_info
    
    def get_test_episode_ray(self, cfg, device, seed=3000, vis_pbar=False, n_process=5):
        n_test_episode = cfg.main.test_episodes

        ray.init(num_cpus=n_process)
        device=torch.device("cpu")
        model = Model_base(self.cfg, device=device)
        model.init_models(device)
        model.planner = model.actor_model
        model.load_state_dict(self.get_state_dict())

        work_in_progresses = [get_test_episode_ray.remote(model, cfg, device, seed, vis_pbar) for seed in range(seed, seed+n_test_episode)]
        
        episode_info = []
        for i in tqdm(range(n_test_episode)):
            finished, work_in_progresses = ray.wait(work_in_progresses, num_returns=1)
            
            result = ray.get(finished[0])
            episode_info.append(result)
        ray.shutdown()

        episodes = dict()
        for name in episode_info[0].keys():
            episodes[name] = np.stack([episode_info[i][name] for i in range(n_test_episode)])
        
        return episodes

    def test_model(self, cfg, device, itr=0):
        # get episode info
        seed = (3000 + (itr*cfg.main.test_episodes)%100)
        episode_info = self.get_test_episode(cfg, device, seed=seed)
        total_rewards = np.sum(np.array(episode_info['reward']), axis=0).tolist()
        seed_ep = episode_info['seed']
        test_reward_mean = np.mean(total_rewards)
        test_reward_max = np.max(total_rewards)
        test_reward_min = np.min(total_rewards)
        tqdm.write("test rewards (seed: {}) mean: {}, max: {}, min: {}".format(seed_ep, test_reward_mean, test_reward_max, test_reward_min))
        if cfg.main.wandb:
            log_data = {
                "reward/test_mean":test_reward_mean,
                "reward/test_max":test_reward_max,
                "reward/test_min":test_reward_min,
            }                
            wandb.log(data=log_data, step=self.itr_optim)

    def data_collection(self, cfg, env, D, device):
        with torch.no_grad():
            observation, total_reward = env.reset(), 0
            belief, posterior_state, action = torch.zeros(1, cfg.model.belief_size, device=device), torch.zeros(
                1, cfg.model.state_size, device=device), torch.zeros(1, env.action_size, device=device)
            pbar = tqdm(range(cfg.env.env_config.horizon // cfg.env.action_repeat), desc="Data collection", leave=False)
            for t in pbar:
                _observation = dict()
                for key in observation.keys():
                    _observation[key] = np.expand_dims(observation[key], 0)
                    
                belief, posterior_state, action, next_observation, reward, done = self.update_belief_and_act(
                    env, belief, posterior_state, action, _observation, explore=True)
                D.append(observation, action.cpu(), reward, done)
                total_reward += reward
                observation = next_observation
                if cfg.main.render:
                    env.render()
                if done:
                    pbar.close()
                    break
        if cfg.main.wandb:
            log_data = {
                "reward/train":total_reward,
            }                
            wandb.log(data=log_data, step=self.itr_optim)
        return total_reward, t

    def test(self, cfg, env, device):
        # Set models to eval mode
        self.rssm.transition_model.eval()
        self.rssm.reward_model.eval()
        self.rssm.encoder.eval()
        with torch.no_grad():
            total_reward = 0
            for _ in tqdm(range(cfg.main.test_episodes), desc="Test", leave=False):
                observation = env.reset()
                belief, posterior_state, action = torch.zeros(1, cfg.model.belief_size, device=device), torch.zeros(
                    1, cfg.model.state_size, device=device), torch.zeros(1, env.action_size, device=device)
                pbar = tqdm(range(cfg.env.env_config.horizon // cfg.env.action_repeat))
                for t in pbar:
                    belief, posterior_state, action, observation, reward, done = self.update_belief_and_act(
                        env, belief, posterior_state, action, observation)
                    total_reward += reward
                    if cfg.main.render:
                        env.render()
                    if done:
                        pbar.close()
                        break
        print('Average Reward:', total_reward / cfg.main.test_episodes)

