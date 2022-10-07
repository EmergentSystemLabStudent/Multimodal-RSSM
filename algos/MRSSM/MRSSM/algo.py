import os

import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F

import wandb

from common.models.encoder import bottle_tupele, bottle_tupele_multimodal, MultimodalEncoder, MultimodalStochasticEncoder, poe, calc_subset_states, mopoe, build_Encoder
from common.models.observation_model import MultimodalObservationModel, build_ObservationModel
from common.models.reward_model import RewardModel
from common.models.transition_model import TransitionModel, MultimodalTransitionModel

def build_RSSM(cfg, device):
    if cfg.model.multimodal:
        if cfg.model.multimodal_params.fusion_method == "NN":
            rssm = MRSSM_NN(cfg, device)
        elif cfg.model.multimodal_params.fusion_method == "PoE":
            rssm = MRSSM_PoE(cfg, device)
        elif cfg.model.multimodal_params.fusion_method == "MoPoE":
            rssm = MRSSM_MoPoE(cfg, device)
        else:
            raise NotImplementedError
    else:
        rssm = RSSM(cfg, device)
    return rssm

class RSSM_base(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.observation_name = cfg.model.observation_names_enc[0]
        self._init_models(device)
        self._init_param_list()
        self._init_optimizer()
        
        self.global_prior = Normal(torch.zeros(cfg.train.batch_size, cfg.model.state_size, device=device), torch.ones(
                                    cfg.train.batch_size, cfg.model.state_size, device=device))  # Global prior N(0, I)
        # Allowed deviation in KL divergence
        self.free_nats = torch.full((1, ), cfg.model.free_nats, device=device)

        self.model_modules = self.transition_model.modules + self.encoder.modules + \
            self.observation_model.modules + self.reward_model.modules

        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.use_amp)
        self.itr_optim = 0

    def _init_models(self, device):
        raise NotImplementedError
            
    def _init_param_list(self):
        raise NotImplementedError
                    
    def _init_optimizer(self):
        self.model_optimizer = optim.Adam(self.param_list, lr=0 if self.cfg.model.learning_rate_schedule !=
                                    0 else self.cfg.model.model_learning_rate, eps=self.cfg.model.adam_epsilon)

    def get_state_dict(self):
        raise NotImplementedError

    def _load_model_dicts(self, model_path):
        print("load model_dicts from {}".format(model_path))
        return torch.load(model_path, map_location=torch.device(self.device))
    
    def load_model(self, model_path):
        model_dicts = self._load_model_dicts(model_path)
        self.load_state_dict(model_dicts)
        self._init_optimizer()

    def save_model(self, results_dir, itr):
        state_dict = self.get_state_dict()
        torch.save(state_dict, os.path.join(results_dir, 'models_%d.pth' % itr))

    def _clip_obs(self, observations, idx_start=0, idx_end=None):
        output = dict()
        for k in observations.keys():
            output[k] = observations[k][idx_start:idx_end]
        return output

    def estimate_state(self,
                        observations,
                        actions, 
                        rewards, 
                        nonterminals,
                        batch_size=None,
                        det=False):
        raise NotImplementedError
    
    def _calc_kl(self,
                states,
                ):
        prior_means         = states["prior_means"]
        prior_std_devs      = states["prior_std_devs"]
        posterior_means     = states["posterior_means"]
        posterior_std_devs  = states["posterior_std_devs"]

        if self.cfg.model.kl_balancing_alpha is None:
            div = kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(
                prior_means, prior_std_devs)).sum(dim=2)
        else:
            kl1 = kl_divergence(Normal(posterior_means.detach(), posterior_std_devs.detach()), Normal(
                prior_means, prior_std_devs)).sum(dim=2)
            kl2 = kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(
                prior_means.detach(), prior_std_devs.detach())).sum(dim=2)
            div = self.cfg.model.kl_balancing_alpha * kl1 + (1-self.cfg.model.kl_balancing_alpha) * kl2
        # Note that normalization by overshooting distance and weighting by overshooting distance cancel out
        kl_loss = torch.max(div, self.free_nats).mean(dim=(0, 1))
        return kl_loss

    def _calc_reward_loss(self,
                          rewards,
                          beliefs,
                          posterior_states,
                          ):
        if self.cfg.model.worldmodel_LogProbLoss:
            reward_loss = -self.reward_model.get_log_prob(beliefs, posterior_states, rewards[:-1])
            reward_loss = reward_loss.mean(dim=(0, 1))
        else:
            reward_mean = self.reward_model(
                h_t=beliefs, s_t=posterior_states)['loc']
            reward_loss = F.mse_loss(reward_mean, rewards[:-1], reduction='none').mean(dim=(0, 1))

        return reward_loss

    def _latent_overshooting(self,
                             actions,
                             rewards,
                             nonterminals,
                             states,
                             ):
        beliefs             = states["beliefs"]
        prior_states        = states["prior_states"]
        posterior_states, posterior_means, posterior_std_devs = self._get_posterior_states(states)
        

        kl_loss_sum = torch.tensor(0., device=self.cfg.main.device)
        reward_loss = torch.tensor(0., device=self.cfg.main.device)

        overshooting_vars = []  # Collect variables for overshooting to process in batch
        for t in range(1, self.cfg.train.chunk_size - 1):
            d = min(t + self.cfg.model.overshooting_distance,
                    self.cfg.train.chunk_size - 1)  # Overshooting distance
            # Use t_ and d_ to deal with different time indexing for latent states
            t_, d_ = t - 1, d - 1
            # Calculate sequence padding so overshooting terms can be calculated in one batch
            seq_pad = (0, 0, 0, 0, 0, t - d + self.cfg.model.overshooting_distance)
            # Store (0) actions, (1) nonterminals, (2) rewards, (3) beliefs, (4) prior states, (5) posterior means, (6) posterior standard deviations and (7) sequence masks
            overshooting_vars.append((F.pad(actions[t:d], seq_pad), F.pad(nonterminals[t:d], seq_pad), F.pad(rewards[t:d], seq_pad[2:]), beliefs[t_], prior_states[t_], F.pad(posterior_means[t_ + 1:d_ + 1].detach(), seq_pad), F.pad(
                posterior_std_devs[t_ + 1:d_ + 1].detach(), seq_pad, value=1), F.pad(torch.ones(d - t, self.cfg.train.batch_size, self.cfg.model.state_size, device=self.cfg.main.device), seq_pad)))  # Posterior standard deviations must be padded with > 0 to prevent infinite KL divergences
        overshooting_vars = tuple(zip(*overshooting_vars))
        # Update belief/state using prior from previous belief/state and previous action (over entire sequence at once)
        beliefs, prior_states, prior_means, prior_std_devs = self.transition_model(torch.cat(overshooting_vars[4], dim=0), torch.cat(
            overshooting_vars[0], dim=1), torch.cat(overshooting_vars[3], dim=0), None, torch.cat(overshooting_vars[1], dim=1))
        seq_mask = torch.cat(overshooting_vars[7], dim=1)
        # Calculate overshooting KL loss with sequence mask
        kl_loss_sum += self.cfg.model.overshooting_kl_beta * torch.max((kl_divergence(Normal(torch.cat(overshooting_vars[5], dim=1), torch.cat(overshooting_vars[6], dim=1)), Normal(
            prior_means, prior_std_devs)) * seq_mask).sum(dim=2), self.free_nats).mean(dim=(0, 1))  # Update KL loss (compensating for extra average over each overshooting/open loop sequence)
        # Calculate overshooting reward prediction loss with sequence mask
        if self.cfg.model.overshooting_reward_scale != 0:
            reward_loss += (1 / self.cfg.model.overshooting_distance) * self.cfg.model.overshooting_reward_scale * F.mse_loss(self.reward_model(beliefs, prior_states)['loc'] * seq_mask[:, :, 0], torch.cat(
                overshooting_vars[2], dim=1), reduction='none').mean(dim=(0, 1)) * (self.cfg.train.chunk_size - 1)  # Update reward loss (compensating for extra average over each overshooting/open loop sequence)
        return kl_loss_sum, reward_loss

    def _calc_observations_loss(self,
                                observation_target,
                                beliefs,
                                posterior_states,
                                ):
        raise NotImplementedError

    def _get_posterior_states(self,
                              states,
                              ):
        posterior_states    = states["posterior_states"]
        posterior_means     = states["posterior_means"]
        posterior_std_devs  = states["posterior_std_devs"]
        return posterior_states, posterior_means, posterior_std_devs

    def _calc_loss(self, 
                  observations_target,
                  actions, 
                  rewards, 
                  nonterminals,
                  states,
                  ):

        beliefs             = states["beliefs"]
        prior_states        = states["prior_states"]
        posterior_states, posterior_means, posterior_std_devs = self._get_posterior_states(states)

        # Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting); sum over final dims, average over batch and time (original implementation, though paper seems to miss 1/T scaling?)
        observations_loss = self._calc_observations_loss(observations_target, beliefs, posterior_states)
        reward_loss = self._calc_reward_loss(rewards, beliefs, posterior_states)
        
        # transition loss
        kl_loss_sum = torch.tensor(0., device=self.cfg.main.device)
        kl_loss = self._calc_kl(states)
        kl_loss_sum += kl_loss

        if self.cfg.model.global_kl_beta != 0:
            kl_loss_sum += self.cfg.model.global_kl_beta * kl_divergence(Normal(posterior_means, posterior_std_devs), 
                                                                         self.global_prior).sum(dim=2).mean(dim=(0, 1))
        # Calculate latent overshooting objective for t > 0
        if self.cfg.model.overshooting_kl_beta != 0:
            _kl_loss, _reward_loss = self._latent_overshooting(actions, rewards, nonterminals, states)
            kl_loss_sum += _kl_loss
            reward_loss += _reward_loss
        # Apply linearly ramping learning rate schedule
        if self.cfg.model.learning_rate_schedule != 0:
            for group in self.model_optimizer.param_groups:
                group['lr'] = min(group['lr'] + self.cfg.model.model_learning_rate /
                                    self.cfg.model.learning_rate_schedule, self.cfg.model.model_learning_rate)
        
        if not self.cfg.model.predict_reward:
            reward_loss = torch.zeros_like(reward_loss)
        
        return observations_loss, reward_loss, kl_loss_sum, kl_loss

    def _get_model_loss(self,
                        observations_target, 
                        actions, 
                        rewards, 
                        nonterminals,
                        states,
                        ):
        

        with torch.cuda.amp.autocast(enabled=self.cfg.train.use_amp):
            observations_loss, reward_loss, kl_loss_sum, kl_loss = self._calc_loss(observations_target, actions, rewards, nonterminals, states)
        
        observations_loss_sum = torch.tensor(0., device=self.cfg.main.device)
        for key in observations_loss.keys():
            observations_loss_sum += observations_loss[key]
        
        model_loss = observations_loss_sum + reward_loss + self.cfg.model.kl_beta*kl_loss_sum
        
        # Log loss info
        loss_info = dict()
        loss_info["observations_loss_sum"] = observations_loss_sum.item()
        for name in observations_loss.keys():
            loss_info["observation_{}_loss".format(name)] = observations_loss[name].item()
        loss_info["reward_loss"] = reward_loss.item()
        loss_info["kl_loss_sum"] = kl_loss_sum.item()
        loss_info["kl_loss"] = kl_loss.item()

        return model_loss, loss_info

    def _sample_data(self,
                     D,
                     ):
        # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
        observations, actions, rewards, nonterminals = D.sample(
            self.cfg.train.batch_size, self.cfg.train.chunk_size)  # Transitions start at time t = 0
        
        observations_target = self._clip_obs(observations, idx_start=1)
        return observations_target, actions, rewards, nonterminals

    def optimize_loss(self,
                      observations_target, 
                      actions, 
                      rewards, 
                      nonterminals,
                      states,
                      itr_optim,
                      ):
        model_loss, loss_info = self._get_model_loss(observations_target, actions, rewards, nonterminals, states)

        # Update model parameters
        self.model_optimizer.zero_grad()

        self.scaler.scale(model_loss).backward()
        nn.utils.clip_grad_norm_(self.param_list, self.cfg.model.grad_clip_norm, norm_type=2)
        self.scaler.step(self.model_optimizer)
        self.scaler.update()

        if self.cfg.main.wandb:
            for name in loss_info.keys():
                wandb.log(data={"{}/train".format(name):loss_info[name]}, step=itr_optim)
            frame = itr_optim * self.cfg.train.batch_size * self.cfg.train.chunk_size
            wandb.log(data={"frame":frame}, step=itr_optim)

    def optimize(self,
                 D, 
                 ):
        self.itr_optim += 1

        with torch.cuda.amp.autocast(enabled=self.cfg.train.use_amp):
            observations_target, actions, rewards, nonterminals = self._sample_data(D)
            states = self.estimate_state(observations_target, actions[:-1], rewards, nonterminals[:-1])
        self.optimize_loss(observations_target, actions, rewards, nonterminals, states, self.itr_optim)

    def validation(self,
                 D, 
                 ):
        self.eval()

        with torch.cuda.amp.autocast(enabled=self.cfg.train.use_amp):
            observations_target, actions, rewards, nonterminals = self._sample_data(D)
            states = self.estimate_state(observations_target, actions[:-1], rewards, nonterminals[:-1])
        _, loss_info = self._get_model_loss(observations_target, actions, rewards, nonterminals, states)
        
        if self.cfg.main.wandb:
            for name in loss_info.keys():        
                wandb.log(data={"{}/validation".format(name):loss_info[name]}, step=self.itr_optim)

        self.train()

class RSSM(RSSM_base):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        print("RSSM")

    def _init_models(self, device):
        self.transition_model = TransitionModel(self.cfg.model.belief_size, self.cfg.model.state_size, self.cfg.env.action_size,
                                        self.cfg.model.hidden_size, dict(self.cfg.model.embedding_size), self.cfg.model.activation_function.dense).to(device=device)

        self.reward_model = RewardModel(h_size=self.cfg.model.belief_size, s_size=self.cfg.model.state_size, hidden_size=self.cfg.model.hidden_size,
                                activation=self.cfg.model.activation_function.dense).to(device=device)
        
        self.observation_model = build_ObservationModel(name=self.cfg.model.observation_names_rec[0],
                                                        observation_shapes=self.cfg.env.observation_shapes,
                                                        embedding_size=dict(self.cfg.model.embedding_size),
                                                        belief_size=self.cfg.model.belief_size,
                                                        state_size=self.cfg.model.state_size,
                                                        hidden_size=self.cfg.model.hidden_size,
                                                        activation_function=dict(self.cfg.model.activation_function),
                                                        normalization=self.cfg.model.normalization,
                                                        ).to(device=device)
        self.encoder = build_Encoder(name=self.cfg.model.observation_names_enc[0],
                                    observation_shapes=self.cfg.env.observation_shapes,
                                    embedding_size=dict(self.cfg.model.embedding_size), 
                                    activation_function=dict(self.cfg.model.activation_function),
                                    normalization=self.cfg.model.normalization,
                                    ).to(device=device)
        if self.cfg.main.wandb:
            wandb.watch(self.transition_model)
            wandb.watch(self.observation_model)
            wandb.watch(self.reward_model)
            wandb.watch(self.encoder)

    def _init_param_list(self):
        self.param_list = list(self.parameters())

    def get_state_dict(self):
        return self.state_dict()

    def estimate_state(self,
                        observations,
                        actions, 
                        rewards, 
                        nonterminals,
                        batch_size=None,
                        det=False):
        if batch_size == None:
            batch_size = actions.shape[1]

        # Create initial belief and state for time t = 0
        init_belief, init_state = torch.zeros(batch_size, self.cfg.model.belief_size, device=self.cfg.main.device), torch.zeros(batch_size, self.cfg.model.state_size, device=self.cfg.main.device)
        # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
        
        obs_emb = bottle_tupele(self.encoder, observations)

        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs, expert_means, expert_std_devs = self.transition_model(
            init_state, actions, init_belief, obs_emb, nonterminals, det=det)

        states = dict(beliefs=beliefs,
                     prior_states=prior_states,
                     prior_means=prior_means,
                     prior_std_devs=prior_std_devs,
                     posterior_states=posterior_states,
                     posterior_means=posterior_means,
                     posterior_std_devs=posterior_std_devs,
                     expert_means=expert_means, 
                     expert_std_devs=expert_std_devs,
                     )
        return states
    
    def _calc_observations_loss(self,
                               observations_target,
                               beliefs,
                               posterior_states,
                               ):
        observations_loss = dict()
        
        if self.cfg.model.worldmodel_LogProbLoss:
            log_probs = self.observation_model.get_log_prob(beliefs, posterior_states, observations_target[self.observation_name])
            observation_loss = (-log_probs).mean(dim=(0,1)).sum()
        else:
            mse = self.observation_model.get_mse(beliefs, posterior_states, observations_target[self.observation_name])
            observation_loss = (mse).mean(dim=(0,1)).sum()
        observations_loss[self.observation_name] = observation_loss
        return observations_loss


class MRSSM_base(RSSM_base):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)

    def eval(self):
        self.transition_model._eval()
        self.observation_model.eval()
        self.reward_model.eval()
        self.encoder.eval()
    
    def train(self):
        self.transition_model._train()
        self.observation_model.train()
        self.reward_model.train()
        self.encoder.train()

    def load_state_dict(self, model_dicts):
        self.observation_model._load_state_dict(model_dicts['observation_model'])        
        self.encoder._load_state_dict(model_dicts['encoder'])
        self.transition_model._load_state_dict(model_dicts['transition_model'])
        self.reward_model.load_state_dict(model_dicts['reward_model'])
        self.model_optimizer.load_state_dict(model_dicts['model_optimizer'])

    def _init_param_list(self):
        transition_model_params = self.transition_model.get_model_params()
        observation_model_params = self.observation_model.get_model_params()
        encoder_params = self.encoder.get_model_params()
        
        self.param_list = transition_model_params \
                + observation_model_params \
                + list(self.reward_model.parameters()) \
                + encoder_params 

    def get_state_dict(self):
        state_dict = {'transition_model': self.transition_model.get_state_dict(),
                    'observation_model': self.observation_model.get_state_dict(),
                    'reward_model': self.reward_model.state_dict(),
                    'encoder': self.encoder.get_state_dict(),
                    'model_optimizer': self.model_optimizer.state_dict(),
                    }
        return state_dict

    def estimate_state(self,
                        observations,
                        actions, 
                        rewards, 
                        nonterminals,
                        batch_size=None,
                        det=False):
        if batch_size == None:
            batch_size = actions.shape[1]

        # Create initial belief and state for time t = 0
        init_belief, init_state = torch.zeros(batch_size, self.cfg.model.belief_size, device=self.cfg.main.device), torch.zeros(batch_size, self.cfg.model.state_size, device=self.cfg.main.device)
        # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
        
        obs_emb = bottle_tupele_multimodal(self.encoder, observations)

        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs, expert_means, expert_std_devs = self.transition_model(
            init_state, actions, init_belief, obs_emb, nonterminals, det=det)

        states = dict(beliefs=beliefs,
                     prior_states=prior_states,
                     prior_means=prior_means,
                     prior_std_devs=prior_std_devs,
                     posterior_states=posterior_states,
                     posterior_means=posterior_means,
                     posterior_std_devs=posterior_std_devs,
                     expert_means=expert_means, 
                     expert_std_devs=expert_std_devs,
                     )
        return states

    def _calc_observations_loss(self,
                               observations_target,
                               beliefs,
                               posterior_states,
                               ):
        observations_loss = dict()
        
        if self.cfg.model.worldmodel_LogProbLoss:
            log_probs = self.observation_model.get_log_prob(beliefs, posterior_states, observations_target)
            for name in log_probs.keys():
                observations_loss[name] = -log_probs[name].mean(dim=(0,1)).sum()
        else:
            
            mse = self.observation_model.get_mse(h_t=beliefs, s_t=posterior_states, o_t=observations_target)
            for name in mse.keys():
                observations_loss[name] = mse[name].mean(dim=(0,1)).sum()
        
        return observations_loss

class MRSSM_NN(MRSSM_base):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        print("Multimodal RSSM (NN)")

    def _init_models(self, device):
        self.transition_model = MultimodalTransitionModel(belief_size=self.cfg.model.belief_size, 
                                                    state_size=self.cfg.model.state_size,
                                                    action_size=self.cfg.env.action_size, 
                                                    hidden_size=self.cfg.model.hidden_size, 
                                                    observation_names_enc=self.cfg.model.observation_names_enc,
                                                    embedding_size=dict(self.cfg.model.embedding_size), 
                                                    device=device,
                                                    fusion_method=self.cfg.model.multimodal_params.fusion_method,
                                                    expert_dist=self.cfg.model.multimodal_params.expert_dist,
                                                    use_prior_expert=self.cfg.model.multimodal_params.use_prior_expert,
                                                    ).to(device=device)

        self.reward_model = RewardModel(h_size=self.cfg.model.belief_size, s_size=self.cfg.model.state_size, hidden_size=self.cfg.model.hidden_size,
                                activation=self.cfg.model.activation_function.dense).to(device=device)
        
        self.observation_model = MultimodalObservationModel(observation_names_rec=self.cfg.model.observation_names_rec,
                                                            observation_shapes=self.cfg.env.observation_shapes,
                                                            embedding_size=dict(self.cfg.model.embedding_size),
                                                            belief_size=self.cfg.model.belief_size,
                                                            state_size=self.cfg.model.state_size,
                                                            hidden_size=self.cfg.model.hidden_size,
                                                            activation_function=dict(self.cfg.model.activation_function),
                                                            normalization=self.cfg.model.normalization,
                                                            device=device)
        self.encoder = MultimodalEncoder(observation_names_enc=self.cfg.model.observation_names_enc,
                                            observation_shapes=self.cfg.env.observation_shapes,
                                            embedding_size=dict(self.cfg.model.embedding_size), 
                                            activation_function=dict(self.cfg.model.activation_function),
                                            normalization=self.cfg.model.normalization,
                                            device=device
                                            )

class MRSSM_PoE(MRSSM_base):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        print("Multimodal RSSM (PoE)")

    def _init_models(self, device):
        self.transition_model = MultimodalTransitionModel(belief_size=self.cfg.model.belief_size, 
                                                    state_size=self.cfg.model.state_size,
                                                    action_size=self.cfg.env.action_size, 
                                                    hidden_size=self.cfg.model.hidden_size, 
                                                    observation_names_enc=self.cfg.model.observation_names_enc,
                                                    embedding_size=dict(self.cfg.model.embedding_size), 
                                                    device=device,
                                                    fusion_method=self.cfg.model.multimodal_params.fusion_method,
                                                    expert_dist=self.cfg.model.multimodal_params.expert_dist,
                                                    use_prior_expert=self.cfg.model.multimodal_params.use_prior_expert,
                                                    ).to(device=device)

        self.reward_model = RewardModel(h_size=self.cfg.model.belief_size, s_size=self.cfg.model.state_size, hidden_size=self.cfg.model.hidden_size,
                                activation=self.cfg.model.activation_function.dense).to(device=device)
        
        self.observation_model = MultimodalObservationModel(observation_names_rec=self.cfg.model.observation_names_rec,
                                                            observation_shapes=self.cfg.env.observation_shapes,
                                                            embedding_size=dict(self.cfg.model.embedding_size),
                                                            belief_size=self.cfg.model.belief_size,
                                                            state_size=self.cfg.model.state_size,
                                                            hidden_size=self.cfg.model.hidden_size,
                                                            activation_function=dict(self.cfg.model.activation_function),
                                                            normalization=self.cfg.model.normalization,
                                                            device=device)
        
        if self.cfg.model.multimodal_params.expert_dist == "q(st|ht,ot)":
            self.encoder = MultimodalEncoder(observation_names_enc=self.cfg.model.observation_names_enc,
                                                observation_shapes=self.cfg.env.observation_shapes,
                                                embedding_size=dict(self.cfg.model.embedding_size), 
                                                activation_function=dict(self.cfg.model.activation_function),
                                                normalization=self.cfg.model.normalization,
                                                device=device
                                                )
        elif self.cfg.model.multimodal_params.expert_dist == "q(st|ot)":
            self.encoder = MultimodalStochasticEncoder(observation_names_enc=self.cfg.model.observation_names_enc,
                                                        observation_shapes=self.cfg.env.observation_shapes,
                                                        embedding_size=dict(self.cfg.model.embedding_size), 
                                                        state_size=self.cfg.model.state_size,
                                                        hidden_size=self.cfg.model.hidden_size,
                                                        activation_function=dict(self.cfg.model.activation_function),
                                                        normalization=self.cfg.model.normalization,
                                                        device=device
                                                        )

    def _get_posterior_states(self,
                              states,
                              ):
        expert_means = states["expert_means"]
        expert_std_devs = states["expert_std_devs"]

        experts_loc = []
        experts_scale = []
        for name in expert_means.keys():
            experts_loc.append(expert_means[name])
            experts_scale.append(expert_std_devs[name])

        experts_loc = torch.stack(experts_loc)
        experts_scale = torch.stack(experts_scale)

        posterior_means, posterior_std_devs = poe(experts_loc, experts_scale)
        posterior_states = Normal(posterior_means, posterior_std_devs).rsample()
        return posterior_states, posterior_means, posterior_std_devs


class MRSSM_MoPoE(MRSSM_base):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        print("Multimodal RSSM (MoPoE)")

    def _init_models(self, device):
        self.transition_model = MultimodalTransitionModel(belief_size=self.cfg.model.belief_size, 
                                                    state_size=self.cfg.model.state_size,
                                                    action_size=self.cfg.env.action_size, 
                                                    hidden_size=self.cfg.model.hidden_size, 
                                                    observation_names_enc=self.cfg.model.observation_names_enc,
                                                    embedding_size=dict(self.cfg.model.embedding_size), 
                                                    device=device,
                                                    fusion_method=self.cfg.model.multimodal_params.fusion_method,
                                                    expert_dist=self.cfg.model.multimodal_params.expert_dist,
                                                    use_prior_expert=self.cfg.model.multimodal_params.use_prior_expert,
                                                    ).to(device=device)

        self.reward_model = RewardModel(h_size=self.cfg.model.belief_size, s_size=self.cfg.model.state_size, hidden_size=self.cfg.model.hidden_size,
                                activation=self.cfg.model.activation_function.dense).to(device=device)
        
        self.observation_model = MultimodalObservationModel(observation_names_rec=self.cfg.model.observation_names_rec,
                                                            observation_shapes=self.cfg.env.observation_shapes,
                                                            embedding_size=dict(self.cfg.model.embedding_size),
                                                            belief_size=self.cfg.model.belief_size,
                                                            state_size=self.cfg.model.state_size,
                                                            hidden_size=self.cfg.model.hidden_size,
                                                            activation_function=dict(self.cfg.model.activation_function),
                                                            normalization=self.cfg.model.normalization,
                                                            device=device)
        
        if self.cfg.model.multimodal_params.expert_dist == "q(st|ht,ot)":
            self.encoder = MultimodalEncoder(observation_names_enc=self.cfg.model.observation_names_enc,
                                                        observation_shapes=self.cfg.env.observation_shapes,
                                                        embedding_size=dict(self.cfg.model.embedding_size), 
                                                        activation_function=dict(self.cfg.model.activation_function),
                                                        normalization=self.cfg.model.normalization,
                                                        device=device
                                                        )
        elif self.cfg.model.multimodal_params.expert_dist == "q(st|ot)":
            self.encoder = MultimodalStochasticEncoder(observation_names_enc=self.cfg.model.observation_names_enc,
                                                        observation_shapes=self.cfg.env.observation_shapes,
                                                        embedding_size=dict(self.cfg.model.embedding_size), 
                                                        state_size=self.cfg.model.state_size,
                                                        hidden_size=self.cfg.model.hidden_size,
                                                        activation_function=dict(self.cfg.model.activation_function),
                                                        normalization=self.cfg.model.normalization,
                                                        device=device
                                                        )

    def _get_posterior_states(self,
                              states,
                              ):
        expert_means = states["expert_means"]
        expert_std_devs = states["expert_std_devs"]
        return mopoe(expert_means, expert_std_devs)
        
    def _latent_overshooting(self,
                             actions,
                             rewards,
                             nonterminals,
                             states,
                             ):
        beliefs             = states["beliefs"]
        prior_states        = states["prior_states"]
        
        subset_means, subset_std_devs = calc_subset_states(states["expert_means"], states["expert_std_devs"])
        n_subset = len(subset_means)
        kl_loss_sum = torch.tensor(0., device=self.cfg.main.device)
        reward_loss = torch.tensor(0., device=self.cfg.main.device)

        for i in range(n_subset):
            overshooting_vars = []  # Collect variables for overshooting to process in batch
            for t in range(1, self.cfg.train.chunk_size - 1):
                d = min(t + self.cfg.model.overshooting_distance,
                        self.cfg.train.chunk_size - 1)  # Overshooting distance
                # Use t_ and d_ to deal with different time indexing for latent states
                t_, d_ = t - 1, d - 1
                # Calculate sequence padding so overshooting terms can be calculated in one batch
                seq_pad = (0, 0, 0, 0, 0, t - d + self.cfg.model.overshooting_distance)
                # Store (0) actions, (1) nonterminals, (2) rewards, (3) beliefs, (4) prior states, (5) posterior means, (6) posterior standard deviations and (7) sequence masks
                overshooting_vars.append((F.pad(actions[t:d], seq_pad), F.pad(nonterminals[t:d], seq_pad), F.pad(rewards[t:d], seq_pad[2:]), beliefs[t_], prior_states[t_], F.pad(subset_means[i][t_ + 1:d_ + 1].detach(), seq_pad), 
                F.pad(subset_std_devs[i][t_ + 1:d_ + 1].detach(), seq_pad, value=1), F.pad(torch.ones(d - t, self.cfg.train.batch_size, self.cfg.model.state_size, device=self.cfg.main.device), seq_pad)))  # Posterior standard deviations must be padded with > 0 to prevent infinite KL divergences
            overshooting_vars = tuple(zip(*overshooting_vars))
            # Update belief/state using prior from previous belief/state and previous action (over entire sequence at once)
            _beliefs, _prior_states, _prior_means, _prior_std_devs = self.transition_model(torch.cat(overshooting_vars[4], dim=0), torch.cat(
                overshooting_vars[0], dim=1), torch.cat(overshooting_vars[3], dim=0), None, torch.cat(overshooting_vars[1], dim=1))
            seq_mask = torch.cat(overshooting_vars[7], dim=1)
            # Calculate overshooting KL loss with sequence mask
            kl_loss_sum += self.cfg.model.overshooting_kl_beta * torch.max((kl_divergence(Normal(torch.cat(overshooting_vars[5], dim=1), torch.cat(overshooting_vars[6], dim=1)), Normal(
                _prior_means, _prior_std_devs)) * seq_mask).sum(dim=2), self.free_nats).mean(dim=(0, 1))  # Update KL loss (compensating for extra average over each overshooting/open loop sequence)
        kl_loss_sum = kl_loss_sum / n_subset
        # Calculate overshooting reward prediction loss with sequence mask
        if self.cfg.model.overshooting_reward_scale != 0:
            reward_loss += (1 / self.cfg.model.overshooting_distance) * self.cfg.model.overshooting_reward_scale * F.mse_loss(self.reward_model(_beliefs, _prior_states)['loc'] * seq_mask[:, :, 0], torch.cat(
                overshooting_vars[2], dim=1), reduction='none').mean(dim=(0, 1)) * (self.cfg.train.chunk_size - 1)  # Update reward loss (compensating for extra average over each overshooting/open loop sequence)
        return kl_loss_sum, reward_loss

    def _calc_mopoe_kl(self,
                expert_means,
                expert_std_devs,
                prior_means,
                prior_std_devs,
                ):
        
        subset_means, subset_std_devs = calc_subset_states(expert_means, expert_std_devs)
        kl_losses = []
        for i in range(len(subset_means)):
            div = kl_divergence(Normal(subset_means[i], subset_std_devs[i]), 
                                Normal(prior_means, prior_std_devs)).sum(dim=2)
            kl_loss = torch.max(div, self.free_nats).mean(dim=(0, 1))
            kl_losses.append(kl_loss)
        
        return torch.stack(kl_losses).mean(dim=0)

    def _calc_kl(self,
                states,
                ):
        prior_means         = states["prior_means"]
        prior_std_devs      = states["prior_std_devs"]
        expert_means = states["expert_means"]
        expert_std_devs = states["expert_std_devs"]

        kl_loss = self._calc_mopoe_kl(expert_means, expert_std_devs, prior_means, prior_std_devs)

        return kl_loss
