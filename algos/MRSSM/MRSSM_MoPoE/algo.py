import torch
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal
from torch.nn import functional as F

from utils.models.encoder import MultimodalEncoder, MultimodalStochasticEncoder, calc_subset_states, get_mopoe_state
from utils.models.observation_model import MultimodalObservationModel
from utils.models.reward_model import RewardModel
from utils.models.transition_model import MultimodalTransitionModel

from algos.MRSSM.base.algo import MRSSM_base

class MRSSM_MoPoE(MRSSM_base):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        print("Multimodal RSSM (MoPoE)")

    def _init_models(self, device):
        self.transition_model = MultimodalTransitionModel(belief_size=self.cfg.rssm.belief_size, 
                                                    state_size=self.cfg.rssm.state_size,
                                                    action_size=self.cfg.env.action_size, 
                                                    hidden_size=self.cfg.rssm.hidden_size, 
                                                    observation_names_enc=self.cfg.rssm.observation_names_enc,
                                                    embedding_size=dict(self.cfg.rssm.embedding_size), 
                                                    device=device,
                                                    fusion_method=self.cfg.rssm.multimodal_params.fusion_method,
                                                    expert_dist=self.cfg.rssm.multimodal_params.expert_dist,
                                                    ).to(device=device)

        self.reward_model = RewardModel(h_size=self.cfg.rssm.belief_size, s_size=self.cfg.rssm.state_size, hidden_size=self.cfg.rssm.hidden_size,
                                activation=self.cfg.rssm.activation_function.dense).to(device=device)
        
        self.observation_model = MultimodalObservationModel(observation_names_rec=self.cfg.rssm.observation_names_rec,
                                                            observation_shapes=self.cfg.env.observation_shapes,
                                                            embedding_size=dict(self.cfg.rssm.embedding_size),
                                                            belief_size=self.cfg.rssm.belief_size,
                                                            state_size=self.cfg.rssm.state_size,
                                                            hidden_size=self.cfg.rssm.hidden_size,
                                                            activation_function=dict(self.cfg.rssm.activation_function),
                                                            normalization=self.cfg.rssm.normalization,
                                                            device=device)
        
        if self.cfg.rssm.multimodal_params.expert_dist == "q(st|ht,ot)":
            self.encoder = MultimodalEncoder(observation_names_enc=self.cfg.rssm.observation_names_enc,
                                                        observation_shapes=self.cfg.env.observation_shapes,
                                                        embedding_size=dict(self.cfg.rssm.embedding_size), 
                                                        activation_function=dict(self.cfg.rssm.activation_function),
                                                        normalization=self.cfg.rssm.normalization,
                                                        device=device
                                                        )
        elif self.cfg.rssm.multimodal_params.expert_dist == "q(st|ot)":
            self.encoder = MultimodalStochasticEncoder(observation_names_enc=self.cfg.rssm.observation_names_enc,
                                                        observation_shapes=self.cfg.env.observation_shapes,
                                                        embedding_size=dict(self.cfg.rssm.embedding_size), 
                                                        state_size=self.cfg.rssm.state_size,
                                                        hidden_size=self.cfg.rssm.hidden_size,
                                                        activation_function=dict(self.cfg.rssm.activation_function),
                                                        normalization=self.cfg.rssm.normalization,
                                                        device=device
                                                        )

    def _get_posterior_states(self,
                              states,
                              ):
        expert_means = states["expert_means"]
        expert_std_devs = states["expert_std_devs"]
        return get_mopoe_state(expert_means, expert_std_devs)
        
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
                d = min(t + self.cfg.rssm.overshooting_distance,
                        self.cfg.train.chunk_size - 1)  # Overshooting distance
                # Use t_ and d_ to deal with different time indexing for latent states
                t_, d_ = t - 1, d - 1
                # Calculate sequence padding so overshooting terms can be calculated in one batch
                seq_pad = (0, 0, 0, 0, 0, t - d + self.cfg.rssm.overshooting_distance)
                # Store (0) actions, (1) nonterminals, (2) rewards, (3) beliefs, (4) prior states, (5) posterior means, (6) posterior standard deviations and (7) sequence masks
                overshooting_vars.append((F.pad(actions[t:d], seq_pad), F.pad(nonterminals[t:d], seq_pad), F.pad(rewards[t:d], seq_pad[2:]), beliefs[t_], prior_states[t_], F.pad(subset_means[i][t_ + 1:d_ + 1].detach(), seq_pad), 
                F.pad(subset_std_devs[i][t_ + 1:d_ + 1].detach(), seq_pad, value=1), F.pad(torch.ones(d - t, self.cfg.train.batch_size, self.cfg.rssm.state_size, device=self.cfg.main.device), seq_pad)))  # Posterior standard deviations must be padded with > 0 to prevent infinite KL divergences
            overshooting_vars = tuple(zip(*overshooting_vars))
            # Update belief/state using prior from previous belief/state and previous action (over entire sequence at once)
            _beliefs, _prior_states, _prior_means, _prior_std_devs = self.transition_model(torch.cat(overshooting_vars[4], dim=0), torch.cat(
                overshooting_vars[0], dim=1), torch.cat(overshooting_vars[3], dim=0), None, torch.cat(overshooting_vars[1], dim=1))
            seq_mask = torch.cat(overshooting_vars[7], dim=1)
            # Calculate overshooting KL loss with sequence mask
            kl_loss_sum += self.cfg.rssm.overshooting_kl_beta * torch.max((kl_divergence(Normal(torch.cat(overshooting_vars[5], dim=1), torch.cat(overshooting_vars[6], dim=1)), Normal(
                _prior_means, _prior_std_devs)) * seq_mask).sum(dim=2), self.free_nats).mean(dim=(0, 1))  # Update KL loss (compensating for extra average over each overshooting/open loop sequence)
        kl_loss_sum = kl_loss_sum / n_subset
        # Calculate overshooting reward prediction loss with sequence mask
        if self.cfg.rssm.overshooting_reward_scale != 0:
            reward_loss += (1 / self.cfg.rssm.overshooting_distance) * self.cfg.rssm.overshooting_reward_scale * F.mse_loss(self.reward_model(_beliefs, _prior_states)['loc'] * seq_mask[:, :, 0], torch.cat(
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
