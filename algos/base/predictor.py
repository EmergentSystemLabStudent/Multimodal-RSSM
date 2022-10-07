import torch
from torch.distributions import Normal

class PredictiveCoder:
    def __init__(self, rssm, action_size, predictive_target="action") -> None:
        self.rssm = rssm
        self.predictive_target = predictive_target
        self.action_size = action_size

    def get_action(self, belief: torch.Tensor, state: torch.Tensor, det=True):
        b, d = state.shape
        action_dummy = torch.zeros((b, self.action_size), dtype=torch.float32, device=self.rssm.device)
        # Compute belief (deterministic hidden state)
        hidden = self.rssm.transition_model.act_fn(self.rssm.transition_model.fc_embed_state_action(
            torch.cat([state, action_dummy], dim=1)))
        # h_t = f(h_{t-1}, s_{t-1}, a_{t-1})
        _belief = self.rssm.transition_model.rnn(hidden, belief)
        if det:
            _state = self.rssm.transition_model.stochastic_state_model(_belief)["loc"]
        else:
            _state = self.rssm.transition_model.stochastic_state_model.sample(_belief)

        action_pred = self.rssm.observation_model.observation_models[self.predictive_target](_belief.unsqueeze(0), _state.unsqueeze(0))
        if det:
            action = action_pred["loc"].squeeze(0)
        else:
            action = Normal(action_pred["loc"], action_pred["scale"]).sample().squeeze(0)
        return action