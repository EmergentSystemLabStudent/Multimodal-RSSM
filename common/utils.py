from common.models.policy import ActorModel
from common.models.transition_model import TransitionModel
import torch
from typing import Iterable
from torch.nn import Module


def imagine_ahead(prev_state, prev_belief, policy: ActorModel, transition_model: TransitionModel, planning_horizon=12, prop_transition_model=False):
  '''
  imagine_ahead is the function to draw the imaginary tracjectory using the dynamics model, actor, critic.
  Input: current state (posterior), current belief (hidden), policy, transition_model  # torch.Size([50, 30]) torch.Size([50, 200])
  Output: generated trajectory of features includes beliefs, prior_states, prior_means, prior_std_devs
          torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
  '''
  flatten = lambda x: x.reshape([-1]+list(x.size()[2:]))
  prev_belief = flatten(prev_belief)
  prev_state = flatten(prev_state)

  # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
  T = planning_horizon
  beliefs, prior_states, prior_means, prior_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
  beliefs[0], prior_states[0] = prev_belief, prev_state

  # Loop over time sequence
  for t in range(T - 1):
    _state = prior_states[t]
    if prop_transition_model:
      actions = policy.get_action(belief=beliefs[t], state=_state)
    else:
      actions = policy.get_action(belief=beliefs[t].detach(), state=_state.detach())
    # Compute belief (deterministic hidden state)
    hidden = transition_model.act_fn(transition_model.fc_embed_state_action(torch.cat([_state, actions], dim=1)))
    beliefs[t + 1] = transition_model.rnn(hidden, beliefs[t])
    # Compute state prior by applying transition dynamics
    """
    hidden = transition_model.act_fn(transition_model.fc_embed_belief_prior(beliefs[t + 1]))
    prior_means[t + 1], _prior_std_dev = torch.chunk(transition_model.fc_state_prior(hidden), 2, dim=1)
    """
    prior_states[t + 1] = transition_model.stochastic_state_model.sample(beliefs[t + 1])
    loc_and_scale = transition_model.stochastic_state_model(h_t=beliefs[t + 1])
    prior_std_devs[t + 1] = loc_and_scale['scale']
    prior_means[t + 1] = loc_and_scale['loc']

  # Return new hidden states
  imagined_traj = [torch.stack(beliefs[1:], dim=0), torch.stack(prior_states[1:], dim=0), torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
  return imagined_traj

def imagine_ahead2(prev_state, prev_belief, policy: ActorModel, transition_model: TransitionModel, planning_horizon=12):
  '''
  imagine_ahead is the function to draw the imaginary tracjectory using the dynamics model, actor, critic.
  Input: current state (posterior), current belief (hidden), policy, transition_model  # torch.Size([50, 30]) torch.Size([50, 200])
  Output: generated trajectory of features includes beliefs, prior_states, prior_means, prior_std_devs
          torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
  '''
  flatten = lambda x: x.reshape([-1]+list(x.size()[2:]))
  prev_belief = flatten(prev_belief)
  prev_state = flatten(prev_state)

  # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
  T = planning_horizon
  beliefs, prior_states, prior_means, prior_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
  beliefs[0], prior_states[0] = prev_belief, prev_state

  actions = [torch.empty(0)] * T

  # Loop over time sequence
  for t in range(T - 1):
    _state = prior_states[t]
    action = policy.get_action(belief=beliefs[t].detach(), state=_state.detach())
    actions[t] = action
    # Compute belief (deterministic hidden state)
    hidden = transition_model.act_fn(transition_model.fc_embed_state_action(torch.cat([_state, actions[t]], dim=1)))
    beliefs[t + 1] = transition_model.rnn(hidden, beliefs[t])
    # Compute state prior by applying transition dynamics
    """
    hidden = transition_model.act_fn(transition_model.fc_embed_belief_prior(beliefs[t + 1]))
    prior_means[t + 1], _prior_std_dev = torch.chunk(transition_model.fc_state_prior(hidden), 2, dim=1)
    """
    prior_states[t + 1] = transition_model.stochastic_state_model.sample(beliefs[t + 1])
    loc_and_scale = transition_model.stochastic_state_model(h_t=beliefs[t + 1])
    prior_std_devs[t + 1] = loc_and_scale['scale']
    prior_means[t + 1] = loc_and_scale['loc']
    

  # Return new hidden states
  imagined_traj = [torch.stack(beliefs[1:], dim=0), torch.stack(prior_states[1:], dim=0), torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0), torch.stack(actions[:-1], dim=0)]
  return imagined_traj

def lambda_return(imged_reward, value_pred, bootstrap, discount=0.99, lambda_=0.95):
  # Setting lambda=1 gives a discounted Monte Carlo return.
  # Setting lambda=0 gives a fixed 1-step return.
  next_values = torch.cat([value_pred[1:], bootstrap[None]], 0)
  discount_tensor = discount * torch.ones_like(imged_reward) #pcont
  inputs = imged_reward + discount_tensor * next_values * (1 - lambda_)
  last = bootstrap
  indices = reversed(range(len(inputs)))
  outputs = []
  for index in indices:
    inp, disc = inputs[index], discount_tensor[index]
    last = inp + disc*lambda_*last
    outputs.append(last)
  outputs = list(reversed(outputs))
  outputs = torch.stack(outputs, 0)
  returns = outputs
  return returns

class ActivateParameters:
  def __init__(self, modules: Iterable[Module]):
      """
      Context manager to locally Activate the gradients.
      example:
      ```
      with ActivateParameters([module]):
          output_tensor = module(input_tensor)
      ```
      :param modules: iterable of modules. used to call .parameters() to freeze gradients.
      """
      self.modules = modules
      self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

  def __enter__(self):
      for param in get_parameters(self.modules):
          param.requires_grad = True

  def __exit__(self, exc_type, exc_val, exc_tb):
      for i, param in enumerate(get_parameters(self.modules)):
          param.requires_grad = self.param_states[i]

# "get_parameters" and "FreezeParameters" are from the following repo
# https://github.com/juliusfrost/dreamer-pytorch
def get_parameters(modules: Iterable[Module]):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters

class FreezeParameters:
  def __init__(self, modules: Iterable[Module]):
      """
      Context manager to locally freeze gradients.
      In some cases with can speed up computation because gradients aren't calculated for these listed modules.
      example:
      ```
      with FreezeParameters([module]):
          output_tensor = module(input_tensor)
      ```
      :param modules: iterable of modules. used to call .parameters() to freeze gradients.
      """
      self.modules = modules
      self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

  def __enter__(self):
      for param in get_parameters(self.modules):
          param.requires_grad = False

  def __exit__(self, exc_type, exc_val, exc_tb):
      for i, param in enumerate(get_parameters(self.modules)):
          param.requires_grad = self.param_states[i]

