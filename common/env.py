import cv2
import numpy as np
import torch
import copy

GYM_ENVS = ['Pendulum-v0', 'MountainCarContinuous-v0', 'Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2',
            'HumanoidStandup-v2', 'InvertedDoublePendulum-v2', 'InvertedPendulum-v2', 'Reacher-v2', 'Swimmer-v2', 'Walker2d-v2']
CONTROL_SUITE_ENVS = ['cartpole-balance', 'cartpole-swingup', 'reacher-easy', 'finger-spin', 'cheetah-run', 'ball_in_cup-catch',
                      'walker-walk', 'reacher-hard', 'walker-run', 'humanoid-stand', 'humanoid-walk', 'fish-swim', 'acrobot-swingup']
ROBOT_SUITE_ENVS = ['Door', 'Lift', 'TwoArmPegInHole']
CONTROL_SUITE_ACTION_REPEATS = {'cartpole': 8, 'reacher': 4, 'finger': 2,
                                'cheetah': 4, 'ball_in_cup': 6, 'walker': 2, 'humanoid': 2, 'fish': 2, 'acrobot': 4}


# Preprocesses an observation inplace (from float32 Tensor [0, 255] to [-0.5, 0.5])
@torch.jit.script
def preprocess_observation_(observation, bit_depth):
  # Quantise to given bit depth and centre
  observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(0.5)
  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)
  observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))
  return observation


# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def postprocess_observation(observation, bit_depth=5):
  return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(np.uint8)


def _images_to_observation(images, bit_depth, noise_scale=0, add_noise=False, noisy_background=False, size=(64,64)):
  images = torch.tensor(cv2.resize(images, size, interpolation=cv2.INTER_LINEAR).transpose(
      2, 0, 1), dtype=torch.float32)  # Resize and put channel first
      
  # Quantise, centre and dequantise inplace
  images = preprocess_observation_(images, bit_depth)
  images = images.numpy()
  if noisy_background == True:
    back_ground = np.random.normal(0, scale=noise_scale, size=(3,*size))
    bools = (images[0] < 0.)
    images[:, bools] = back_ground[:, bools]
    
  if add_noise:
    images = np.random.normal(images, scale=noise_scale)
  
  return images

def MultimodalControlSuiteEnv(env_config, symbolic, seed, episode_horizon, action_repeat, bit_depth, info_names=[], multimodal=True, noise_scale=0, add_noise=False, noisy_background=False):
  if env_config.env_name in CONTROL_SUITE_ENVS:
    return MultimodalDMControlSuiteEnv(env_config, symbolic, seed, episode_horizon, action_repeat, bit_depth, info_names, multimodal, noise_scale, add_noise, noisy_background)
  elif env_config.env_name in ROBOT_SUITE_ENVS:
    return MultimodalRoboSuiteEnv(env_config, symbolic, seed, episode_horizon, action_repeat, bit_depth, info_names, multimodal, noise_scale, add_noise, noisy_background)
  else:
    raise NotImplementedError("{} is not implemented".format(env_config.env_name))
  
class MultimodalDMControlSuiteEnv():
  def __init__(self, env_config, symbolic, seed, episode_horizon, action_repeat, bit_depth, info_names=[], multimodal=True, noise_scale=0, add_noise=False, noisy_background=False):
    from dm_control import suite
    from dm_control.suite.wrappers import pixels
    domain, task = env_config.env_name.split('-')
    self.symbolic = symbolic
    self._env = suite.load(
        domain_name=domain, task_name=task, task_kwargs={'random': seed})
    if not symbolic:
      self._env = pixels.Wrapper(self._env)
    self.episode_horizon = episode_horizon
    self.action_repeat = action_repeat
    if action_repeat != CONTROL_SUITE_ACTION_REPEATS[domain]:
      print('Using action repeat %d; recommended action repeat for domain is %d' % (
          action_repeat, CONTROL_SUITE_ACTION_REPEATS[domain]))
    self.bit_depth = bit_depth
    self.info_names = info_names
    self.multimodal = multimodal

    self.noise_scale=noise_scale
    self.add_noise=add_noise
    self.noisy_background=noisy_background

  def _reset(self):
    self.t = 0  # Reset internal timer
    state = self._env.reset()
    if self.symbolic:
      return torch.tensor(np.concatenate([np.asarray([obs]) if isinstance(obs, float) else obs for obs in state.observation.values()], axis=0), dtype=torch.float32)#.unsqueeze(dim=0)
    else:
      return _images_to_observation(self._env.physics.render(camera_id=0), self.bit_depth,
                                    self.noise_scale, self.add_noise, self.noisy_background)

  def reset(self):
    if not self.multimodal:
      return self._reset()

    image = self._reset()
    kinematic_info = self.get_kinematic_info(reset=True)
    
    observation = dict(image=image)
    for k in kinematic_info.keys():
      observation[k] = kinematic_info[k]
    return observation


  def _step(self, action):
    action = action.detach().numpy()
    reward = 0
    for k in range(self.action_repeat):
      state = self._env.step(action)
      reward += state.reward
      self.t += 1  # Increment internal timer
      done = state.last() or self.t == self.episode_horizon
      if done:
        break
    if self.symbolic:
      observation = torch.tensor(np.concatenate([np.asarray([obs]) if isinstance(
          obs, float) else obs for obs in state.observation.values()], axis=0), dtype=torch.float32)
    else:
      observation = _images_to_observation(
          self._env.physics.render(camera_id=0), self.bit_depth,
          self.noise_scale, self.add_noise, self.noisy_background)
    return observation, reward, done

  def step(self, action):
    if not self.multimodal:
      return self._step(action)
    image, reward, done = self._step(action)
    kinematic_info = self.get_kinematic_info()

    observation = dict(image=image)
    for k in kinematic_info.keys():
      observation[k] = kinematic_info[k]

    return observation, reward, done

  def render(self):
    cv2.imshow('screen', self._env.physics.render(camera_id=0)[:, :, ::-1])
    cv2.waitKey(1)

  def close(self):
    cv2.destroyAllWindows()
    self._env.close()

  @property
  def observation_size(self):
    return sum([(1 if len(obs.shape) == 0 else obs.shape[0]) for obs in self._env.observation_spec().values()]) if self.symbolic else (3, 64, 64)

  @property
  def action_size(self):
    return self._env.action_spec().shape[0]

  # Sample an action randomly from a uniform distribution over all valid actions
  def sample_random_action(self):
    spec = self._env.action_spec()
    return torch.from_numpy(np.random.uniform(spec.minimum, spec.maximum, spec.shape))

  def get_xpos(self, name):
    return copy.deepcopy(self._env._physics.named.data.geom_xpos[name])[:2]

  def get_kinematic_info(self, reset=False):
    info = dict()
    for name in self.info_names:
      info[name] = self.get_xpos(name)

    if 'arm' in info.keys() and 'hand' in info.keys():
      angle, angle_complex = self.get_angle(info['arm'], info['hand'])
      info['angle'] = angle
      info['angle_comp'] = angle_complex
      info['position'] = self.get_joint_position(info['arm'], info['hand'])

      if reset:
        info['angular_velocity'] = info['angle'] * 0.
      else:
        info['angular_velocity'] = info['angle'] - self.pre_angle
      self.pre_angle = copy.deepcopy(info['angle'])
    return info

  def get_angle_complex(self, arm_pos, hand_pos):
    angle_a = np.arctan2(arm_pos[0], arm_pos[1])
    angle_h = np.arctan2(hand_pos[0]-arm_pos[0], hand_pos[1]-arm_pos[1]) - angle_a

    return np.array([np.cos(angle_a), np.cos(angle_h), np.sin(angle_a), np.sin(angle_h)])

  def get_angle(self, arm_pos, hand_pos):
    angle_complex = self.get_angle_complex(arm_pos, hand_pos)
    angle = np.array([np.angle(angle_complex[0]+angle_complex[2]*1j), np.angle(angle_complex[1]+angle_complex[3]*1j)])/np.pi
    return angle, angle_complex

  def get_joint_position(self, arm_pos, hand_pos):
    return np.hstack([arm_pos[:2], hand_pos[:2]])


class MultimodalRoboSuiteEnv():
  def __init__(self, env_config, symbolic, seed, episode_horizon, action_repeat, bit_depth, info_names=[], multimodal=True, noise_scale=0, add_noise=False, noisy_background=False):
    import robosuite as suite
    from robosuite.controllers import load_controller_config
    # load default controller parameters for Operational Space Control (OSC)
    controller_config = load_controller_config(default_controller="OSC_POSE")
    from robosuite.utils import macros
    macros.IMAGE_CONVENTION = "opencv"
    
    self.image_size = (env_config["camera_heights"], env_config["camera_widths"])
    self.image_name = "image_{}".format(env_config["camera_widths"])
    self.symbolic = symbolic

    self.camera_names = env_config["camera_names"]
    env_config = dict(env_config)
    robots = list(env_config.pop("robots"))
    self._env = suite.make(**env_config, robots=robots, controller_configs=controller_config)
    self.episode_horizon=episode_horizon
    self.action_repeat = action_repeat
    self.bit_depth = bit_depth
    self.info_names = info_names
    self.multimodal = multimodal

    self.noise_scale=noise_scale
    self.add_noise=add_noise
    self.noisy_background=noisy_background

  def _reset(self):
    self.t = 0  # Reset internal timer
    state = self._env.reset()
    if self.symbolic:
      return torch.tensor(np.concatenate([np.asarray([obs]) if isinstance(obs, float) else obs for obs in state.observation.values()], axis=0), dtype=torch.float32)
    else:
      return _images_to_observation(state[self.camera_names+"_image"], self.bit_depth,
                                    self.noise_scale, self.add_noise, self.noisy_background, size=self.image_size)

  def reset(self):
    if not self.multimodal:
      return self._reset()

    image = self._reset()
    observation = dict()
    observation[self.image_name] = image
    return observation


  def _step(self, action):
    action = action.detach().numpy()
    reward = 0
    for k in range(self.action_repeat):
      _observation, _reward, _done, _ = self._env.step(action)
      reward += _reward
      self.t += 1  # Increment internal timer
      done = _done or self.t == self.episode_horizon
      if done:
        break
    if self.symbolic:
      observation = torch.tensor(np.concatenate([np.asarray([obs]) if isinstance(
          obs, float) else obs for obs in observation.values()], axis=0), dtype=torch.float32)
    else:
      observation = _images_to_observation(
          _observation[self.camera_names+"_image"], self.bit_depth,
          self.noise_scale, self.add_noise, self.noisy_background, size=self.image_size)
    return observation, reward, done

  def step(self, action):
    if not self.multimodal:
      return self._step(action)
    image, reward, done = self._step(action)
    observation = dict()
    observation[self.image_name] = image

    return observation, reward, done

  def render(self):
    cv2.imshow('screen', self._env.physics.render(camera_id=0)[:, :, ::-1])
    cv2.waitKey(1)

  def close(self):
    cv2.destroyAllWindows()
    self._env.close()

  @property
  def observation_size(self):
    return sum([(1 if len(obs.shape) == 0 else obs.shape[0]) for obs in self._env.observation_spec().values()]) if self.symbolic else (3, *self.image_size)

  @property
  def action_size(self):
    return self._env.action_spec[0].shape[0]

  # Sample an action randomly from a uniform distribution over all valid actions
  def sample_random_action(self):
    spec = dict()
    spec["minimum"] = self._env.action_spec[0]
    spec["maximum"] = self._env.action_spec[1]
    spec["shape"] = len(self._env.action_spec[0])
    return torch.from_numpy(np.random.uniform(spec["minimum"], spec["maximum"], spec["shape"]))


# Wrapper for batching environments together
class MultimodalEnvBatcher():
  def __init__(self, env_class, env_args, env_kwargs, n):
    self.n = n

    seed_base = env_kwargs['seed']
    _env_kwargs = copy.deepcopy(env_kwargs)
    self.envs = []
    for i in range(n):
      _env_kwargs['seed'] = seed_base + i
      self.envs.append(env_class(*env_args, **_env_kwargs))
    self.dones = [True] * n

    self.action_size = self.envs[0].action_size

  def reshape_observations(self, _observations):
    observations = dict()
    for key in _observations[0].keys():
      observations[key] = torch.tensor(np.stack([_observations[i][key] for i in range(len(_observations))]), dtype=torch.float32)
    return observations

  # Resets every environment and returns observation
  def reset(self):
    _observations = [env.reset() for env in self.envs]
    
    observations = self.reshape_observations(_observations)

    self.dones = [False] * self.n
    return observations

 # Steps/resets every environment and returns (observation, reward, done)
  def step(self, actions):
    # Done mask to blank out observations and zero rewards for previously terminated environments
    done_mask = torch.nonzero(torch.tensor(self.dones))[:, 0]
    _observations, rewards, dones = zip(
        *[env.step(action) for env, action in zip(self.envs, actions)])
    # Env should remain terminated if previously terminated
    dones = [d or prev_d for d, prev_d in zip(dones, self.dones)]
    self.dones = dones
    observations = self.reshape_observations(_observations)
    rewards, dones = torch.tensor(rewards, dtype=torch.float32), torch.tensor(dones, dtype=torch.uint8)
    for key in observations.keys():
      observations[key][done_mask] = 0
    rewards[done_mask] = 0
    return observations, rewards, dones

  def close(self):
    [env.close() for env in self.envs]

  def sample_random_action(self):
    actions = []
    for env in self.envs:
      actions.append(env.sample_random_action())
    return torch.stack(actions)
