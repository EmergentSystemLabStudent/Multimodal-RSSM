import os
import numpy as np
import torch
from tqdm import tqdm

import glob

from common.env import postprocess_observation, preprocess_observation_
from algos.base.processing import preprocess_pose
from algos.base.data_augment import augment_image_data, crop_image_data, torch_cov

from omegaconf import ListConfig

def _load_dataset(cfg, cwd, experience_replay, D, metrics=None):
    dataset_dir = os.path.join(cwd, experience_replay)
    
    if os.path.exists(dataset_dir):    
        print("load dataset from {}".format(dataset_dir))
        if os.path.isdir(dataset_dir):
            D.load_dataset(dataset_dir=dataset_dir, 
                           n_episode=cfg.train.n_episode)
            if not metrics is None:
                metrics['steps'], metrics['episodes'] = [D.steps] * \
                    D.episodes, list(range(1, D.episodes + 1))
        else:
            dataset = torch.load(dataset_dir)
            D.convert_dataset(dataset)
            if not metrics is None:
                metrics['steps'], metrics['episodes'] = [D.steps] * \
                    D.episodes, list(range(1, D.episodes + 1))
    else:
        raise NotImplementedError("{} is not exist".format(dataset_dir))

def load_dataset(cfg, cwd, D, experience_replay, metrics=None):
    if type(experience_replay) == str:
        _load_dataset(cfg, cwd, experience_replay, D, metrics)
    elif type(experience_replay) == ListConfig:
        for exp_replay in experience_replay:
            _load_dataset(cfg, cwd, exp_replay, D, metrics)


def clip_episode(data):
    data_length = []
    for k in data.keys():
        if not k == "seed":
            data_length.append(len(data[k]))
    episode_length = np.min(data_length)
    output = dict()
    for k in data.keys():
        if not k == "seed":
            output[k] = data[k][:episode_length]
    return output


def preprocess_data(data):
    data = clip_episode(data)

    for name in data.keys():
        if ("image" in name) and (data[name].shape[1] > data[name].shape[3]):
            data[name] = data[name].transpose(0, 3, 1, 2)
        if ("image" in name) and (not type(data[name][0,0,0,0]) == np.uint8):
            data[name] = postprocess_observation(data[name])
    
    if "image" in data.keys():
        image_shape = data["image"].shape[2]
        if not image_shape == 64: 
            data["image_{}".format(image_shape)] = data.pop("image")

    data = preprocess_pose(data)

    # crop sound
    if "sound" in data.keys():
        data["sound_crop"] = data["sound"][:,:64]

    if "weight_value_norm" in data.keys():
        data["weight_value_norm"] = np.expand_dims(data["weight_value_norm"], -1)

    data["nonterminals"] = 1. - np.expand_dims(data["done"], -1)
    return data

def calc_image_shape(shape, n_crop=None, dw_base=2, dh_base=2):
    if n_crop is None:
        return shape
    else:
        d, h, w = shape
        k = int(np.sqrt(n_crop-1))
        return [d, int(h+k*dh_base), int(w+k*dw_base)]

def to_tensor(data, device=torch.device("cpu")):
    if not torch.is_tensor(data):
        if data.dtype == np.uint8:
            return torch.tensor(data, dtype=torch.uint8, device=device)
        else:
            return torch.tensor(data, dtype=torch.float32, device=device)
    else:
        return data

def get_file_names(dataset_dir):
    file_names = glob.glob(os.path.join(dataset_dir, '*.npy'))
    return file_names

def get_data(file_name, n_crop=1, dh_base=1, dw_base=1, encoding='ASCII'):
    _data = np.load(file_name, allow_pickle=True, encoding=encoding).item()
    if encoding == 'ASCII':
        data = _data
    else:
        data = dict()
        for key in _data.keys():
            data[key.decode('utf-8')] = _data[key]

    data = preprocess_data(data)
    data = crop_image_data(data, n_crop=n_crop, dh_base=dh_base, dw_base=dw_base)
    
    episode_length = None
    for name in data.keys():
        if episode_length is None:
            episode_length = len(data[name])
        elif episode_length > len(data[name]):
            episode_length = len(data[name])

    idx_start, idx_end = 0, episode_length

    for name in data.keys():
        data[name] = data[name][idx_start:idx_end]
    
    return data, idx_end-idx_start

class ExperienceReplay_Multimodal:
    def __init__(self,
                 size,
                 observation_names=["image"],
                 observation_shapes=dict(image=[3, 64, 64]),
                 n_crop=None,
                 noise_scales=None,
                 pca_scales=None,
                 dh_base=None,
                 dw_base=None,
                 action_name="action",
                 action_size=None,
                 bit_depth=5,
                 device=torch.device("cpu"),
                 taskset_name="cobotta"):
        self.taskset_name = taskset_name
        self.device = device
        self.size = size
        self.observation_names = observation_names
        self.observation_shapes = observation_shapes
        self.action_name = action_name
        self.action_size = action_size

        self.idx = 0
        self.full = False  # Tracks if memory has been filled/all slots are valid
        # Tracks how much experience has been used in total
        self.steps, self.episodes = 0, 0
        self.bit_depth = bit_depth
        self.bit_depth_t = torch.tensor(bit_depth)
        self.episode_length = None

        self.n_crop = n_crop
        self.noise_scales = noise_scales
        self.pca_scales = pca_scales
        self.dh_base = dh_base
        self.dw_base = dw_base

        self.lambd_eigen_values = dict()
        self.p_eigen_vectors = dict()
        for name in observation_names:
            self.lambd_eigen_values[name] = None
            self.p_eigen_vectors[name] = None

        self.init_buffer(size)

        self.file_names = []

    def init_buffer(self, size):
        self.observations = dict()
        for name in self.observation_names:
            if "image" in name:
                self.observations[name] = torch.empty(
                    (size,
                     *calc_image_shape(
                         self.observation_shapes[name],
                         self.n_crop,
                         self.dw_base,
                         self.dh_base)),
                    dtype=torch.uint8)
            else:
                self.observations[name] = torch.empty(
                    (size, *self.observation_shapes[name]), dtype=torch.float32)
        self.actions = torch.empty((size, self.action_size), dtype=torch.float32)
        self.rewards = torch.empty((size, ), dtype=torch.float32)
        self.nonterminals = torch.empty((size, 1), dtype=torch.float32)

    # Returns an index for a valid single sequence chunk uniformly sampled
    # from the memory
    def _sample_idx(self, L, idx_max=None):
        valid_idx = False
        _idx_max = self.size if self.full else self.idx - L
        if not (idx_max is None):
            _idx_max = np.min([idx_max, _idx_max])
        while not valid_idx:
            idx = np.random.randint(0, _idx_max)
            idxs = np.arange(idx, idx + L) % self.size
            # Make sure data does not cross the memory index
            valid_idx = self.idx not in idxs[1:]
        return idxs

    def _retrieve_batch(self, idxs, n, L):
        vec_idxs = idxs.transpose().reshape(-1)  # Unroll indices
        
        pca_rand = None
        observations = dict()
        for name in self.observation_names:
            observations[name] = self.observations[name][vec_idxs].reshape(
                L, n, *self.observations[name].shape[1:]).to(self.device).to(torch.float32)
            if "image" in name:
                if "bin" in name:
                    observations[name], _ = augment_image_data(
                        observations[name], name, self.n_crop, self.dh_base, self.dw_base)
                else:
                    observations[name], pca_rand = augment_image_data(
                        observations[name], name, self.n_crop, self.dh_base, self.dw_base, self.noise_scales, pca_rand, self.lambd_eigen_values[name], self.p_eigen_vectors[name], self.pca_scales)
                    observations[name] = preprocess_observation_(observations[name], torch.tensor(self.bit_depth))

        actions = self.actions[vec_idxs].reshape(L, n, -1).to(device=self.device)
        rewards = self.rewards[vec_idxs].reshape(L, n).to(device=self.device)
        nonterminals = self.nonterminals[vec_idxs].reshape(L, n, 1).to(device=self.device)
        return observations, actions, rewards, nonterminals

    # Returns a batch of sequence chunks uniformly sampled from the memory
    def sample(self, n, L):
        batch = self._retrieve_batch(np.asarray(
            [self._sample_idx(L) for _ in range(n)]), n, L)
        # [1578 1579 1580 ... 1625 1626 1627]
        # [1049 1050 1051 ... 1096 1097 1098]
        # [1236 1237 1238 ... 1283 1284 1285]
        # ...
        # [2199 2200 2201 ... 2246 2247 2248]
        # [ 686  687  688 ...  733  734  735]
        # [1377 1378 1379 ... 1424 1425 1426]]
        return [item for item in batch]

    # buffer data from environment
    def append(self, observation, action, reward, done):
        for name in self.observation_names:
            if "image" in name:
                self.observations[name][self.idx] = torch.tensor(postprocess_observation(observation[name], self.bit_depth))
            else:
                self.observations[name][self.idx] = torch.tensor(observation[name])
        
        self.actions[self.idx] = action
        self.rewards[self.idx] = torch.tensor(reward)
        self.nonterminals[self.idx] = not done
        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0
        self.steps, self.episodes = self.steps + \
            1, self.episodes + (1 if done else 0)

    # buffer data from demonstration data
    def set_data_to_buffer(self, file_name):
        data, episode_length = get_data(file_name, self.n_crop, self.dh_base, self.dw_base)

        idx = np.arange(self.idx, self.idx + episode_length)

        for name in self.observation_names:
            self.observations[name][idx] = to_tensor(data[name])
        if self.action_name == "dummy":
            self.actions[idx] = to_tensor(np.zeros(
                (episode_length, self.actions.shape[-1]), dtype=np.float32))
        else:
            self.actions[idx] = to_tensor(data[self.action_name])
        self.rewards[idx] = to_tensor(data["reward"])
        self.nonterminals[idx] = to_tensor(data["nonterminals"])

        self.full = self.full or (
            self.idx + episode_length) / self.size >= 1
        self.idx = (self.idx + episode_length) % self.size
        self.steps = self.steps + episode_length
        self.episodes += 1

    def load_dataset(self, dataset_dir, n_episode=None):
        file_names = get_file_names(dataset_dir)
        # print(file_names)
        if n_episode is None:
            n_episode = len(file_names)
        else:
            n_episode = np.min((n_episode, len(file_names)))

        print("find %d npy files!" % len(file_names))
        self.file_names += file_names[:n_episode]

        for file_name in tqdm(file_names[:n_episode], desc="load dataset"):
            # tqdm.write(file_name)
            self.set_data_to_buffer(file_name)

        if not self.pca_scales is None:
            print("set color augment params")
            self.set_color_aug_params()

    def load_observation(self, data, i, idx):
        for name in self.observation_names:
            if name not in data[i].keys():
                observation = data[i]["observation"][name]
            else:
                observation = data[i][name]

            if name == self.action_name:
                self.observations[name][idx] = np.vstack([np.zeros(
                    (1, 1, observation.shape[-1]), dtype=np.float32, device=observation.device), observation[:len(idx) - 1]])
            else:
                self.observations[name][idx] = observation[:len(idx)]

    def load_action(self, data, i, idx, episode_length):
        if self.action_name == "dummy":
            self.actions[idx] = np.zeros(
                (episode_length, self.actions.shape[-1]), dtype=np.float32)
        else:
            self.actions[idx] = data[i][self.action_name]

    def set_dataset(self, data, i, episode_length):
        idx = self.idx + np.arange(episode_length)
        self.load_observation(data, i, idx)
        self.load_action(data, i, idx, episode_length)
        self.rewards[idx] = data[i]["reward"]
        self.nonterminals[idx] = np.expand_dims(data[i]["nonterminals"], -1)

        self.full = self.full or (self.idx + episode_length) / self.size >= 1
        self.idx = (self.idx + episode_length) % self.size
        self.steps = self.steps + episode_length
        self.episodes = self.episodes + 1
        self.episode_length = episode_length

    # output all data
    def all_data(self):
        observations = dict()
        for name in self.observation_names:
            observations[name] = self.observations[name][:self.idx]
        actions = self.actions[:self.idx]
        rewards = self.rewards[:self.idx]
        nonterminals = self.nonterminals[:self.idx]
        return observations, actions, rewards, nonterminals

    def episode_data(self, idx=0, crop_idx=None, pca_rand=None):
        idx_done, _ = np.where(self.nonterminals[:self.idx] == 0)
        idx_done = np.hstack([[0], idx_done + 1])
        idx_start = idx_done[idx]
        idx_end = idx_done[idx + 1]
        _observations, _actions, _rewards, _nonterminals = self.all_data()

        observations = dict()
        for name in _observations.keys():
            observations[name] = _observations[name][idx_start:idx_end].to(device=self.device).unsqueeze(1).to(torch.float32)
            if "image" in name:
                if "bin" in name:
                    observations[name], _ = augment_image_data(
                        observations[name], name, self.n_crop, self.dh_base, self.dw_base, crop_idx=crop_idx)
                else:
                    observations[name], pca_rand = augment_image_data(
                        observations[name], name, self.n_crop, self.dh_base, self.dw_base, self.noise_scales, pca_rand, self.lambd_eigen_values[name], self.p_eigen_vectors[name], self.pca_scales, crop_idx=crop_idx)
                    observations[name] = preprocess_observation_(observations[name], torch.tensor(self.bit_depth))

        actions = _actions[idx_start:idx_end].to(device=self.device).unsqueeze(1)
        rewards = _rewards[idx_start:idx_end].to(self.device).unsqueeze(1)
        nonterminals = _nonterminals[idx_start:idx_end].to(self.device).unsqueeze(1)

        return observations, actions, rewards, nonterminals

    # data augmentation
    def calc_pca(self, image):
        image = image[0::100]
        print("calc pca from {} data".format(image.shape))
        
        image = image.reshape(3, -1).to(torch.float32)
        image = (image.transpose(1, 0) - torch.mean(image, axis=1)) / torch.std(image, axis=1)

        cov = torch_cov(image, rowvar=False)
        lambd_eigen_value, p_eigen_vector = torch.symeig(cov, eigenvectors=True)
        return lambd_eigen_value.to(self.device), p_eigen_vector.to(self.device)
        
    def set_color_aug_params(self):
        self.lambd_eigen_values = dict()
        self.p_eigen_vectors = dict()

        for name in self.observations.keys():
            if ("image" in name) and (not "bin" in name):
                lambd_eigen_value, p_eigen_vector = self.calc_pca(self.observations[name][:self.idx])
                self.lambd_eigen_values[name] = lambd_eigen_value
                self.p_eigen_vectors[name] = p_eigen_vector
