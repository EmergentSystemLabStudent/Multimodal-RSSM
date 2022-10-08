import os
import numpy as np
import torch
from tqdm import tqdm

import glob

from utils.processing.image_processing import normalize_image, reverse_normalized_image
from utils.replay_buffer.data_augment import augment_image_data, crop_image_data, calc_params_of_pca

from omegaconf import ListConfig

def _load_dataset(cfg, cwd, dataset_path, D):
    dataset_dir = os.path.join(cwd, dataset_path)
    
    if os.path.exists(dataset_dir):    
        print("load dataset from {}".format(dataset_dir))
        if os.path.isdir(dataset_dir):
            D.load_dataset(dataset_dir=dataset_dir)
        else:
            dataset = torch.load(dataset_dir)
            D.convert_dataset(dataset)
    else:
        raise NotImplementedError("{} is not exist".format(dataset_dir))

# load dataset from dataset path
def load_dataset(cfg, cwd, D, dataset_path):
    if type(dataset_path) == str:
        _load_dataset(cfg, cwd, dataset_path, D)
    elif type(dataset_path) == ListConfig:
        for exp_replay in dataset_path:
            _load_dataset(cfg, cwd, exp_replay, D)

# clip episode by minimum length modal
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
    return output, episode_length

# preprocess for strage buffer
def preprocess_data(data):
    data, episode_length = clip_episode(data)

    for name in data.keys():
        if ("image" in name) and (data[name].shape[1] > data[name].shape[3]):
            data[name] = data[name].transpose(0, 3, 1, 2)
        if ("image" in name) and (not type(data[name][0,0,0,0]) == np.uint8):
            data[name] = reverse_normalized_image(data[name])
    
    if "image" in data.keys():
        image_shape = data["image"].shape[2]
        if not image_shape == 64: 
            data["image_{}".format(image_shape)] = data.pop("image")

    data["nonterminals"] = 1. - np.expand_dims(data["done"], -1)
    return data, episode_length

# calc image shape of buffer
def calc_image_shape(shape, n_crop=None, dw_base=2, dh_base=2):
    if n_crop is None:
        return shape
    else:
        d, h, w = shape
        k = int(np.sqrt(n_crop-1))
        return [d, int(h+k*dh_base), int(w+k*dw_base)]

# trans data from numpy to torch tensor
def np2tensor(data, device=torch.device("cpu")):
    if not torch.is_tensor(data):
        if data.dtype == np.uint8:
            return torch.tensor(data, dtype=torch.uint8, device=device)
        else:
            return torch.tensor(data, dtype=torch.float32, device=device)
    else:
        return data

# get file names from dataset directory
def get_file_names(dataset_dir):
    file_names = glob.glob(os.path.join(dataset_dir, '*.npy'))
    return file_names

# get data from file path
def get_data(file_name, n_crop=1, dh_base=1, dw_base=1, encoding='ASCII'):
    _data = np.load(file_name, allow_pickle=True, encoding=encoding).item()
    if encoding == 'ASCII':
        data = _data
    else:
        data = dict()
        for key in _data.keys():
            data[key.decode('utf-8')] = _data[key]

    data, episode_length = preprocess_data(data)
    data = crop_image_data(data, n_crop=n_crop, dh_base=dh_base, dw_base=dw_base)
    
    idx_start, idx_end = 0, episode_length

    for name in data.keys():
        data[name] = data[name][idx_start:idx_end]
    
    return data, idx_end-idx_start

# buffer class
class ExperienceReplay_Multimodal:
    def __init__(self,
                 size,
                 observation_names=["image"],
                 observation_shapes=dict(image=[3, 64, 64]),
                 n_crop=None,
                 dh_base=None,
                 dw_base=None,
                 noise_scales=None,
                 pca_scales=None,
                 action_name="action",
                 action_size=None,
                 bit_depth=5,
                 device=torch.device("cpu")):
        self.device = device
        self.size = size
        self.observation_names = observation_names
        self.observation_shapes = observation_shapes
        self.action_name = action_name
        self.action_size = action_size
        self.file_names = []

        self.idx = 0
        self.full = False  # Tracks if memory has been filled/all slots are valid
        # Tracks how much experience has been used in total
        self.steps, self.episodes = 0, 0
        self.bit_depth = bit_depth

        # params for Data augmentation
        self.n_crop = n_crop
        self.dh_base = dh_base
        self.dw_base = dw_base

        self.noise_scales = noise_scales
        
        self.pca_scales = pca_scales
        self.lambd_eigen_values = dict()
        self.p_eigen_vectors = dict()
        for name in observation_names:
            self.lambd_eigen_values[name] = None
            self.p_eigen_vectors[name] = None

        # initialize buffer
        self._init_buffer(size)

    # initialize buffer
    def _init_buffer(self, size):
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
                    observations[name] = normalize_image(observations[name], torch.tensor(self.bit_depth))

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
                self.observations[name][self.idx] = torch.tensor(reverse_normalized_image(observation[name], self.bit_depth))
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
    def _set_data_to_buffer(self, file_name):
        data, episode_length = get_data(file_name, self.n_crop, self.dh_base, self.dw_base)

        idx = np.arange(self.idx, self.idx + episode_length)

        for name in self.observation_names:
            self.observations[name][idx] = np2tensor(data[name])
        if self.action_name == "dummy":
            self.actions[idx] = np2tensor(np.zeros(
                (episode_length, self.actions.shape[-1]), dtype=np.float32))
        else:
            self.actions[idx] = np2tensor(data[self.action_name])
        self.rewards[idx] = np2tensor(data["reward"])
        self.nonterminals[idx] = np2tensor(data["nonterminals"])

        self.full = self.full or (
            self.idx + episode_length) / self.size >= 1
        self.idx = (self.idx + episode_length) % self.size
        self.steps = self.steps + episode_length
        self.episodes += 1

    def load_dataset(self, dataset_dir):
        file_names = get_file_names(dataset_dir)
        
        print("find %d npy files!" % len(file_names))
        self.file_names += file_names

        for file_name in tqdm(file_names, desc="load dataset"):
            self._set_data_to_buffer(file_name)

        if not self.pca_scales is None:
            print("set color augment params")
            self._set_color_aug_params()

    # data augmentation    
    def _set_color_aug_params(self):
        self.lambd_eigen_values = dict()
        self.p_eigen_vectors = dict()

        for name in self.observations.keys():
            if ("image" in name) and (not "bin" in name):
                lambd_eigen_value, p_eigen_vector = calc_params_of_pca(self.observations[name][:self.idx])
                self.lambd_eigen_values[name] = lambd_eigen_value.to(self.device)
                self.p_eigen_vectors[name] = p_eigen_vector.to(self.device)
