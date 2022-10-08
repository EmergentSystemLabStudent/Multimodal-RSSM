from typing import Dict
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions

from torch.distributions import Normal

class ObservationModel_base(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def get_state_dict(self):
        return self.state_dict()

    def _load_state_dict(self, state_dict):
        self.load_state_dict(state_dict)

    def get_model_params(self):
        return list(self.parameters())

    def get_log_prob(self, h_t, s_t, o_t):
        loc_and_scale = self.forward(h_t, s_t)
        dist = Normal(loc_and_scale['loc'], loc_and_scale['scale'])
        log_prob = dist.log_prob(o_t)
        return log_prob

    def get_mse(self, h_t, s_t, o_t):
        loc_and_scale = self.forward(h_t, s_t)
        mse = F.mse_loss(loc_and_scale['loc'], o_t, reduction='none')
        return mse

class DenseDecoder(ObservationModel_base):
    def __init__(self, observation_size: torch.Tensor, belief_size: torch.Tensor, state_size: int, embedding_size: int, activation_function: str =      'relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, observation_size)
        self.modules = [self.fc1, self.fc2, self.fc3]

    def forward(self, h_t, s_t) -> Dict:
        # reshape inputs
        (T, B), features_shape = h_t.size()[:2], h_t.size()[2:]
        h_t = h_t.reshape(T*B, *features_shape)
        (T, B), features_shape = s_t.size()[:2], s_t.size()[2:]
        s_t = s_t.reshape(T*B, *features_shape)

        hidden = self.act_fn(self.fc1(torch.cat([h_t, s_t], dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        observation = self.fc3(hidden)
        features_shape = observation.size()[1:]
        observation = observation.reshape(T, B, *features_shape)
        return {'loc': observation, 'scale': 1.0}

    

class ImageDecoder(ObservationModel_base):
    __constants__ = ['embedding_size']

    def __init__(self, belief_size: int, state_size: int, embedding_size: int, activation_function: str='relu', image_dim=3, normalization=None):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        if normalization == None:
            self.conv = nn.Sequential(nn.ConvTranspose2d(embedding_size, 128, 5, stride=2),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(128, 64, 5, stride=2),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(64, 32, 6, stride=2),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(32, image_dim, 6, stride=2)
            )
        elif normalization == "BatchNorm":
            self.conv = nn.Sequential(nn.ConvTranspose2d(embedding_size, 128, 5, stride=2,bias=False),
                                        nn.BatchNorm2d(128, affine=True, track_running_stats=True),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(128, 64, 5, stride=2,bias=False),
                                        nn.BatchNorm2d(64, affine=True, track_running_stats=True),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(64, 32, 6, stride=2,bias=False),
                                        nn.BatchNorm2d(32, affine=True, track_running_stats=True),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(32, image_dim, 6, stride=2)
            )
        else:
            raise NotImplementedError
        self.modules = [self.fc1, self.conv]

    def forward(self, h_t: torch.Tensor, s_t: torch.Tensor) -> Dict:
        # reshape input tensors
        (T, B), features_shape = h_t.size()[:2], h_t.size()[2:]
        h_t = h_t.reshape(T*B, *features_shape)

        (T, B), features_shape = s_t.size()[:2], s_t.size()[2:]
        s_t = s_t.reshape(T*B, *features_shape)

        # No nonlinearity here
        hidden = self.fc1(torch.cat([h_t, s_t], dim=1))
        hidden = hidden.reshape(-1, self.embedding_size, 1, 1)
        observation = self.conv(hidden)
        features_shape = observation.size()[1:]
        observation = observation.reshape(T, B, *features_shape)
        return {'loc': observation, 'scale': 1.0}


class ImageDecoder_84(ObservationModel_base):
    __constants__ = ['embedding_size']

    def __init__(self, belief_size: int, state_size: int, embedding_size: int, activation_function: str='relu', image_dim=3, normalization=None):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.fc = nn.Linear(belief_size + state_size, embedding_size)
        if normalization == None:
            self.conv = nn.Sequential(nn.ConvTranspose2d(embedding_size, 128, 3, stride=2),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(128, 64, 4, stride=2),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(64, 32, 4, stride=2),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(32, 16, 6, stride=2),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(16, image_dim, 6, stride=2)
            )
        elif normalization == "BatchNorm":
            self.conv = nn.Sequential(nn.ConvTranspose2d(embedding_size, 128, 3, stride=2,bias=False),
                                        nn.BatchNorm2d(128, affine=True, track_running_stats=True),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(128, 64, 4, stride=2,bias=False),
                                        nn.BatchNorm2d(64, affine=True, track_running_stats=True),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(64, 32, 4, stride=2,bias=False),
                                        nn.BatchNorm2d(32, affine=True, track_running_stats=True),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(32, 16, 6, stride=2,bias=False),
                                        nn.BatchNorm2d(16, affine=True, track_running_stats=True),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(16, image_dim, 6, stride=2)
            )
        else:
            raise NotImplementedError
        self.modules = [self.fc, self.conv]

    def forward(self, h_t: torch.Tensor, s_t: torch.Tensor) -> Dict:
        # reshape input tensors
        (T, B), features_shape = h_t.size()[:2], h_t.size()[2:]
        h_t = h_t.reshape(T*B, *features_shape)

        (T, B), features_shape = s_t.size()[:2], s_t.size()[2:]
        s_t = s_t.reshape(T*B, *features_shape)

        # No nonlinearity here
        hidden = self.fc(torch.cat([h_t, s_t], dim=1))
        hidden = hidden.reshape(-1, self.embedding_size, 1, 1)
        observation = self.conv(hidden)
        features_shape = observation.size()[1:]
        observation = observation.reshape(T, B, *features_shape)
        return {'loc': observation, 'scale': 1.0}
    
class ImageDecoder_128(ObservationModel_base):
    __constants__ = ['embedding_size']

    def __init__(self, belief_size: int, state_size: int, embedding_size: int, activation_function: str='relu', image_dim=3, normalization=None):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        scale = 2
        if normalization == None:
            self.conv = nn.Sequential(nn.ConvTranspose2d(embedding_size, 128*scale, 6, stride=2,bias=True),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(128*scale, 64*scale, 4, stride=2,bias=True),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(64*scale, 32*scale, 4, stride=2,bias=True),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(32*scale, 16*scale, 4, stride=2,bias=True),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(16*scale, image_dim, 6, stride=2)
            )
        elif normalization == "BatchNorm":
            # self.conv = nn.Sequential(nn.ConvTranspose2d(embedding_size, 128*scale, 5, stride=2,bias=False),
            #                         nn.BatchNorm2d(128*scale, affine=True, track_running_stats=True),
            #                         nn.ReLU(),
            #                         nn.ConvTranspose2d(128*scale, 64*scale, 5, stride=2,bias=False),
            #                         nn.BatchNorm2d(64*scale, affine=True, track_running_stats=True),
            #                         nn.ReLU(),
            #                         nn.ConvTranspose2d(64*scale, 32*scale, 5, stride=2,bias=False),
            #                         nn.BatchNorm2d(32*scale, affine=True, track_running_stats=True),
            #                         nn.ReLU(),
            #                         nn.ConvTranspose2d(32*scale, 16*scale, 6, stride=2,bias=False),
            #                         nn.BatchNorm2d(16*scale, affine=True, track_running_stats=True),
            #                         nn.ReLU(),
            #                         nn.ConvTranspose2d(16*scale, image_dim, 6, stride=2)
            # )
            self.conv = nn.Sequential(nn.ConvTranspose2d(embedding_size, 128*scale, 6, stride=2,bias=False),
                                    nn.BatchNorm2d(128*scale, affine=True, track_running_stats=True),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(128*scale, 64*scale, 4, stride=2,bias=False),
                                    nn.BatchNorm2d(64*scale, affine=True, track_running_stats=True),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(64*scale, 32*scale, 4, stride=2,bias=False),
                                    nn.BatchNorm2d(32*scale, affine=True, track_running_stats=True),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(32*scale, 16*scale, 4, stride=2,bias=False),
                                    nn.BatchNorm2d(16*scale, affine=True, track_running_stats=True),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(16*scale, image_dim, 6, stride=2)
            )
        else:
            raise NotImplementedError
        self.modules = [self.fc1, self.conv]

    def forward(self, h_t: torch.Tensor, s_t: torch.Tensor) -> Dict:
        # reshape input tensors
        (T, B), features_shape = h_t.size()[:2], h_t.size()[2:]
        h_t = h_t.reshape(T*B, *features_shape)

        (T, B), features_shape = s_t.size()[:2], s_t.size()[2:]
        s_t = s_t.reshape(T*B, *features_shape)

        # No nonlinearity here
        hidden = self.fc1(torch.cat([h_t, s_t], dim=1))
        hidden = hidden.reshape(-1, self.embedding_size, 1, 1)
        observation = self.conv(hidden)
        features_shape = observation.size()[1:]
        observation = observation.reshape(T, B, *features_shape)
        return {'loc': observation, 'scale': 1.0}

class ImageDecoder_256(ObservationModel_base):
    __constants__ = ['embedding_size']

    def __init__(self, belief_size: int, state_size: int, embedding_size: int, activation_function: str='relu', image_dim=3, normalization=None):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        scale = 2
        if normalization == None:
            # self.conv = nn.Sequential(nn.ConvTranspose2d(embedding_size, 128*scale, 5, stride=2,bias=True),
            #                         nn.ReLU(),
            #                         nn.ConvTranspose2d(128*scale, 64*scale, 5, stride=2,bias=True),
            #                         nn.ReLU(),
            #                         nn.ConvTranspose2d(64*scale, 32*scale, 5, stride=2,bias=True),
            #                         nn.ReLU(),
            #                         nn.ConvTranspose2d(32*scale, 16*scale, 5, stride=2,bias=True),
            #                         nn.ReLU(),
            #                         nn.ConvTranspose2d(16*scale, 8*scale, 6, stride=2,bias=True),
            #                         nn.ReLU(),
            #                         nn.ConvTranspose2d(8*scale, image_dim, 6, stride=2)
            # )
            self.conv = nn.Sequential(nn.ConvTranspose2d(embedding_size, 128*scale, 6, stride=2,bias=True),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(128*scale, 64*scale, 4, stride=2,bias=True),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(64*scale, 32*scale, 4, stride=2,bias=True),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(32*scale, 16*scale, 4, stride=2,bias=True),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(16*scale, 8*scale, 4, stride=2,bias=True),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(8*scale, image_dim, 6, stride=2)
            )
        elif normalization == "BatchNorm":
            # self.conv = nn.Sequential(nn.ConvTranspose2d(embedding_size, 128*scale, 5, stride=2,bias=False),
            #                           nn.BatchNorm2d(128*scale, affine=True, track_running_stats=True),
            #                           nn.ReLU(),
            #                           nn.ConvTranspose2d(128*scale, 64*scale, 5, stride=2,bias=False),
            #                           nn.BatchNorm2d(64*scale, affine=True, track_running_stats=True),
            #                           nn.ReLU(),
            #                           nn.ConvTranspose2d(64*scale, 32*scale, 5, stride=2,bias=False),
            #                           nn.BatchNorm2d(32*scale, affine=True, track_running_stats=True),
            #                           nn.ReLU(),
            #                           nn.ConvTranspose2d(32*scale, 16*scale, 5, stride=2,bias=False),
            #                           nn.BatchNorm2d(16*scale, affine=True, track_running_stats=True),
            #                           nn.ReLU(),
            #                           nn.ConvTranspose2d(16*scale, 8*scale, 6, stride=2,bias=False),
            #                           nn.BatchNorm2d(8*scale, affine=True, track_running_stats=True),
            #                           nn.ReLU(),
            #                           nn.ConvTranspose2d(8*scale, image_dim, 6, stride=2)

            # )
            self.conv = nn.Sequential(nn.ConvTranspose2d(embedding_size, 128*scale, 6, stride=2,bias=False),
                                      nn.BatchNorm2d(128*scale, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(128*scale, 64*scale, 4, stride=2,bias=False),
                                      nn.BatchNorm2d(64*scale, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(64*scale, 32*scale, 4, stride=2,bias=False),
                                      nn.BatchNorm2d(32*scale, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(32*scale, 16*scale, 4, stride=2,bias=False),
                                      nn.BatchNorm2d(16*scale, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(16*scale, 8*scale, 4, stride=2,bias=False),
                                      nn.BatchNorm2d(8*scale, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(8*scale, image_dim, 6, stride=2)

            )
        elif normalization == "InstanceNorm":
            # self.conv = nn.Sequential(nn.ConvTranspose2d(embedding_size, 128*scale, 5, stride=2,bias=False),
            #                           nn.InstanceNorm2d(128*scale, affine=True, track_running_stats=True),
            #                           nn.ReLU(),
            #                           nn.ConvTranspose2d(128*scale, 64*scale, 5, stride=2,bias=False),
            #                           nn.InstanceNorm2d(64*scale, affine=True, track_running_stats=True),
            #                           nn.ReLU(),
            #                           nn.ConvTranspose2d(64*scale, 32*scale, 5, stride=2,bias=False),
            #                           nn.InstanceNorm2d(32*scale, affine=True, track_running_stats=True),
            #                           nn.ReLU(),
            #                           nn.ConvTranspose2d(32*scale, 16*scale, 5, stride=2,bias=False),
            #                           nn.InstanceNorm2d(16*scale, affine=True, track_running_stats=True),
            #                           nn.ReLU(),
            #                           nn.ConvTranspose2d(16*scale, 8*scale, 6, stride=2,bias=False),
            #                           nn.InstanceNorm2d(8*scale, affine=True, track_running_stats=True),
            #                           nn.ReLU(),
            #                           nn.ConvTranspose2d(8*scale, image_dim, 6, stride=2)

            # )
            self.conv = nn.Sequential(nn.ConvTranspose2d(embedding_size, 128*scale, 6, stride=2,bias=False),
                                      nn.InstanceNorm2d(128*scale, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(128*scale, 64*scale, 4, stride=2,bias=False),
                                      nn.InstanceNorm2d(64*scale, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(64*scale, 32*scale, 4, stride=2,bias=False),
                                      nn.InstanceNorm2d(32*scale, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(32*scale, 16*scale, 4, stride=2,bias=False),
                                      nn.InstanceNorm2d(16*scale, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(16*scale, 8*scale, 4, stride=2,bias=False),
                                      nn.InstanceNorm2d(8*scale, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(8*scale, image_dim, 6, stride=2)

            )
        elif normalization == "GroupNorm":
            num_groups = 4
            self.conv = nn.Sequential(nn.ConvTranspose2d(embedding_size, 128*scale, 6, stride=2,bias=False),
                                      nn.GroupNorm(num_groups=num_groups, num_channels=128*scale, affine=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(128*scale, 64*scale, 4, stride=2,bias=False),
                                      nn.GroupNorm(num_groups=num_groups, num_channels=64*scale, affine=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(64*scale, 32*scale, 4, stride=2,bias=False),
                                      nn.GroupNorm(num_groups=num_groups, num_channels=32*scale, affine=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(32*scale, 16*scale, 4, stride=2,bias=False),
                                      nn.GroupNorm(num_groups=num_groups, num_channels=16*scale, affine=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(16*scale, 8*scale, 4, stride=2,bias=False),
                                      nn.GroupNorm(num_groups=num_groups, num_channels=8*scale, affine=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(8*scale, image_dim, 6, stride=2)

            )
        else:
            raise NotImplementedError

        self.modules = [self.fc1, self.conv]

    def forward(self, h_t: torch.Tensor, s_t: torch.Tensor) -> Dict:
        # reshape input tensors
        (T, B), features_shape = h_t.size()[:2], h_t.size()[2:]
        h_t = h_t.reshape(T*B, *features_shape)

        (T, B), features_shape = s_t.size()[:2], s_t.size()[2:]
        s_t = s_t.reshape(T*B, *features_shape)

        # No nonlinearity here
        hidden = self.fc1(torch.cat([h_t, s_t], dim=1))
        hidden = hidden.reshape(-1, self.embedding_size, 1, 1)
        observation = self.conv(hidden)
        features_shape = observation.size()[1:]
        observation = observation.reshape(T, B, *features_shape)
        return {'loc': observation, 'scale': 1.0}

class SoundDecoder(ObservationModel_base):
    def __init__(self, belief_size: int, state_size: int):
        super(SoundDecoder, self).__init__()
        self.state_size = state_size
        self.belief_size = belief_size

        self.fc1 = nn.Sequential(nn.Linear(self.state_size + self.belief_size, 250), nn.Tanh(),
                                 nn.Linear(250, 250))
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(5, 64, kernel_size=(5, 5), stride=(3, 1), padding=(1, 2),bias=False),
                                  nn.BatchNorm2d(64, affine=True, track_running_stats=True), 
                                  nn.GLU(dim=1),
                                  nn.ConvTranspose2d(32, 128, kernel_size=(5, 5), stride=(1, 1), padding=(1, 2),bias=False),
                                  nn.BatchNorm2d(128, affine=True, track_running_stats=True), 
                                  nn.GLU(dim=1),
                                  nn.ConvTranspose2d(64, 64, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3),bias=False),
                                  nn.BatchNorm2d(64, affine=True, track_running_stats=True), 
                                  nn.GLU(dim=1),
                                  nn.ConvTranspose2d(32, 32, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3),bias=False),
                                  nn.BatchNorm2d(32, affine=True, track_running_stats=True), 
                                  nn.GLU(dim=1),
                                  nn.ConvTranspose2d(16, 1, kernel_size=(3, 9), stride=(1, 1), padding=(1, 4),bias=False))
        
        self.modules = [self.conv1, self.fc1]

    def forward(self, s_t: torch.Tensor, h_t: torch.Tensor):
        (T, B), features_shape = h_t.size()[:2], h_t.size()[2:]
        h_t = h_t.reshape(T*B, *features_shape)
        (T, B), features_shape = s_t.size()[:2], s_t.size()[2:]
        s_t = s_t.reshape(T*B, *features_shape)
        x = torch.cat([h_t, s_t], dim=1)
        recon = self.fc1(x.reshape(T*B,-1))
        recon = self.conv1(recon.reshape(T*B, 5, 10, 5))
        recon = recon.squeeze(1)
        features_shape = recon.size()[1:]
        recon = recon.reshape(T, B, *features_shape)
        # return recon
        return {'loc': recon, 'scale': 1.0}
    

# inspired by https://github.com/SamuelBroughton/StarGAN-Voice-Conversion-2/blob/master/model.py
class SoundDecoder_v2(ObservationModel_base):
    def __init__(self, belief_size: int, state_size: int, channels_base=128):
        super(SoundDecoder_v2, self).__init__()
        self.state_size = state_size
        self.belief_size = belief_size
        self.channels_base = channels_base
        # Up-conversion layers.
        self.up_conversion = nn.Conv1d(in_channels=self.state_size + self.belief_size,
                                       out_channels=int(channels_base*2*32*4),
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bias=False)

        # # Up-sampling layers.
        self.up_sample_0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=int(channels_base*2), out_channels=int(channels_base*4), kernel_size=(3,4), stride=(1,1), padding=(1,1), bias=False),
            nn.InstanceNorm2d(num_features=int(channels_base*4), affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )
        self.up_sample_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=int(channels_base*2), out_channels=int(channels_base*2), kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=int(channels_base*2), affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )
        self.up_sample_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=channels_base, out_channels=channels_base, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=channels_base, affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )

        # Out.
        self.out = nn.Conv2d(in_channels=int(channels_base/2), out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)
        
        self.modules = [self.up_conversion, self.up_sample_0, self.up_sample_1, self.up_sample_2, self.out]

    def forward(self, s_t: torch.Tensor, h_t: torch.Tensor):
        (T, B), features_shape = h_t.size()[:2], h_t.size()[2:]
        h_t = h_t.reshape(T*B, *features_shape, 1)
        (T, B), features_shape = s_t.size()[:2], s_t.size()[2:]
        s_t = s_t.reshape(T*B, *features_shape, 1)

        x = torch.cat([h_t, s_t], dim=1)
        x = self.up_conversion(x)
        x = x.view(-1, int(self.channels_base*2), 32, 4)
        x = self.up_sample_0(x)
        x = self.up_sample_1(x)
        x = self.up_sample_2(x)
        x = self.out(x)
        x = x.squeeze(1)
        features_shape = x.size()[1:]
        x = x.reshape(T, B, *features_shape)
        return {'loc': x, 'scale': 1.0}

class Discriminator(ObservationModel_base):
    def __init__(self, belief_size: int, state_size: int, hidden_size: int, output_size: int, activation='relu', eps=1e-7):
        # D(s_t, a_t)
        super().__init__()
        self.act_fn = getattr(F, activation)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)
        self.modules = [self.fc1, self.fc2, self.fc3]
        self.eps = eps

    def forward(self, s_t: torch.Tensor, a_t: torch.Tensor) -> Dict:
        # reshape input tensors
        (T, B), features_shape = s_t.size()[:2], s_t.size()[2:]
        s_t = s_t.reshape(T*B, *features_shape)

        (T, B), features_shape = a_t.size()[:2], a_t.size()[2:]
        a_t = a_t.reshape(T*B, *features_shape)

        x = torch.cat([s_t, a_t], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        x = self.fc3(hidden)
        # x = self.softmax(x)#.squeeze(dim=1)
        
        features_shape = x.size()[1:]
        x = x.reshape(T, B, *features_shape)
        
        return {'loc': x, 'scale': 1.0}
    
    def get_log_prob(self, h_t, s_t, o_t):
        loc_and_scale = self.forward(h_t, s_t)
        # return o_t * torch.log(loc_and_scale['loc']+self.eps)
        return F.cross_entropy(loc_and_scale['loc'], o_t, reduction='none')

    def get_mse(self, h_t, s_t, o_t):
        loc_and_scale = self.forward(h_t, s_t)
        # return o_t * torch.log(loc_and_scale['loc']+self.eps)
        return F.cross_entropy(loc_and_scale['loc'], o_t, reduction='none')

def build_ObservationModel(name, observation_shapes, belief_size, state_size, hidden_size, embedding_size, activation_function, normalization=None):
    if "image" in name:
        image_size = observation_shapes[name][1:]
        image_dim = observation_shapes[name][0]
        if image_size == [256,256]:
            observation_models = ImageDecoder_256(belief_size, state_size, embedding_size["image"], activation_function["cnn"], image_dim=image_dim, normalization=normalization)
        elif image_size == [128,128]:
            observation_models = ImageDecoder_128(belief_size, state_size, embedding_size["image"], activation_function["cnn"], image_dim=image_dim, normalization=normalization)
        elif image_size == [84,84]:
            observation_models = ImageDecoder_84(belief_size, state_size, embedding_size["image"], activation_function["cnn"], image_dim=image_dim, normalization=normalization)
        elif image_size == [64,64]:
            observation_models = ImageDecoder(belief_size, state_size, embedding_size["image"], activation_function["cnn"], image_dim=image_dim, normalization=normalization)
    elif "sound" in name:
        observation_models = SoundDecoder_v2(belief_size=belief_size, state_size=state_size)
    elif name == "draw_target":
        observation_models = Discriminator(belief_size=belief_size, state_size=state_size, hidden_size=hidden_size, output_size=observation_shapes[name][0])
    else:
        observation_models = DenseDecoder(observation_shapes[name][0], belief_size, state_size, embedding_size["other"], activation_function["dense"])
    return observation_models



class MultimodalObservationModel:
    __constants__ = ['embedding_size']

    def __init__(self, 
                observation_names_rec,
                observation_shapes,
                embedding_size,
                belief_size: int,
                state_size: int,
                hidden_size: int,
                activation_function,
                normalization=None,
                device=torch.device("cpu")):
        self.observation_names_rec = observation_names_rec

        self.observation_models = dict()
        self.modules = []
        for name in self.observation_names_rec:
            self.observation_models[name] = build_ObservationModel(name, observation_shapes, belief_size, state_size, hidden_size, embedding_size, activation_function, normalization=normalization).to(device)
            self.modules += self.observation_models[name].modules

    def __call__(self, h_t: torch.Tensor, s_t: torch.Tensor):
        return self.forward(h_t, s_t)

    def forward(self, h_t: torch.Tensor, s_t: torch.Tensor):
        preds = dict()
        for name in self.observation_models.keys():
            pred = self.observation_models[name](h_t, s_t)
            preds[name] = pred

        return preds

    def get_log_prob(self, h_t, s_t, o_t):
        observation_log_prob = dict()
        for name in self.observation_names_rec:
            log_prob = self.observation_models[name].get_log_prob(h_t, s_t, o_t[name])
            observation_log_prob[name] = log_prob
        return observation_log_prob

    def get_mse(self, h_t, s_t, o_t):
        observation_mse = dict()
        for name in self.observation_names_rec:
            mse = self.observation_models[name].get_mse(h_t, s_t, o_t[name])
            observation_mse[name] = mse
        return observation_mse

    def get_pred_value(self, h_t: torch.Tensor, s_t: torch.Tensor, key):
        return self.observation_models[key](h_t, s_t)

    def get_pred_key(self, h_t: torch.Tensor, s_t: torch.Tensor, key):
        return self.get_pred_value(h_t, s_t, key)

    def get_state_dict(self): 
        observation_model_state_dict = dict()
        for name in self.observation_models.keys():
            observation_model_state_dict[name] = self.observation_models[name].state_dict()
        return observation_model_state_dict

    def _load_state_dict(self, state_dict):
        for name in self.observation_names_rec:
            self.observation_models[name].load_state_dict(state_dict[name])

    def get_model_params(self):
        model_params = []
        for model in self.observation_models.values():
            model_params += list(model.parameters())
        return model_params

    def eval(self):
        for name in self.observation_models.keys():
            self.observation_models[name].eval()

    def train(self):
        for name in self.observation_models.keys():
            self.observation_models[name].train()

    