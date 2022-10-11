import torch

from sklearn.decomposition import PCA

from utils.processing.image_processing import reverse_normalized_image

def np2tensor(data, dtype=torch.float32):
    if torch.is_tensor(data):
        return data
    else:
        return torch.tensor(data, dtype=dtype)

def tensor2np(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().detach().numpy().copy()
    else:
        return tensor

def reverse_image_observation(image, bit_depth=5):    
    image = reverse_normalized_image(image, bit_depth=bit_depth).transpose(1, 2, 0)
    return image

def get_xyz(feat):
    feat_flat = flat(feat)
    return feat_flat[:,0], feat_flat[:,1], feat_flat[:,2]

def flat(feat):
    feat_size = feat.shape[-1]
    return feat.reshape(-1, feat_size)

#-------------------------------------------------------------------------#
#-------------------------------------------------------------------------#

def get_pca_model(feat, n_components=3):
    pca = PCA(n_components=n_components)
    feat = tensor2np(feat)
    feat_flat = flat(feat)
    print(feat_flat.shape)
    pca.fit(feat_flat)
    return pca

