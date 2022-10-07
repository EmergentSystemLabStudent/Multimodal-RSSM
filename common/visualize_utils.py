import torch

from sklearn.decomposition import PCA

def numpy2tensor(data, dtype=torch.float32):
    if torch.is_tensor(data):
        return data
    else:
        return torch.tensor(data, dtype=dtype)

def tensor2numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().detach().numpy().copy()
    else:
        return tensor

def image_postprocess(image, bit_depth=5):
    from common.env import postprocess_observation
    image = postprocess_observation(image, bit_depth=bit_depth).transpose(1, 2, 0)
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
    if not torch.is_tensor(feat):
        feat = torch.vstack(feat)
    feat = tensor2numpy(feat)
    feat_flat = flat(feat)
    pca.fit(feat_flat)
    return pca

