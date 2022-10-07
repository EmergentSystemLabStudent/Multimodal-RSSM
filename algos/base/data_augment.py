import torch
import numpy as np


def torch_cov(x, rowvar=False, bias=False, ddof=None, aweights=None, device=torch.device("cpu")):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float32, device=device)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w/w_sum)[:,None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()

@torch.jit.script
def calc_delta(p_eigen_vector, lambd_eigen_value, rand):
    delta = torch.matmul(p_eigen_vector, rand * lambd_eigen_value)
    delta = (delta * 255.0).unsqueeze(1).unsqueeze(1)
    return delta

def generate_delta_of_pca_color_augmentation(lambd_eigen_value, p_eigen_vector, rand, pca_scales, device=torch.device("cpu")):
    if rand == None:
        pca_idx = np.random.randint(0, len(pca_scales))
        scale = pca_scales[pca_idx]
        if scale > 0:
            rand = torch.tensor(np.random.randn(3) * scale, device=device, dtype=torch.float32)
        else:
            rand = torch.zeros(3, device=device, dtype=torch.float32)
    
    delta = calc_delta(p_eigen_vector, lambd_eigen_value, rand)
            
    return delta, rand

def generate_gaussian_noise(image, scale=0.1):
    if scale > 0:
        noise = torch.randn(*image.shape, device=image.device) * scale  # mean=0,scale=1
        noise = noise * 255.0
    else:
        noise = torch.zeros(*image.shape, device=image.device)
    return noise



def get_dx(idx):
    num = 0
    count = 0
    next_num = 1
    for i in range(idx):
        if not num == next_num:
            if next_num > 0:
                num += 1
            else:
                num -= 1
        else:
            if next_num > 0:
                if count < num*2-1:
                    count +=1
                else:
                    next_num = -next_num
                    count = 0
                    num -= 1
            else:
                if count < (-num)*2+1-1:
                    count += 1
                else:
                    next_num = -next_num + 1
                    count = 0
                    num += 1
    return -num

def get_dy(idx):
    num = 0
    count = 0
    next_num = 0
    for i in range(idx):
        if not num == next_num:
            if next_num > 0:
                num += 1
            else:
                num -= 1
        else:
            if next_num >= 0:
                if count < (num+1)*2-1:
                    count +=1
                else:
                    next_num = -next_num-1
                    count = 0
                    num -= 1
            else:
                if count < (-num-1)*2+2:
                    count += 1
                else:
                    next_num = -next_num
                    count = 0
                    num += 1
    return num

# def idx_to_idx_w_h(idx):
#     # base position of crapping
#     # | 0| 1| 4| 9|
#     # | 2| 3| 5|10|
#     # | 6| 7| 8|11|
#     # |12|13|14|15|
#     k = int(np.sqrt(idx))
#     if idx < k * (k + 1):
#         idx_w = idx // k
#         idx_h = idx % k
#     else:
#         idx_w = idx % (k + 1)
#         idx_h = idx // (k + 1)
#     return idx_w, idx_h

def idx_to_idx_w_h(idx, image_shape, size, dh_base, dw_base):
    # base position of crapping
    # |12|13|14|15|
    # |11| 2| 3| 4|
    # |10| 1| 0| 5|
    # | 9| 8| 7| 6|
    dx = get_dx(idx)
    dy = get_dy(idx)
    xy_center = (np.array(image_shape[-2:]) - np.array(size))/(dh_base, dw_base)
    (x,y) = np.floor(xy_center/2)
    idx_w = int(x+dx)
    idx_h = int(y+dy)

    return idx_w, idx_h


def crop_image(image, idx=0, size=(64, 64), dh_base=2, dw_base=2):
    # print(image.shape)
    # idx_w, idx_h = idx_to_idx_w_h(idx)
    idx_w, idx_h = idx_to_idx_w_h(idx, image.shape[-2:], size, dh_base, dw_base)

    h, w = image.shape[-2:]
    dh = dh_base * idx_h
    dw = dw_base * idx_w
    if len(image.shape) == 3:
        ims = image[:, dh:size[0] + dh, dw:size[1] + dw]
    elif len(image.shape) == 4:
        ims = image[:, :, dh:size[0] + dh, dw:size[1] + dw]
    else:
        ims = image[:, :, :, dh:size[0] + dh, dw:size[1] + dw]
    return ims


def crop_image_data(data, n_crop=None, dh_base=None, dw_base=None):
    if not n_crop is None:
        k = int(np.sqrt(n_crop-1))
        for name in data.keys():
            if "image" in name:
                if ("_256" in name) or ("high_resolution" in name):
                    data[name] = crop_image(
                        data[name], idx=0, size=(
                            256+k*dh_base, 256+k*dw_base), dh_base=dh_base, dw_base=dw_base)
                elif ("_128" in name):
                    data[name] = crop_image(
                        data[name], idx=0, size=(
                            128+k*dh_base, 128+k*dw_base), dh_base=dh_base, dw_base=dw_base)
                else:
                    data[name] = crop_image(
                        data[name], idx=0, size=(
                            64+k*dh_base, 64+k*dw_base), dh_base=dh_base, dw_base=dw_base)
    return data


def augment_image_data(image, name, n_crop=None, dh_base=None, dw_base=None, noise_scales=None, pca_rand=None, lambd_eigen_value=None, p_eigen_vector=None, pca_scales=None, crop_idx=None):

    # crop
    if not n_crop is None:
        if crop_idx is None:
            crop_idx = np.random.randint(0, n_crop)
        if ("_256" in name) or ("high_resolution" in name):
            image = crop_image(
                image, idx=crop_idx, size=(
                    256, 256), dh_base=dh_base, dw_base=dw_base)
        elif ("_128" in name):
            image = crop_image(
                image, idx=crop_idx, size=(
                    128, 128), dh_base=dh_base, dw_base=dw_base)
        else:
            image = crop_image(
                image, idx=crop_idx, size=(
                    64, 64), dh_base=dh_base, dw_base=dw_base)
    
    # noise
    if not noise_scales is None:
        noise_idx = np.random.randint(0, len(noise_scales))
        noise = generate_gaussian_noise(image, scale=noise_scales[noise_idx])
    else:
        noise = torch.zeros_like(image)
    
    # PCA
    if not pca_scales is None:
        delta_pca, pca_rand = generate_delta_of_pca_color_augmentation(lambd_eigen_value, p_eigen_vector, pca_rand, pca_scales, device=image.device)
    else:
        delta_pca = torch.zeros_like(image)
    image = torch.clip(image + delta_pca + noise, 0, 255)
    
    return image, pca_rand