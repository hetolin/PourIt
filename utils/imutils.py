import torch
import torchvision
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import random
import logging
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_logger(filename='test.log'):
    ## setup logger
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)

def encode_cmap(label):
    cmap = colormap()
    return cmap[label.astype(np.int16),:]

def denormalize_img(imgs=None, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    _imgs = torch.zeros_like(imgs)
    _imgs[:,0,:,:] = imgs[:,0,:,:] * std[0] + mean[0]
    _imgs[:,1,:,:] = imgs[:,1,:,:] * std[1] + mean[1]
    _imgs[:,2,:,:] = imgs[:,2,:,:] * std[2] + mean[2]
    _imgs = _imgs.type(torch.uint8)

    return _imgs

def tensorboard_image(imgs=None, cam=None,):
    ## images
    _imgs = denormalize_img(imgs=imgs)
    grid_imgs = torchvision.utils.make_grid(tensor=_imgs, nrow=2)

    if cam is None:
        return grid_imgs, None

    cam = F.interpolate(cam, size=_imgs.shape[2:], mode='bilinear', align_corners=False)
    cam = cam.cpu()
    cam_max = cam.max(dim=1)[0]
    cam_heatmap = plt.get_cmap('plasma')(cam_max.numpy())[:,:,:,0:3]*255
    cam_cmap = torch.from_numpy(cam_heatmap).permute([0, 3, 1, 2])
    cam_img = cam_cmap*0.5 + _imgs[:,:3,...].cpu()*0.5
    grid_cam = torchvision.utils.make_grid(tensor=cam_img.type(torch.uint8), nrow=2)

    return grid_imgs, grid_cam


def tensorboard_fmap(fmaps, n_row=2):
    #(B,512,32,32)
    fmaps_resized = F.interpolate(fmaps, size=[224, 224], mode='bilinear', align_corners=False) #(B, 64, 224,224)
    fmap_cmaps = []
    for i in range(8):
        fmap = fmaps_resized[:,i,...] #(B,224,224)
        fmap = fmap.cpu()
        fmap_heatmap = plt.get_cmap('viridis')(fmap.detach().numpy())[:,:,:,0:3]*255
        fmap_cmap = torch.from_numpy(fmap_heatmap).permute([0, 3, 1, 2])
        fmap_cmaps.append(fmap_cmap)

    fmap_img = torch.cat(fmap_cmaps, dim=0)
    grid_fmap = torchvision.utils.make_grid(tensor=fmap_img.type(torch.uint8), nrow=n_row)

    return grid_fmap

def tensorboard_label(labels=None):
    ## labels (B,H,W)
    # labels_cmap = encode_cmap(np.squeeze(labels))
    labels_cmap = encode_cmap(labels) #(B,H,W)
    labels_cmap = torch.from_numpy(labels_cmap).permute([0, 3, 1, 2])
    grid_labels = torchvision.utils.make_grid(tensor=labels_cmap, nrow=2)

    return grid_labels

def tensorboard_label_normalized(labels=None):
    ## labels
    grid_labels = torchvision.utils.make_grid(tensor=labels*255, nrow=2)

    return grid_labels


def colormap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap