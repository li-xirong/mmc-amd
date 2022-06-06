from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import cv2
import random
import torchvision.transforms as transforms


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0      
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]
    #print(image_numpy.astype(imtype))
    return image_numpy.astype(imtype)

def tensor2gry(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
        image_numpy = image_numpy[:,:,0]
    #print(image_numpy.astype(imtype))
    img_gray = cv2.cvtColor(image_numpy,cv2.COLOR_RGB2GRAY)
    return img_gray.astype(imtype)

# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()    
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 35: # cityscape
        cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)], 
                     dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

    import random


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))

    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params, method=Image.BICUBIC, normalize=True):
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.loadSize, method)))

    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
    if not opt.isTrain and opt.test_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, True)))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def weighted_sigmoid(arr, w=1):
    return 1. / (1 + np.exp(-arr * w))

def load_cam(path, opt):
    cams = np.load(path)
    cams = weighted_sigmoid(cams, 0.1) * 255
    cams = np.transpose(cams, (1,2,0))
    cams = Image.fromarray(cams.astype('uint8')).convert('RGB')
    
    params = get_params(opt, cams.size)
    transformer = get_transform(opt, params)
    data_tensor = transformer(cams)
    
    return torch.unsqueeze(data_tensor, dim=0), cams
