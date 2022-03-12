import os
import sys
import torch
import argparse
import importlib
import numpy as np
from models import load_single_stream_model
from data.DataLoaders import FundusDataLoader, OctDataLoader
from utils import load_config, splitprint, predict_dataloader
import shutil
import cv2 as cv


def temp_test(a, b):
    from metrics import accuracy_score
    print(accuracy_score(a, b))


def parse_args():
    parser = argparse.ArgumentParser(description="generat_cam")
    parser.add_argument("--collection", type=str, required=True, help="collection path")
    parser.add_argument("--model_configs", type=str, required=True, help="filename of the model configuration file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="an existing checkpoint.")
    parser.add_argument("--device", default=0, type=str, help="cuda:n or cpu (default: 0)")
    parser.add_argument("--num_workers", default=0, type=int, help="number of threads for sampling. (default: 0)")
    args = parser.parse_args()
    return args


def select_dataloader(modality):
    print("initialize dataloader for {}".format(modality))
    if modality == "mm":
        return MultiDataLoader
    elif modality == "cfp":
        return FundusDataLoader
    elif modality == "oct":
        return OctDataLoader
    else:
        print("{} is not support.").format(modality)
        return None


class Drawer(object):
    def __init__(self, model, configs, device=None):
        self.model = model
        if device:
            self.model = self.model.to(device)
        self.weights = self.get_fc_weights()
        self.weights = self.weights.cpu().detach().numpy()
        self.weights = self.weights[:, :, np.newaxis, np.newaxis]
        self.device = device
        self.configs = configs

    def get_fc_weights(self):
        return self.model.fc.weight.data

    def get_heat_map(
            self, inputs, labels_onehot, imgnames, collection, modality):
        if modality == "cfp":
            _dir = "cfp-clahe-448x448"
        elif modality == "oct":
            _dir = "oct-median3x3-448x448"
        else:
            print("{} is not support.").format(modality)
        label = np.argmax(np.squeeze(labels_onehot.numpy()))
        img_path = os.path.join(collection, "ImageData", _dir, imgnames)

        if self.device:
            inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs, feature_maps = self.model(inputs)

        class_num = len(outputs[0])
        pred = np.argmax(outputs[0].cpu().numpy())

        feature_maps = feature_maps[0].cpu().numpy()
        fmc, fmh, fmw = feature_maps.shape

        cls_cam = np.zeros((class_num, fmh, fmw))

        for index in range(class_num):
            cam_map = feature_maps * self.weights[index]
            cam_map = cam_map.sum(axis=0)
            cls_cam[index] = cam_map

        return label, pred, cls_cam


def main(opts):
    # cuda number
    device = torch.device("cuda: {}".format(opts.device)
                          if (torch.cuda.is_available() and opts.device != "cpu") else "cpu")

    # load model configs
    configs = load_config(opts.model_configs)
    configs.heatmap = True

    # get dataloader for cam generator
    data_initializer = select_dataloader(configs.modality)(opts, configs)
    data_loader = data_initializer.get_camgenerating_dataloader()

    # load model
    splitprint()
    print("load model {}".format(configs.net_name))
    checkpoint = opts.checkpoint
    model = load_single_stream_model(configs, device, opts.checkpoint)
    model.eval()
    splitprint()

    # cam generator
    heatmap_drawer = Drawer(model, configs, device)

    checkpoint_root, checkpoint_name = os.path.split(checkpoint)
    dstPath = os.path.join(checkpoint_root, "cam", checkpoint_name)
    if os.path.exists(dstPath):
        shutil.rmtree(dstPath)
    os.makedirs(dstPath)
    print("cam imgs are saved in {}.".format(dstPath))

    for i, (inputs, labels_onehot, imagenames) in enumerate(data_loader):
        if np.argmax(labels_onehot) == 0:
            continue
        label, pred, cls_cam = heatmap_drawer.get_heat_map(
            inputs, labels_onehot, np.squeeze(imagenames).item(), opts.collection, configs.modality)
        np.save(os.path.join(dstPath, np.squeeze(imagenames).item() + '.npy'), cls_cam)


if __name__ == "__main__":
    opts = parse_args()
    main(opts)
    print("finish.")
