import os
import numpy as np
import copy
import argparse
from PIL import Image
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument("--cams_dir", type=str, required=True, help="initial cams path")
    parser.add_argument("--dst_dir", type=str, required=True, help="manipulated cams path")
    parser.add_argument("--modality", type=str, required=True, help="cfp / oct")
    args = parser.parse_args()
    return args


def main(opts):
    # manipulate cam
    CamFolder = opts.cams_dir
    dstFolder = opts.dst_dir
    if not os.path.exists(dstFolder):
        os.mkdir(dstFolder)

    assert opts.modality in ["cfp", "oct"], "modality should be cfp or oct."

    verticle = True if opts.modality == "cfp" else False

    categorization = ['h', 'd', 'p', 'w']
    for name in os.listdir(CamFolder):
        cls = categorization.index(name[2])
        cam_ori = np.load(os.path.join(CamFolder, name))
        np.save(os.path.join(dstFolder, name[:-8] + "-ori.npy"), cam_ori)
        cam_flip = flip(cam_ori)
        np.save(os.path.join(dstFolder, name[:-8] + "-flr.npy"), cam_flip)
        cam_move = move(cam_ori, cls, verticle=verticle)
        np.save(os.path.join(dstFolder, name[:-8] + "-trs.npy"), cam_move)


if __name__ == "__main__":
    opts = parse_args()
    main(opts)
