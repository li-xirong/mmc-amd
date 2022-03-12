import os
import numpy as np
import copy
from PIL import Image


def visualize_cams(cams_4channels):
    cams_4channels = weighted_sigmoid(cams_4channels, 0.1) * 255
    # remove normal channel
    cams_3channels = cams_4channels[1:]
    camsArray = np.transpose(cams_3channels, (1, 2, 0))
    camsImg = Image.fromarray(camsArray.astype('uint8')).convert('RGB')
    camsImg = camsImg.resize((448, 448), Image.BICUBIC)
    return camsImg


def weighted_sigmoid(arr, w=1):
    return 1. / (1 + np.exp(-arr * w))


def flip(cam):
    cam_flip = np.zeros_like(cam)
    for column in range(cam_flip.shape[-1]):
        cam_flip[:, :, column] = cam[:, :, cam_flip.shape[-1] - column - 1]
    return cam_flip


def move(cam, cls, pathsize=5, verticle=True,
         startRow=4, endRow=4, startColumn=4, endColumn=4):
    camGT = cam[cls]
    maxScore = -100
    maxScoreIndex = (-1, -1)
    for i in range(startRow - 1, cam.shape[1] - endRow + 1):
        for j in range(startColumn - 1, cam.shape[1] - endColumn + 1):
            patchGT = camGT[i - 2:i + 3, j - 2:j + 3]
            score = np.sum(patchGT)
            if score > maxScore:
                maxScore = score
                maxScoreIndex = (i, j)

    i, j = maxScoreIndex
    randi = np.random.randint(startRow - 1, cam.shape[1] - endRow + 1)
    randj = np.random.randint(startColumn - 1, cam.shape[1] - endColumn + 1)
    if not verticle:
        randi = i

    # exchange position.
    maxScorePatch = copy.deepcopy(cam[:, i - 2:i + 3, j - 2:j + 3])
    tempPatch = copy.deepcopy(cam[:, randi - 2:randi + 3, randj - 2:randj + 3])
    cam_move = copy.deepcopy(cam)
    cam_move[:, i - 2:i + 3, j - 2:j + 3] = tempPatch
    cam_move[:, randi - 2:randi + 3, randj - 2:randj + 3] = maxScorePatch

    return cam_move