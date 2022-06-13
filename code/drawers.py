import copy
import torch
import cv2 as cv
import numpy as np
from PIL import Image


class CAMDrawer(object):
    def __init__(self, model, device=None):
        self.model = model
        if device:
            self.model = self.model.to(device)
        self.weights = self.get_fc_weights()
        self.weights = self.weights.cpu().detach().numpy()
        self.weights = self.weights[:, :, np.newaxis, np.newaxis]
        self.device = device

    def get_fc_weights(self):
        return self.model.fc.weight.data

    def get_cam(self, inputs, save_path="", rm_normal=True):
        if self.device:
            inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs, feature_maps = self.model(inputs)

        class_num = len(outputs[0])
        feature_maps = feature_maps[0].cpu().numpy()
        fmc, fmh, fmw = feature_maps.shape

        cams = np.zeros((class_num, fmh, fmw))

        for index in range(class_num):
            cam_map = feature_maps * self.weights[index]
            cam_map = cam_map.sum(axis=0)
            cams[index] = cam_map
        if rm_normal:
            cams = cams[1:]
        if save_path:
            # print("CAM is saved in {}.".format(save_path))
            np.save(save_path, cams)

        return cams

    @staticmethod
    def flip(cam, save_path=None):
        cam_flip = np.zeros_like(cam)
        for column in range(cam_flip.shape[-1]):
            cam_flip[:, :, column] = cam[:, :, cam_flip.shape[-1] - column - 1]

        if save_path:
            # print("CAM is saved in {}.".format(save_path))
            np.save(save_path, cam_flip)
        return cam_flip

    @staticmethod
    def move(cam, reference_channel, save_path=None,
             verticle=True, startRow=4, endRow=4, startColumn=4, endColumn=4):
        camGT = cam[reference_channel]
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

        if save_path:
            # print("CAM is saved in {}.".format(save_path))
            np.save(save_path, cam_move)

        return cam_move

    @staticmethod
    def visualize(raw_img, cams, visual_size=(448, 448)):
        if type(cams) == Image.Image:
            cams_rgb = cams

        else:
            assert cams.shape[0] == 3, "only 3-channel cams can be shown in RGB format."
            cams = weighted_sigmoid(cams, 0.1) * 255
            cams_rgb = np.transpose(cams, (1, 2, 0))
            cams_rgb = Image.fromarray(cams_rgb.astype('uint8')).convert('RGB')

        cams_rgb = cams_rgb.resize(visual_size, Image.BICUBIC)

        raw = Image.fromarray(cv.cvtColor(raw_img, cv.COLOR_BGR2RGB)).resize(visual_size, Image.BICUBIC)
        splicing = Image.new('RGB', (visual_size[0] * 2, visual_size[1]), (255, 255, 255))
        splicing.paste(raw, (0, 0))
        splicing.paste(cams_rgb, (visual_size[0], 0))
        return splicing

    @staticmethod
    def loop_visualize(img_and_cams_list, visual_size=(448, 448), space=10):
        row_num = len(img_and_cams_list)
        splicing = Image.new('RGB', ((visual_size[0]+space)*2, (visual_size[1]+space)*row_num), (255, 255, 255))
        
        for row_idx, (raw_img, cams) in enumerate(img_and_cams_list):
            if type(cams) == Image.Image:
                cams_rgb = cams

            else:
                assert cams.shape[0] == 3, "only 3-channel cams can be shown in RGB format."
                cams = weighted_sigmoid(cams, 0.1) * 255
                cams_rgb = np.transpose(cams, (1, 2, 0))
                cams_rgb = Image.fromarray(cams_rgb.astype('uint8')).convert('RGB')

            cams_rgb = cams_rgb.resize(visual_size, Image.BICUBIC)
            raw = Image.fromarray(cv.cvtColor(raw_img, cv.COLOR_BGR2RGB)).resize(visual_size, Image.BICUBIC)
            
            splicing.paste(raw, (0, (visual_size[1]+space)*row_idx))
            splicing.paste(cams_rgb, (visual_size[0]+space, (visual_size[1]+space)*row_idx))
            
        return splicing

    @staticmethod
    def sequence_visualize(raw_img, cams_list, visual_size=(448, 448), bound=10):
        l = len(cams_list)
        splicing = Image.new('RGB', (visual_size[0]*(l+1)+bound*(l+1), visual_size[1]), (255, 255, 255))
        raw = Image.fromarray(cv.cvtColor(raw_img, cv.COLOR_BGR2RGB)).resize(visual_size, Image.BICUBIC)
        splicing.paste(raw, (0, 0))
        for i, cams in enumerate(cams_list):
            cams = weighted_sigmoid(cams, 0.1) * 255
            cams_rgb = np.transpose(cams, (1, 2, 0))
            cams_rgb = Image.fromarray(cams_rgb.astype('uint8')).convert('RGB')
            cams_rgb = cams_rgb.resize(visual_size, Image.BICUBIC)
            splicing.paste(cams_rgb, (visual_size[0]*(i+1) + bound*(i+1), 0))
        return splicing


class FusionDrawer(object):
    def __init__(self, model, device=None):
        self.model = model
        if device:
            self.model = self.model.to(device)
        self.weights = self.get_fc_weights()
        self.weights = self.weights.cpu().detach().numpy()
        self.weights = self.weights[:, :, np.newaxis, np.newaxis]

        self.weights1 = self.weights[:, :512, :, :]
        self.weights2 = self.weights[:, 512:, :, :]

        self.device = device

    def get_fc_weights(self):
        return self.model.fc_cat.weight.data

    @staticmethod
    def process_feature_map(feature_maps, weights, size, thred=144):
        feature_maps = feature_maps[0].cpu().numpy()
        grey_map = (feature_maps * weights).sum(axis=0)
        grey_map = weighted_sigmoid(grey_map, 0.1) * 255
        grey_map[np.where(grey_map < thred)] = 0

        h, w = grey_map.shape
        heat_map = np.zeros((h, w, 3), np.uint8)
        for i in range(h):
            for j in range(w):
                heat_map[i, j] = grey2heat(grey_map[i, j])

        return cv.resize(heat_map, size)

    def get_fusion(self, inputs, raw_imgs):
        input1, input2 = inputs[0], inputs[1]
        raw_img_f, raw_img_o = raw_imgs[0], raw_imgs[1]

        gray_raw_f = cv.cvtColor(raw_img_f, cv.COLOR_BGR2GRAY)
        grey_raw_f = np.dstack((gray_raw_f, gray_raw_f, gray_raw_f))

        if self.device:
            input1 = input1.to(self.device)
            input2 = input2.to(self.device)

        with torch.no_grad():
            outputs, fundus_maps, oct_maps = self.model(input1, input2)
            pred = np.argmax(outputs[0].cpu().numpy())

        heat_map_fundus = self.process_feature_map(fundus_maps, self.weights1[pred], size=raw_img_f.shape[:2],
                                                   thred=144)
        heat_map_oct = self.process_feature_map(fundus_maps, self.weights2[pred], size=raw_img_o.shape[:2], thred=160)

        return fusion(grey_raw_f, heat_map_fundus[:, :, [2, 1, 0]], 0.7), \
               fusion(raw_img_o, heat_map_oct[:, :, [2, 1, 0]], 0.7)

    @staticmethod
    def visualize(fusion_img, raw_img, visual_size=(448, 448)):
        fusion_img = Image.fromarray(cv.cvtColor(fusion_img, cv.COLOR_BGR2RGB)).resize(visual_size, Image.BICUBIC)
        raw = Image.fromarray(cv.cvtColor(raw_img, cv.COLOR_BGR2RGB)).resize(visual_size, Image.BICUBIC)
        splicing = Image.new('RGB', (visual_size[0] * 2, visual_size[1]), (255, 255, 255))
        splicing.paste(raw, (0, 0))
        splicing.paste(fusion_img, (visual_size[0], 0))
        return splicing


def weighted_sigmoid(arr, w=1):
    return 1. / (1 + np.exp(-arr * w))


# convert a grey scale color into RGB
def grey2heat(grey):
    heat_stages_grey = np.array((0, 144, 160, 176, 192, 224, 256))
    heat_stages_color = np.array(
        ((0, 0, 0), (0, 0, 255), (165, 0, 255), (255, 0, 255), (255, 0, 0), (255, 255, 0), (255, 255, 255)))

    np.clip(grey, 0, 255)

    for i in range(1, len(heat_stages_grey)):
        if heat_stages_grey[i] > grey >= heat_stages_grey[i - 1]:
            weight = (grey - heat_stages_grey[i - 1]) / float(heat_stages_grey[i] - heat_stages_grey[i - 1])
            color = weight * heat_stages_color[i] + (1 - weight) * heat_stages_color[i - 1]
            break
    return color.astype(np.int)


def fusion(im1, im2, weight=0.5):
    f_im = im1 * weight + im2 * (1 - weight)
    f_im = f_im.astype(np.uint8)
    return f_im
