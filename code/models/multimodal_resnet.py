# coding: utf-8
import torch
import numpy as np
import torch.nn as nn

class CombineNet_linear_concatenate(nn.Module):
    def __init__(
            self, model1, model2, fc_ins, num_classes=4, heatmap=False):

        super(CombineNet_linear_concatenate, self).__init__()

        self.heatmap = heatmap
        self.model1 = model1
        self.model2 = model2
        self.fc_cat = nn.Linear(fc_ins, num_classes)
        self.gap = nn.AvgPool2d(14, stride=1)

    def predict_per_instance(self, inputs):
        with torch.no_grad():
            output, fm_f, fm_o = self.forward(inputs[0], inputs[1])
        output = np.squeeze(torch.softmax(output, dim=1).cpu().numpy())
        pred = np.argmax(output)
        score = np.max(output)
        return pred, score

    def forward(self, x1, x2):
        if self.heatmap:
            x1, feature_map1 = self.model1(x1)
            x2, feature_map2 = self.model2(x2)
            x = torch.cat([x1, x2], 1)
            x = x.view(x.size(0), -1)
            x = self.fc_cat(x)
            return x, feature_map1, feature_map2
        else:
            x1 = self.model1(x1)
            x2 = self.model2(x2)
            x = torch.cat([x1, x2], 1)
            x = x.view(x.size(0), -1)
            x = self.fc_cat(x)
            return x



