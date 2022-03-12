import os
import copy
import torch
import shutil
import importlib
import numpy as np
import numpy as np
from metrics import accuracy_score, confusion_matrix, sensitivity_score, specificity_score, f1_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_config(config_filename):
    config_path = "configs.{}".format(config_filename.split('.')[0])
    module = importlib.import_module(config_path)
    return module.Config()


def splitprint():
    print("#"*100)


def runid_checker(opts, if_syn=False):
    rootpath = opts.train_collection
    if if_syn:
        rootpath = opts.syn_collection
    valset_name = os.path.split(opts.val_collection)[-1]
    config_filename = opts.model_configs
    run_id = opts.run_id
    target_path = os.path.join(rootpath, "models", valset_name, config_filename, "run_" + str(run_id))
    if os.path.exists(target_path):
        if opts.overwrite:
            shutil.rmtree(target_path)
        else:
            print("'{}' exists!".format(target_path))
            return False
    os.makedirs(target_path)
    print("checkpoints are saved in '{}'".format(target_path))
    return True


def predict_dataloader(model, loader, device, net_name="mm-model", if_test=False):
    model.eval()
    predicts = []
    scores = []
    expects = []
    imagename_list = []
    for i, (inputs, labels_onehot, imagenames) in enumerate(loader):
        if if_test:
            label = None
        else:
            label = torch.from_numpy(np.argmax(labels_onehot.cpu().numpy(), axis=1).astype(np.int64))
        with torch.no_grad():
            if net_name == "mm-model":
                outputs = model(inputs[0].to(device), inputs[1].to(device))
            elif net_name in ["cfp-model", "oct-model"]:
                outputs = model(inputs.to(device))
        output = np.squeeze(torch.softmax(outputs, dim=1).cpu().numpy())
        predicts.append(np.argmax(output))
        scores.append(np.max(output))
        expects.append(label.numpy())
        imagename_list.append(np.squeeze(imagenames))
    if if_test:
        return predicts, scores, imagename_list
    return predicts, scores, expects


def batch_eval(predicts, expects, cls_num=4, verbose=False):
    def multi_to_binary(Y, pos_cls_idx):
        Y_cls = copy.deepcopy(np.array(Y))
        pos_idx = np.where(np.array(Y) == pos_cls_idx)
        neg_idx = np.where(np.array(Y) != pos_cls_idx)
        Y_cls[neg_idx] = 0
        Y_cls[pos_idx] = 1
        return Y_cls

    metrics = {"overall": {}, "normal": {}, "dry": {}, "pcv": {}, "wet": {}}
    cls_list = ["normal", "dry", "pcv", "wet"]
    metrics["overall"]["accuracy"] = accuracy_score(expects, predicts)
    metrics["overall"]["confusion_matrix"] = confusion_matrix(expects, predicts)
    # metrics per class
    for cls_idx in range(cls_num):
        cls_name = cls_list[cls_idx]
        predicts_cls = multi_to_binary(predicts, cls_idx)
        expects_cls = multi_to_binary(expects, cls_idx)
        sen = sensitivity_score(expects_cls, predicts_cls)
        spe = specificity_score(expects_cls, predicts_cls)
        f1 = f1_score(sen, spe)
        metrics[cls_name]["sensitivity"] = sen
        metrics[cls_name]["specificity"] = spe
        metrics[cls_name]["f1_score"] = f1
    metrics["overall"]["f1_score"] = np.average(
        [metrics[cls_name]["f1_score"] for cls_name in cls_list])

    if verbose:
        print(" Class\tSen.\tSpe.\tF1score\n",
              "normal\t{sen:.4f}\t{spe:.4f}\t{f1:.4f}\n".format(
                  sen=metrics["normal"]["sensitivity"],
                  spe=metrics["normal"]["specificity"],
                  f1=metrics["normal"]["f1_score"]),
              "dAMD\t{sen:.4f}\t{spe:.4f}\t{f1:.4f}\n".format(
                  sen=metrics["dry"]["sensitivity"],
                  spe=metrics["dry"]["specificity"],
                  f1=metrics["dry"]["f1_score"]),
              "PCV\t{sen:.4f}\t{spe:.4f}\t{f1:.4f}\n".format(
                  sen=metrics["pcv"]["sensitivity"],
                  spe=metrics["pcv"]["specificity"],
                  f1=metrics["pcv"]["f1_score"]),
              "wAMD\t{sen:.4f}\t{spe:.4f}\t{f1:.4f}\n".format(
                  sen=metrics["wet"]["sensitivity"],
                  spe=metrics["wet"]["specificity"],
                  f1=metrics["wet"]["f1_score"]),
              "-"*99+"\n",
              "overall\tf1_score:{f1:.4f}\taccuracy:{acc:.4f}\n".format(
                  f1=metrics["overall"]["f1_score"],
                  acc=metrics["overall"]["accuracy"]),
              "confusion matrix:\n {}".format(metrics["overall"]["confusion_matrix"]))
    return metrics
