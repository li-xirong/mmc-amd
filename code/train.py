import os
import sys
import copy
import time
import torch
import argparse
import importlib
import numpy as np
from models import save_model, load_two_stream_model, load_single_stream_model
from data.DataLoaders import MultiDataLoader, FundusDataLoader, OctDataLoader
from utils import AverageMeter, load_config, splitprint, runid_checker, predict_dataloader, batch_eval


def parse_args():
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument("--train_collection", type=str, required=True, help="train collection path")
    parser.add_argument("--syn_collection", default="", type=str, help="syn collection path")
    parser.add_argument("--val_collection", type=str, required=True, help="val collection path")
    parser.add_argument("--print_freq", default=20, type=int, help="print frequent (default: 20)")
    parser.add_argument("--model_configs", type=str, required=True, help="filename of the model configuration file.")
    parser.add_argument("--run_id", default=0, type=int, help="run_id (default: 0)")
    parser.add_argument("--device", default=0, type=str, help="cuda:n or cpu (default: 0)")
    parser.add_argument("--num_workers", default=0, type=int, help="number of threads for sampling. (default: 0)")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing files")
    parser.add_argument("--checkpoint", default=None, type=str, help="checkpoint path")
    args = parser.parse_args()
    return args


def validate(model, val_loader, selected_metric, device, cls_num, net_name="mm-model", verbose=False):
    if verbose:
        print("-" * 45 + "validation" + "-" * 45)
    predicts, scores, expects = predict_dataloader(model, val_loader, device, net_name, if_test=False)
    results = batch_eval(predicts, expects, cls_num, verbose)
    return results["overall"][selected_metric]


def adjust_learning_rate(optimizer, optim_params):
    optim_params['lr'] *= 0.5
    print('learning rate:', optim_params['lr'])
    for param_group in optimizer.param_groups:
        param_group['lr'] = optim_params['lr']
    if optim_params['lr'] < optim_params['lr_min']:
        return True
    else:
        return False


def model_structure(net_name):
    print("load model {}".format(net_name))
    if net_name == "mm-model":
        return load_two_stream_model
    elif net_name == 'cfp-model' or net_name == 'oct-model':
        return load_single_stream_model
    else:
        print("{} is not support.").format(net_name)
        return None


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


def main(opts):
    # load model configs
    configs = load_config(opts.model_configs)

    # check that the save path is available
    if not runid_checker(opts, configs.if_syn):
        return
    splitprint()
    # cuda number
    device = torch.device("cuda: {}".format(opts.device)
                          if (torch.cuda.is_available() and opts.device != "cpu") else "cpu")

    # get trainset and valset dataloaders for training
    data_initializer = select_dataloader(configs.modality)(opts, configs)
    train_loader, val_loader = data_initializer.get_training_dataloader()

    # load model
    splitprint()
    # checkpoint = configs.checkpoint if len(configs.checkpoint) else None
    checkpoint = opts.checkpoint
    model = model_structure(configs.net_name)(configs, device, checkpoint)

    criterion = torch.nn.CrossEntropyLoss()
    if configs.train_params["optimizer"] == "sgd":
        optimizer_params = configs.train_params["sgd"]
        optimizer = torch.optim.SGD(model.parameters(), lr=optimizer_params["lr"],
                                    momentum=optimizer_params["momentum"],
                                    weight_decay=optimizer_params["weight_decay"])

    tolerance = 0
    best_epoch = 0
    best_metric = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(configs.train_params["max_epoch"]):

        if epoch == 0:
            print("eval initial state.")
            model.eval()
            _ = validate(model, val_loader, configs.train_params["best_metric"],
                         device, configs.cls_num, configs.net_name, not configs.if_syn)

        splitprint()
        print('Epoch {}/{}'.format(epoch + 1, configs.train_params["max_epoch"]))

        # train step
        model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()

        for i, (inputs, labels_onehot, imagenames) in enumerate(train_loader):
            data_time.update(time.time() - end)

            labels = torch.from_numpy(np.argmax(labels_onehot.cpu().numpy(), axis=1).astype(np.int64))

            optimizer.zero_grad()
            if configs.net_name == "mm-model":
                outputs = model(inputs[0].to(device), inputs[1].to(device))
                inputs_size = inputs[0].size(0)
            elif configs.net_name in ["cfp-model", "oct-model"]:
                outputs = model(inputs.to(device))
                inputs_size = inputs.size(0)
            else:
                print("model {} is not support.").format(configs.net_name)
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            losses.update(loss, inputs_size)
            batch_time.update(time.time() - end)
            end = time.time()

            if i % opts.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, i, len(train_loader),
                    batch_time=batch_time, data_time=data_time, loss=losses))

        # val step
        model.eval()
        cur_metric = validate(model, val_loader, configs.train_params["best_metric"],
                         device, configs.cls_num, configs.net_name, not configs.if_syn)

        if cur_metric > best_metric:
            best_epoch = epoch
            best_metric = cur_metric
            best_model_wts = copy.deepcopy(model.state_dict())
            print("save the better weights, metric value: {}".format(best_metric))
            save_model(best_model_wts, opts, epoch, best_metric, if_syn=configs.if_syn)
            tolerance = 0
        elif epoch > optimizer_params["lr_decay_start"]:
            tolerance += 1
            if tolerance % optimizer_params["tolerance_iter_num"] == 0:
                if_stop = adjust_learning_rate(optimizer, optimizer_params)
                print("best:", best_metric)
                if if_stop:
                    break

    save_model(best_model_wts, opts, epoch, best_metric, True, best_epoch, if_syn=configs.if_syn)
    print("finish. metric value: {}".format(best_metric))


if __name__ == "__main__":
    opts = parse_args()
    main(opts)
