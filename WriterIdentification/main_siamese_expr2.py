import torch
import torch.nn as nn
import torch.utils.data as D
import os
import yaml
import numpy as np
import argparse
import json

from general_models import weights_init
from writer_identification import *
from datasets import *
from tqdm import tqdm
from statistics import mean
from torch.nn import functional as F



def batch_stats(preds_same, preds_diff, y_same, y_diff, return_measures=False, alpha=0.5, thresh=0.1):
    tp = 0
    tn = 0
    fp = 0
    fn = 0 

    preds = torch.cat([preds_same, preds_diff], dim=0)
    y = torch.cat([y_same, y_diff], dim=0)
    eucledian_distances = torch.sigmoid(preds).clone().detach()
    preds = [SiameseDataset2.same_style if d < thresh else SiameseDataset2.different_style for d in eucledian_distances]
    
    for pred, gt in zip(preds, y):
        if pred == gt.item() and gt.item() == SiameseDataset2.same_style:
            tp +=1
        elif pred == gt.item() and gt.item() == SiameseDataset2.different_style:
            tn += 1
        elif pred != gt.item() and gt.item() == SiameseDataset2.same_style:
            fp += 1
        elif pred != gt.item() and gt.item() == SiameseDataset2.different_style:
            fn += 1

    if return_measures:
        acc = (tp + tn) / (tp + tn + fp + fn + 1e-4)
        precision = tp / (tp + fp + 1e-4)
        recall = tp / (tp + fn + 1e-4)
        f_measure = alpha * precision + (1-alpha) * recall
        return acc, precision, recall, f_measure, tp, tn, fp, fn
    return tp, tn, fp, fn


def get_all_metrics(model, dataloader, device, alpha=0.5):
    model.eval()
    TP, TN, FP, FN = 0, 0, 0, 0

    for data in dataloader:

        target_x, same_style_x, different_style_x, \
            same_style_y, different_style_y = data

        target_x = target_x.to(device)
        same_style_x, different_style_x = same_style_x.to(device), different_style_x.to(device)
        same_style_y, different_style_y = same_style_y.to(device), different_style_y.to(device)

        pred_same = model(target_x, same_style_x)
        pred_diff = model(target_x, different_style_x)

        tp, tn, fp, fn = batch_stats(pred_same, pred_diff, same_style_y, different_style_y, alpha=alpha)
        TP += tp
        TN += tn
        FP += fp
        FN += fn

    acc = (TP + TN) / (TP + TN + FP + FN + 1e-4)
    precision = TP / (TP + FP + 1e-4)
    recall = TP / (TP + FN + 1e-4)
    f_measure = alpha * precision + (1-alpha) * recall
    return acc, precision, recall, f_measure, TP, TN, FP, FN


def init_epoch_stats():
    epoch_stats = dict()

    epoch_stats["train_batch_loss"] = list()

    epoch_stats["train_batch_accuracy"] = list()
    epoch_stats["train_batch_precision"] = list()
    epoch_stats["train_batch_recall"] = list()
    epoch_stats["train_batch_f_measure"] = list()
    epoch_stats["train_batch_tp"] = list()
    epoch_stats["train_batch_tn"] = list()
    epoch_stats["train_batch_fp"] = list()
    epoch_stats["train_batch_fn"] = list()

    epoch_stats["val_accuracy"] = list()
    epoch_stats["val_precision"] = list()
    epoch_stats["val_recall"] = list()
    epoch_stats["val_f_measure"] = list()
    epoch_stats["val_tp"] = list()
    epoch_stats["val_tn"] = list()
    epoch_stats["val_fp"] = list()
    epoch_stats["val_fn"] = list()

    return epoch_stats


def update_epoch_stats(epoch_stats, train_batch_loss, train_batch_accuracy, train_batch_precision,
    train_batch_recall, train_batch_f_measure, train_batch_tp, train_batch_tn, train_batch_fp, train_batch_fn,
    val_accuracy, val_precision, val_recall, val_f_measure, val_tp, val_tn, val_fp, val_fn):

    epoch_stats["train_batch_loss"].append(train_batch_loss)
    epoch_stats["train_batch_accuracy"].append(train_batch_accuracy)
    epoch_stats["train_batch_precision"].append(train_batch_precision)
    epoch_stats["train_batch_recall"].append(train_batch_recall)
    epoch_stats["train_batch_f_measure"].append(train_batch_f_measure)
    epoch_stats["train_batch_tp"].append(train_batch_tp)
    epoch_stats["train_batch_tn"].append(train_batch_tn)
    epoch_stats["train_batch_fp"].append(train_batch_fp)
    epoch_stats["train_batch_fn"].append(train_batch_fn)

    epoch_stats["val_accuracy"].append(val_accuracy)
    epoch_stats["val_precision"].append(val_precision)
    epoch_stats["val_recall"].append(val_recall)
    epoch_stats["val_f_measure"].append(val_f_measure)
    epoch_stats["val_tp"].append(val_tp)
    epoch_stats["val_tn"].append(val_tn)
    epoch_stats["val_fp"].append(val_fp)
    epoch_stats["val_fn"].append(val_fn)

    return epoch_stats


def save_metrics2file(epoch_metrics, metrics_file):
    metric_keys = list(epoch_metrics.keys())
    num_values = len(epoch_metrics[metric_keys[0]])

    for j in range(num_values):
        for i, k in enumerate(metric_keys):
            assert num_values == len(epoch_metrics[k]), "Invalid metrics"
            if i+1 == len(metric_keys):
                metrics_file.write(f"{epoch_metrics[k][j]}\n")
            else:
                metrics_file.write(f"{epoch_metrics[k][j]} ")
            metrics_file.flush()


def train_one_epoch(train_loader, val_loader, opt, sched, epoch, model, 
                    loss, print_batch, device):

    epoch_stats = init_epoch_stats()

    for i, data in enumerate(train_loader):
        opt.zero_grad()
        
        target_x, same_style_x, different_style_x, same_style_y, different_style_y = data
        target_x = target_x.to(device)
        same_style_x, different_style_x = same_style_x.to(device), different_style_x.to(device)
        same_style_y, different_style_y = same_style_y.to(device), different_style_y.to(device)
        
        pred_same = model(target_x, same_style_x)
        # print("pred_same=", pred_same)
        pred_diff = model(target_x, different_style_x)
        # print("pred_diff=", pred_diff)
        loss_same = loss(pred_same, same_style_y)
        # print("loss_same=", loss_same)
        loss_diff = loss(pred_diff, different_style_y)
        # print("loss_diff=", loss_diff)
        # print("\n*************************************\n")
        losses = torch.stack([loss_same, loss_diff])
        batch_loss = torch.mean(losses)
        batch_loss.backward()

        if i % print_batch == 0 and i != 0:
            val_accuracy, val_precision, val_recall, val_f_measure, \
                val_tp, val_tn, val_fp, val_fn = get_all_metrics(model, val_loader, device)
            train_batch_accuracy, train_batch_precision, train_batch_recall, train_batch_f_measure, \
                train_batch_tp, train_batch_tn, train_batch_fp, train_batch_fn = \
                    batch_stats(pred_same, pred_diff, same_style_y, different_style_y, return_measures=True)
            epoch_stats = update_epoch_stats(epoch_stats, batch_loss.detach().item(),
                train_batch_accuracy, train_batch_precision, train_batch_recall,
                train_batch_f_measure, train_batch_tp, train_batch_tn, train_batch_fp, train_batch_fn,
                val_accuracy, val_precision, val_recall, val_f_measure,
                val_tp, val_tn, val_fp, val_fn)

            print(f"Epoch-{epoch+1}-Batch-{i+1}: Loss = {round(batch_loss.detach().item(), 3)}, Loss Positive = {round(loss_same.detach().item(), 3)}, Loss Negative = {round(loss_diff.detach().item(), 3)}, Train Accuracy = {round(train_batch_accuracy, 3)}, Validation Accuracy = {round(val_accuracy, 3)}")

        opt.step()
    sched.step()

    return epoch_stats


def train(train_loader, val_loader, model, model_save_dir, 
          num_epochs, lr, device, save_epoch, print_batch, accuracy_save_thresh=0.95):

    start_epoch, best_accuracy = 0, 0
    new_best_acc = False
    model_params = list(model.parameters())
    opt = torch.optim.Adam([p for p in model_params if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, num_epochs,
        eta_min=0, last_epoch=-1)
    loss_func = ContrastiveLoss() # Siamese loss

    metrics_path = os.path.join(model_save_dir, f"metrics_target.txt")
    with open(metrics_path, 'w') as metrics_file:
        for epoch in range(start_epoch, num_epochs):
            epoch_metrics = train_one_epoch(
                train_loader, val_loader, opt, sched, epoch, model, loss_func, \
                print_batch, device)
           
            save_metrics2file(epoch_metrics, metrics_file)

            if best_accuracy < epoch_metrics["val_accuracy"][-1] and \
                epoch_metrics["val_accuracy"][-1] > accuracy_save_thresh:
                best_accuracy = epoch_metrics["val_accuracy"][-1]
                new_best_acc = True

            if (epoch+1) % save_epoch == 0 or new_best_acc:
                save_path = os.path.join(model_save_dir, f"target_identifier-e{epoch+1}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'scheduler_state_dict': sched.state_dict()
                    }, save_path)


def main(cfg):
    device = torch.device(f"cuda:{cfg['runtime']['gpu']}" if torch.cuda.is_available() and cfg["runtime"]["gpu"] is not None else "cpu")
    model_path, model_type, model_save_dir = cfg["model"]["path"], cfg["model"]["type"], cfg['runtime']["model_save_dir"]
    images_shape = (cfg["dataloader"]["image_height"], cfg["dataloader"]["image_width"])
    istrain, json_path = cfg["dataloader"]["istrain"], cfg["dataloader"]["json_path"] 
    batch_size, num_workers = cfg["dataloader"]["batch_size"], cfg["dataloader"]["num_workers"]
    mode_ratio, num_epochs, lr = cfg["dataloader"]["mode_ratio"], cfg["runtime"]["num_epochs"], cfg["runtime"]["lr"]
    save_epoch, print_batch = cfg["runtime"]["save_epoch"], cfg["runtime"]["print_batch"]

    model = siamese_factory(int(model_type))
    assert model is not None, "Invalid model_type"

    if model_path is not None:
        print(f"Loading checkpoint from {model_path}")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.apply(weights_init2)
    model = model.to(device)

    if istrain:
        model.train()
        for p in model.parameters():
            p.requires_grad = True

        start_mode_ratio, end_mode_ratio = 0, mode_ratio

        train_ds = SiameseDataset2(json_path, modes[0], start_mode_ratio, end_mode_ratio, images_shape[1], images_shape[0])

        start_mode_ratio = end_mode_ratio
        end_mode_ratio = 1

        val_ds = SiameseDataset2(json_path, modes[1], start_mode_ratio, end_mode_ratio, images_shape[1], images_shape[0])

        train_dl = D.DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers)
        val_dl = D.DataLoader(val_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers)

        print("Train...\n")
        print(f"Train set size: {len(train_ds)}, Train number of batches: {len(train_dl)}\nValidation set size: {len(val_ds)}, Validation number of batches: {len(val_dl)}\n")
        train(train_dl, val_dl, model, model_save_dir, num_epochs, lr, device, save_epoch, print_batch)
    
    else: # Test
        test_ds = SiameseDataset2(json_path, modes[2], start_mode_ratio, end_mode_ratio, images_shape[1], images_shape[0])
        test_dl = D.DataLoader(test_ds, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers)

        with torch.no_grad():
            model.eval()
            print("Test...")
            print(f"Test set size: {len(test_ds)}, Test number of batches: {len(test_dl)}")
            print(f"Metrics on {len(test_ds)}:\n{get_all_metrics(model, test_dl, device, is_siamese=is_siamese)}")
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Writer Identification Runtime Module')
    parser.add_argument('-c', '--config_path', default=None, type=str, 
        help="Path to a configuration .yaml file")

    args = parser.parse_args()
    assert args.config_path is not None, "Invalid config_path"
    with open(args.config_path, 'r') as cfg_f:
        cfg = yaml.load(cfg_f)
        main(cfg)