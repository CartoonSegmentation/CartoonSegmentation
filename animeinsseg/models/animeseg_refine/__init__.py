# modified from https://github.com/SkyTNT/anime-segmentation/blob/main/train.py
import os

import argparse
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import cv2
from torch.cuda import amp

from utils.constants import DEFAULT_DEVICE
# from data_loader import create_training_datasets


import pytorch_lightning as pl
import warnings

from .isnet import ISNetDIS, ISNetGTEncoder
from .u2net import U2NET, U2NET_full, U2NET_full2, U2NET_lite2
from .modnet import MODNet

# warnings.filterwarnings("ignore")

def get_net(net_name):
    if net_name == "isnet":
        return ISNetDIS()
    elif net_name == "isnet_is":
        return ISNetDIS()
    elif net_name == "isnet_gt":
        return ISNetGTEncoder()
    elif net_name == "u2net":
        return U2NET_full2()
    elif net_name == "u2netl":
        return U2NET_lite2()
    elif net_name == "modnet":
        return MODNet()
    raise NotImplemented


def f1_torch(pred, gt):
    # micro F1-score
    pred = pred.float().view(pred.shape[0], -1)
    gt = gt.float().view(gt.shape[0], -1)
    tp1 = torch.sum(pred * gt, dim=1)
    tp_fp1 = torch.sum(pred, dim=1)
    tp_fn1 = torch.sum(gt, dim=1)
    pred = 1 - pred
    gt = 1 - gt
    tp2 = torch.sum(pred * gt, dim=1)
    tp_fp2 = torch.sum(pred, dim=1)
    tp_fn2 = torch.sum(gt, dim=1)
    precision = (tp1 + tp2) / (tp_fp1 + tp_fp2 + 0.0001)
    recall = (tp1 + tp2) / (tp_fn1 + tp_fn2 + 0.0001)
    f1 = (1 + 0.3) * precision * recall / (0.3 * precision + recall + 0.0001)
    return precision, recall, f1


class AnimeSegmentation(pl.LightningModule):

    def __init__(self, net_name):
        super().__init__()
        assert net_name in ["isnet_is", "isnet", "isnet_gt", "u2net", "u2netl", "modnet"]
        self.net = get_net(net_name)
        if net_name == "isnet_is":
            self.gt_encoder = get_net("isnet_gt")
            self.gt_encoder.requires_grad_(False)
        else:
            self.gt_encoder = None

    @classmethod
    def try_load(cls, net_name, ckpt_path, map_location=None):
        state_dict = torch.load(ckpt_path, map_location=map_location)
        if "epoch" in state_dict:
            return cls.load_from_checkpoint(ckpt_path, net_name=net_name, map_location=map_location)
        else:
            model = cls(net_name)
            if any([k.startswith("net.") for k, v in state_dict.items()]):
                model.load_state_dict(state_dict)
            else:
                model.net.load_state_dict(state_dict)
            return model

    def configure_optimizers(self):
        optimizer = optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        return optimizer

    def forward(self, x):
        if isinstance(self.net, ISNetDIS):
            return self.net(x)[0][0].sigmoid()
        if isinstance(self.net, ISNetGTEncoder):
            return self.net(x)[0][0].sigmoid()
        elif isinstance(self.net, U2NET):
            return self.net(x)[0].sigmoid()
        elif isinstance(self.net, MODNet):
            return self.net(x, True)[2]
        raise NotImplemented

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        if isinstance(self.net, ISNetDIS):
            ds, dfs = self.net(images)
            loss_args = [ds, dfs, labels]
        elif isinstance(self.net, ISNetGTEncoder):
            ds = self.net(labels)[0]
            loss_args = [ds, labels]
        elif isinstance(self.net, U2NET):
            ds = self.net(images)
            loss_args = [ds, labels]
        elif isinstance(self.net, MODNet):
            trimaps = batch["trimap"]
            pred_semantic, pred_detail, pred_matte = self.net(images, False)
            loss_args = [pred_semantic, pred_detail, pred_matte, images, trimaps, labels]
        else:
            raise NotImplemented
        if self.gt_encoder is not None:
            fs = self.gt_encoder(labels)[1]
            loss_args.append(fs)

        loss0, loss = self.net.compute_loss(loss_args)
        self.log_dict({"train/loss": loss, "train/loss_tar": loss0})
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        if isinstance(self.net, ISNetGTEncoder):
            preds = self.forward(labels)
        else:
            preds = self.forward(images)
        pre, rec, f1, = f1_torch(preds.nan_to_num(nan=0, posinf=1, neginf=0), labels)
        mae_m = F.l1_loss(preds, labels, reduction="mean")
        pre_m = pre.mean()
        rec_m = rec.mean()
        f1_m = f1.mean()
        self.log_dict({"val/precision": pre_m, "val/recall": rec_m, "val/f1": f1_m, "val/mae": mae_m}, sync_dist=True)


def get_gt_encoder(train_dataloader, val_dataloader, opt):
    print("---start train ground truth encoder---")
    gt_encoder = AnimeSegmentation("isnet_gt")
    trainer = Trainer(precision=32 if opt.fp32 else 16, accelerator=opt.accelerator,
                      devices=opt.devices, max_epochs=opt.gt_epoch,
                      benchmark=opt.benchmark, accumulate_grad_batches=opt.acc_step,
                      check_val_every_n_epoch=opt.val_epoch, log_every_n_steps=opt.log_step,
                      strategy="ddp_find_unused_parameters_false" if opt.devices > 1 else None,
                      )
    trainer.fit(gt_encoder, train_dataloader, val_dataloader)
    return gt_encoder.net


def load_refinenet(refine_method = 'animeseg', device: str = None) -> AnimeSegmentation:
    if device is None:
        device = DEFAULT_DEVICE
    if refine_method == 'animeseg':
        model = AnimeSegmentation.try_load('isnet_is', 'models/anime-seg/isnetis.ckpt', device)
    elif refine_method == 'refinenet_isnet':
        model = ISNetDIS(in_ch=4)
        sd = torch.load('models/AnimeInstanceSegmentation/refine_last.ckpt', map_location='cpu')
        # sd = torch.load('models/AnimeInstanceSegmentation/refine_noweight_dist.ckpt', map_location='cpu')
        # sd = torch.load('models/AnimeInstanceSegmentation/refine_f3loss.ckpt', map_location='cpu')
        model.load_state_dict(sd)
    else:
        raise NotImplementedError
    return model.eval().to(device)
    
def get_mask(model, input_img, use_amp=True, s=640):
    h0, w0 = h, w = input_img.shape[0], input_img.shape[1]
    if h > w:
        h, w = s, int(s * w / h)
    else:
        h, w = int(s * h / w), s
    ph, pw = s - h, s - w
    tmpImg = np.zeros([s, s, 3], dtype=np.float32)
    tmpImg[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(input_img, (w, h)) / 255
    tmpImg = tmpImg.transpose((2, 0, 1))
    tmpImg = torch.from_numpy(tmpImg).unsqueeze(0).type(torch.FloatTensor).to(model.device)
    with torch.no_grad():
        if use_amp:
            with amp.autocast():
                pred = model(tmpImg)
            pred = pred.to(dtype=torch.float32)
        else:
            pred = model(tmpImg)
        pred = pred[0, :, ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
        pred = cv2.resize(pred.cpu().numpy().transpose((1, 2, 0)), (w0, h0))[:, :, np.newaxis]
        return pred