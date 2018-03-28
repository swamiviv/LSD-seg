import datetime
import math
import os
import os.path as osp
import shutil
import fcn
import numpy as np
import pytz
import scipy.misc
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm
import itertools,datetime
import torchvision.utils as vutils
import torchfcn
import torch.nn as nn
from util_fns import get_parameters
from utils import cross_entropy2d, step_scheduler

class Trainer_sourceonly(object):
    """
    Trainer class defining functions to train and validate the specified model
    """
    def __init__(self, cuda, model, optimizer,
                train_loader, target_loader, val_loader, out, max_iter,
                 size_average=False, interval_validate=None):
        self.cuda = cuda
        self.model = model
        self.optim = optimizer
        self.train_loader = train_loader
        self.target_loader = target_loader
        self.val_loader = val_loader
        self.timestamp_start = \
            datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        self.size_average = size_average
        self.n_class = len(self.train_loader.dataset.class_names)

        if interval_validate is None:
            self.interval_validate = min(len(self.train_loader), len(self.target_loader))
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'loss',
            'acc',
            'acc_cls',
            'mean_iu',
            'fwavacc',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.best_mean_iu = 0
        
    def validate(self):
        """
        Function to validate a training model on the val split.
        """
        
        self.model.eval()
        val_loss = 0
        num_vis = 8
        visualizations = []
        label_trues, label_preds = [], []
        
        # Loop to forward pass the data points into the model and measure the performance
        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration=%d' % self.iteration, ncols=80,
                leave=False):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            
            data, target = Variable(data, volatile=True), Variable(target)
            score = self.model(data)
            loss = cross_entropy2d(score, target,
                                   size_average=self.size_average)
            if np.isnan(float(loss.data[0])):
                raise ValueError('loss is nan while validating')
            val_loss += float(loss.data[0]) / len(data)

            imgs = data.data.cpu()
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu()
            
            # Function to save visualizations of the predicted label map
            for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                img = self.val_loader.dataset.untransform(img.numpy())
                lt = lt.numpy()
                label_trues.append(lt)
                label_preds.append(lp)
                if len(visualizations) < num_vis:
                    viz = fcn.utils.visualize_segmentation(
                        lbl_pred=lp, lbl_true=lt, img=img, n_class=self.n_class)
                    visualizations.append(viz)
        
        
        # Measuring the performance
        metrics = torchfcn.utils.label_accuracy_score(
            label_trues, label_preds, self.n_class)

        out = osp.join(self.out, 'visualization_viz')
        if not osp.exists(out):
            os.makedirs(out)
        out_file = osp.join(out, 'iter%012d.jpg' % self.iteration)
        scipy.misc.imsave(out_file, fcn.utils.get_tile_image(visualizations))

        val_loss /= len(self.val_loader)

        # Logging 
        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = \
                datetime.datetime.now(pytz.timezone('Asia/Tokyo')) - \
                self.timestamp_start
            log = [self.epoch, self.iteration] + [''] * 5 + \
                  [val_loss] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        # Saving the model checkpoint
        mean_iu = metrics[2]
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iu': self.best_mean_iu,
        }, osp.join(self.out, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))

    def train_epoch(self):
        """
        Function to train the model for one epoch
        """
        
        self.model.train()

        # Loop for training the model
        for batch_idx, datas in tqdm.tqdm(
                enumerate(self.train_loader), total= len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
            
            batch_size = 1
            iteration = batch_idx + self.epoch * len(self.train_loader)
            self.iteration = iteration
            if self.iteration % self.interval_validate == 0 and self.iteration>0:
                self.validate()
            self.model.train()

            # Obtaining data in the right format
            data_source, labels_source = datas
            if self.cuda:
                data_source, labels_source = data_source.cuda(), labels_source.cuda()
            data_source, labels_source = Variable(data_source), Variable(labels_source)
            
            # Forward pass
            self.optim.zero_grad()
            source_pred = self.model(data_source)
            
            # Computing the segmentation loss
            
            loss_seg = cross_entropy2d(source_pred, labels_source, size_average=self.size_average)
            loss_seg /= len(data_source)
            
            # Updating the model (backward pass)
            self.optim.zero_grad()
            loss_seg.backward()
            self.optim.step()
            
            if np.isnan(float(loss_seg.data[0])):
                raise ValueError('loss is nan while training')

            # Computing and logging performance metrics
            metrics = []
            lbl_pred = source_pred.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = labels_source.data.cpu().numpy()
            for lt, lp in zip(lbl_true, lbl_pred):
                acc, acc_cls, mean_iu, fwavacc = \
                    torchfcn.utils.label_accuracy_score(
                        [lt], [lp], n_class=self.n_class)
                metrics.append((acc, acc_cls, mean_iu, fwavacc))
            metrics = np.mean(metrics, axis=0)

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.datetime.now(pytz.timezone('Asia/Tokyo')) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [loss_seg.data[0]] + \
                    metrics.tolist() + [''] * 5 + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')

            if self.iteration >= self.max_iter:
                break

    def train(self):
        """
        Main training function
        """
        max_epoch = int(math.ceil(self.max_iter/len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch,
                                     desc='Train', ncols=80):
            self.epoch = epoch
            if self.epoch % 3 == 0 and self.epoch > 0:
                self.optim = step_scheduler(self.optim, self.epoch)
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break
