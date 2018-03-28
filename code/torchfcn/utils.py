import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc
    
    
def cross_entropy2d(input, target, weight=None, size_average=True):
    """
    Function to compute pixelwise cross-entropy for 2D image. This is the segmentation loss.
    Args:
        input: input tensor of shape (minibatch x num_channels x h x w)
        target: 2D label map of shape (minibatch x h x w)
        weight (optional): tensor of size 'C' specifying the weights to be given to each class
        size_average (optional): boolean value indicating whether the NLL loss has to be normalized
            by the number of pixels in the image 
    """
    
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input)
    
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    try:
        log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    except:
        print "Exception: ", target.size()
    log_p = log_p.view(-1, c)
    
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    target = torch.squeeze(target)
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()

    return loss

def step_scheduler(optimizer, epoch):
    """
    Function to perform step learning rate decay
    Args:
        optimizer: Optimizer for which step decay has to be applied
        epoch: Current epoch number
    """
    
    decay_factor = 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']*decay_factor

    return optimizer

