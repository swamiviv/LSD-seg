import torch
import os
import os.path as osp
import datetime
import pytz
import yaml
import torchfcn

def get_log_dir(log_dir, model_name, lr, opt):
    
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
        
    # load config
    name = 'MODEL-%s_CFG-%s_LR_%.8f' % (model_name, opt, lr)
    
    # create out
    log_dir = osp.join(log_dir, name)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def get_parameters(model, bias=False):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.Sequential,
        torchfcn.models.FCN8s_sourceonly,
        torchfcn.models.FCN8s_LSD,
        torchfcn.models.Res_Deeplab
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        elif isinstance(m, modules_skipped):
            continue
        else:
            raise ValueError('Unexpected module: %s' % str(m))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.uniform_(-0.01, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


