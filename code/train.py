import argparse
import os
import os.path as osp
import sys
import torch
import torchfcn
from util_fns import get_log_dir
from util_fns import get_parameters
from util_fns import weights_init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True, help='Path to source dataset')
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--num_iters', type=int, default=100000, help='Number of training iterations')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer to use | SGD, Adam')
    parser.add_argument('--lr', type=float, default=1.0e-5, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.99, help='Momentum for SGD')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--interval_validate', type=int, default=500, help='Period for validation. Model is validated every interval_validate iterations')
    parser.add_argument('--resume', default='', help="path to the current checkpoint for resuming training. Do not specify if model has to be trained from scratch")
    parser.add_argument('--logdir', default='logs', help="Path to the directory to store log files")
    parser.add_argument('--method', default='LSD', help="Method to use for training | LSD, sourceonly")
    parser.add_argument('--l1_weight', type=float, default=1, help='L1 weight')
    parser.add_argument('--adv_weight', type=float, default=0.1, help='Adv_weight')
    parser.add_argument('--c_weight', type=float, default=0.1, help='C_weight')
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    print(args)

    gpu = args.gpu
    out = get_log_dir(args.logdir, args.method, args.lr, args.optimizer)
    resume = args.resume
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()
    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # Defining data loaders
    
    image_size=[640, 320]
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.SYNTHIA('SYNTHIA', args.dataroot, split='train', transform=True, image_size=image_size),
        batch_size=args.batchSize, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.SYNTHIA('SYNTHIA', args.dataroot, split='val', transform=True, image_size=image_size),
        batch_size=args.batchSize, shuffle=False, **kwargs)
    target_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.CityScapes('cityscapes', args.dataroot, split='train', transform=True, image_size=image_size),
        batch_size=1, shuffle=True)

    # Defining models

    start_epoch = 0
    start_iteration = 0
    if args.method == 'sourceonly':
        model = torchfcn.models.FCN8s_sourceonly(n_class=19)
    elif args.method == 'LSD':
        model = torchfcn.models.FCN8s_LSD(n_class=19)
        netG = torchfcn.models._netG()
        netD = torchfcn.models._netD()
        netD.apply(weights_init)
        netG.apply(weights_init)
    else:
        raise ValueError('method argument can be either sourceonly or LSD')
    
    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        vgg16 = torchfcn.models.VGG16(pretrained=True)
        model.copy_params_from_vgg16(vgg16)
    if cuda:
        model = model.cuda()    
        if args.method == 'LSD':
            netD = netD.cuda()
            netG = netG.cuda()
        
    # Defining optimizer
    
    if args.optimizer == 'SGD':
        optim = torch.optim.SGD(
            [
                {'params': get_parameters(model, bias=False)},
                {'params': get_parameters(model, bias=True),
                 'lr': args.lr * 2, 'weight_decay': args.weight_decay},
            ],
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optim = torch.optim.Adam(
            [
                {'params': get_parameters(model, bias=False)},
                {'params': get_parameters(model, bias=True),
                 'lr': args.lr * 2},
            ],
            lr=args.lr,
            betas=(args.beta1, 0.999))
    else:
        raise ValueError('Invalid optmizer argument. Has to be SGD or Adam')
    
    if args.method == 'LSD':
        optimD = torch.optim.Adam(netD.parameters(), lr=0.0001, betas=(0.7, 0.999))
        optimG = torch.optim.Adam(netG.parameters(), lr=0.0001, betas=(0.7, 0.999))
    
    if resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    # Defining trainer object, and start training
    if args.method == 'sourceonly':
        trainer = torchfcn.Trainer_sourceonly(
            cuda=cuda,
            model=model,
            optimizer=optim,
            train_loader=train_loader,
            target_loader=target_loader,
            val_loader=val_loader,
            out=out,
            max_iter=args.num_iters,
            interval_validate=args.interval_validate
        )
        trainer.epoch = start_epoch
        trainer.iteration = start_iteration
        trainer.train()
    elif args.method == 'LSD':
        trainer = torchfcn.Trainer_LSD(
            cuda=cuda,
            model=model,
            netD=netD,
            netG=netG,
            optimizer=optim,
            optimizerD=optimD,
            optimizerG=optimG,
            train_loader=train_loader,
            target_loader=target_loader,
            l1_weight=args.l1_weight,
            adv_weight=args.adv_weight,
            c_weight=args.c_weight,
            val_loader=val_loader,
            out=out,
            max_iter=args.num_iters,
            interval_validate=args.interval_validate,
            image_size=image_size
        )
        trainer.epoch = start_epoch
        trainer.iteration = start_iteration
        trainer.train()
    

if __name__ == '__main__':
    main()
