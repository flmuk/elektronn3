'''PointCNN.Pytorch
Adopted from https://github.com/hxdengBerkeley/PointCNN.Pytorch/blob/master/train_pytorch.py
Workflow of spinal semantic segmentation based on pointclouds (3D semantic segmentation).

It learns how to differentiate between spine head, spine neck and spine shaft.
'''
#!/usr/bin/env python3

# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch, Philipp Schubert
import numpy as np
import matplotlib
matplotlib.use("agg", force=True, warn=False)
import argparse
import os
from elektronn3.models.pointcnn_pytorch import Classifier
import torch
from torch import nn
from torch import optim
from elektronn3.training.loss import BlurryBoarderLoss, DiceLoss, LovaszLoss
from torch.nn import CrossEntropyLoss, NLLLoss
from elektronn3.data import transforms
from elektronn3.data.transforms import RotatePointCloud, JitterScalePointCloud


def get_model():
    print("------Building model-------")
    model = Classifier().cuda()
    print("------Successfully Built model-------")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a network.')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('-n', '--exp-name', default="PointCNN_small_batch_size_and_epoch", help='Manually set experiment name')
    parser.add_argument(
        '-m', '--max-steps', type=int, default=500000,
        help='Maximum number of training steps to perform.'
    )

    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
    parser.add_argument('--max_epoch', type=int, default=2, help='Epoch to run [default: 250]')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
    parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
    parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
    parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
    args = parser.parse_args()

    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f'Running on device: {device}')

    # Don't move this stuff, it needs to be run this early to work
    import elektronn3
    elektronn3.select_mpl_backend('Agg')

    from elektronn3.training import Trainer, Backup
    from elektronn3.data.cnndata import PointCNNData

    torch.manual_seed(0)


    # USER PATHS
    save_root = os.path.expanduser('~/e3training/')

    max_steps = args.max_steps
    lr = 0.0001
    lr_stepsize = 1000
    lr_dec = 0.99
    batch_size = 2 #set 15

    model = get_model()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        batch_size = batch_size * torch.cuda.device_count()
        # dim = 0 [20, xxx] -> [10, ...], [10, ...] on 2 GPUs
        model = nn.DataParallel(model)
    model.to(device)

    # Specify data set and augment batched point clouds by rotation and jittering
    class_names = ['neck', 'head' , 'shaft', 'other']
    jit = JitterScalePointCloud(scale=np.array([106640 / 2, 109130 / 2, 114000 / 2]),
                                clip=2)
    rot = RotatePointCloud()
    transform = transforms.Compose([jit, rot])
    train_dataset = PointCNNData(class_names, train=True, transform = transform)
    valid_dataset = PointCNNData(class_names, train=False, transform = transform)

# Set up optimization
    optimizer = optim.Adam(
        model.parameters(),
        weight_decay=0.5e-4,
        lr=lr,
        amsgrad=True
    )
    lr_sched = optim.lr_scheduler.StepLR(optimizer, lr_stepsize, lr_dec)

    criterion = NLLLoss().to(device)

    # Create and run trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        batchsize=batch_size,
        num_workers=6,
        save_root=save_root,
        exp_name=args.exp_name,
        schedulers={"lr": lr_sched},
        ipython_on_error=False
    )

    # Archiving training script, src folder, env info
    bk = Backup(script_path=__file__,save_path=trainer.save_path).archive_backup()

    trainer.train(max_steps)
