import sys
sys.path.append('./AutoAugment/')

import torch
import torchvision
import torchvision.transforms as transforms
from AutoAugment.autoaugment import CIFAR10Policy, SVHNPolicy, ImageNetPolicy
#from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset

"""
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import \
    RandomResizedCropRGBImageDecoder, IntDecoder
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.transforms import *
from ffcv.fields import IntField, RGBImageField
import numpy as np
"""
#ffcv.transforms.NormalizeImage(mean: ndarray, std: ndarray, type: dtype)



def get_dataloaders(args):
    """
    if args.use_ffcv:
        args.num_classes = 100
        args.padding = 28
        args.size = 224
        IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
        IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
        #print(args.device)
        train_image_pipeline = [
            RandomResizedCropRGBImageDecoder((224, 224)),
            ToTensor(), 
            # Move to GPU asynchronously as uint8:
            #ToDevice(torch.device(args.device, non_blocking=True)), 
            ToDevice(args.device, non_blocking=True), 
            # Automatically channels-last:
            ToTorchImage(), 
            torchvision.transforms.AutoAugment(fill  = (128, 128, 128)),
            torchvision.transforms.Normalize(mean = IMAGENET_MEAN, std = IMAGENET_STD),
            #NormalizeImage(mean = IMAGENET_MEAN, std = IMAGENET_STD, type = np.float16),
            Convert(torch.float16), 
            ToDevice(args.device, non_blocking=True), 
            # Standard torchvision transforms still work!
            #torchvision.transforms.Normalize(mean = IMAGENET_MEAN, std = IMAGENET_STD),
        ]

        test_image_pipeline = [
            
            RandomResizedCropRGBImageDecoder((224, 224)),
            ToTensor(), 
            # Move to GPU asynchronously as uint8:
            #ToDevice(torch.device(args.device, non_blocking=True)), 
            ToDevice(args.device, non_blocking=True), 
            # Automatically channels-last:
            ToTorchImage(), 
            #NormalizeImage(mean = IMAGENET_MEAN, std = IMAGENET_STD, type = np.float16),
            torchvision.transforms.Normalize(mean = IMAGENET_MEAN, std = IMAGENET_STD),
            Convert(torch.float16), 
            ToDevice(args.device, non_blocking=True), 
            # Standard torchvision transforms still work!
            #torchvision.transforms.Normalize(mean = IMAGENET_MEAN, std = IMAGENET_STD)
            #NormalizeImage(mean = IMAGENET_MEAN, std = IMAGENET_STD, type = np.float16)
        ]

        train_loader = Loader('/raid/users/agrawal/imgnet/train/train_set.beton', batch_size=args.batch_size, 
        num_workers=args.num_workers, order=OrderOption.RANDOM,
        pipelines={'image': train_image_pipeline,
            'label': [
                IntDecoder(),
                ToTensor(),
                Squeeze(),
                ToDevice(torch.device(args.device), non_blocking=True)
            ]})

        val_loader = Loader('/raid/users/agrawal/imgnet/val/val_set.beton', batch_size=args.eval_batch_size, 
        num_workers=args.num_workers, order=OrderOption.RANDOM,
        pipelines={'image': test_image_pipeline,
            'label': [
                IntDecoder(),
                ToTensor(),
                Squeeze(),
                ToDevice(torch.device(args.device), non_blocking=True)
            ]})
        print('FFCV Data loader complete')
        return train_loader, val_loader
    """
    
    
    train_transform, test_transform = get_transform(args)

    if args.dataset == "c10":
        train_ds = torchvision.datasets.CIFAR10('./datasets', train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR10('./datasets', train=False, transform=test_transform, download=True)
        args.num_classes = 10
    elif args.dataset == "c100":
        train_ds = torchvision.datasets.CIFAR100('./datasets', train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR100('./datasets', train=False, transform=test_transform, download=True)
        args.num_classes = 100
    elif args.dataset == "svhn":
        train_ds = torchvision.datasets.SVHN('./datasets', split='train', transform=train_transform, download=True)
        test_ds = torchvision.datasets.SVHN('./datasets', split='test', transform=test_transform, download=True)
        args.num_classes = 10

    elif args.dataset == "imgnet":
        #train_ds = torchvision.datasets.ImageFolder('/scratch/vagrawal/data/imagenet-100/train', transform=train_transform)
        #test_ds = torchvision.datasets.ImageFolder('/scratch/vagrawal/data/imagenet-100/val', transform=test_transform)
        train_ds = torchvision.datasets.ImageFolder('../data/imagenet-100/train', transform=train_transform)
        test_ds = torchvision.datasets.ImageFolder('../data/imagenet-100/val', transform=test_transform)
        args.num_classes = 100
    else:
        raise ValueError(f"No such dataset:{args.dataset}")


    
    
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, prefetch_factor=4,shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, prefetch_factor=4,num_workers=args.num_workers, pin_memory=True)

    return train_dl, test_dl

def get_transform(args):
    if args.dataset in ["c10", "c100", 'svhn', 'imgnet']:
        if args.dataset != 'imgnet':
            args.padding=4
            args.size = 32
        else:
            args.padding = 28
            args.size = 224
            
        if args.dataset=="c10":
            args.mean, args.std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        elif args.dataset=="c100":
            args.mean, args.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        elif args.dataset=="svhn":
            args.mean, args.std = [0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]
        elif args.dataset=="imgnet":
            args.mean, args.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        args.padding=28
        args.size = 224
        args.mean, args.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        
    
    train_transform_list = [transforms.RandomCrop(size=(args.size,args.size), padding=args.padding)]

    if args.autoaugment:
        if args.dataset == 'c10' or args.dataset=='c100':
            train_transform_list.append(CIFAR10Policy())
        elif args.dataset == 'svhn':
            train_transform_list.append(SVHNPolicy())
        elif args.dataset == 'imgnet':
            train_transform_list.append(ImageNetPolicy())
        else:
            print(f"No AutoAugment for {args.dataset}")   
        
    if args.dataset=="imgnet":
        train_transform_list = [transforms.Resize(args.size)] + train_transform_list
        #train_transform_list = [transforms.RandomResizedCrop(args.size)] + [transforms.RandAugment(num_ops= args.rand_augment_ops, magnitude=args.rand_augment_mag)]
        
    train_transform = transforms.Compose(
        train_transform_list+[
            transforms.ToTensor(),
            transforms.Normalize(
                mean=args.mean,
                std = args.std
            )
        ]
    )
    
    if args.dataset=="imgnet":
        test_transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=args.mean,
            std = args.std
        )
    ])
        
    else:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=args.mean,
                std = args.std
            )
        ])

    return train_transform, test_transform
