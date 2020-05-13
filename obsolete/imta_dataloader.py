import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from ops.dataset import TSNDataSet
from ops.transforms import *

def get_dataloaders(args):
    train_loader, val_loader, test_loader = None, None, None
    if args.data == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                         std=[0.2471, 0.2435, 0.2616])
        train_set = datasets.CIFAR10(args.data_root, train=True,
                                     transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                     ]))
        val_set = datasets.CIFAR10(args.data_root, train=False,
                                   transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    normalize
                                   ]))
    elif args.data == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        train_set = datasets.CIFAR100(args.data_root, train=True,
                                      transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                      ]))
        val_set = datasets.CIFAR100(args.data_root, train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize
                                    ]))
    elif args.data == 'dummy': # TODO (yue)
        return [(train_loader,train_loader)]*99, [(val_loader,val_loader)]*66, [(test_loader,test_loader)]*66
    elif args.data == 'actnet':  # TODO (yue)
        normalize = GroupNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_spec_aug = torchvision.transforms.Compose([GroupMultiScaleCrop(224, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])

        train_aug = transforms.Compose([
                           train_spec_aug,
                           Stack(roll=False),
                           ToTorchFormatTensor(div=True),
                           normalize,
                       ])

        val_aug = transforms.Compose([
                           GroupScale(256),
                           GroupCenterCrop(224),
                           Stack(roll=False),
                           ToTorchFormatTensor(div=True),
                           normalize,
                       ])
        from ops import dataset_config
        num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset('actnet')

        train_loader = torch.utils.data.DataLoader(
            TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                       image_tmpl=prefix,
                       transform=train_aug,
                       dataset='actnet',
                       folder_suffix="_self"),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True,
            drop_last=True)  # prevent something not % n_GPU

        val_loader = torch.utils.data.DataLoader(
            TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                       image_tmpl=prefix,
                       random_shift=False,
                       transform=val_aug,
                       dataset='actnet',
                       folder_suffix="_self"),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        return train_loader, val_loader, val_loader

    else:
        # ImageNet
        traindir = os.path.join(args.data_root, 'train')
        valdir = os.path.join(args.data_root, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_set = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]))
        val_set = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]))
    if args.use_valid:
        train_set_index = torch.randperm(len(train_set))
        if os.path.exists(os.path.join(args.save, 'index.pth')):
            print('!!!!!! Load train_set_index !!!!!!')
            train_set_index = torch.load(os.path.join(args.save, 'index.pth'))
        else:
            print('!!!!!! Save train_set_index !!!!!!')
            torch.save(train_set_index, os.path.join(args.save, 'index.pth'))
        if args.data.startswith('cifar'):
            num_sample_valid = 5000
        else:
            num_sample_valid = 50000
        # num_sample_valid = len(val_set)
        print("------------------------------------")
        print("split num_sample_valid: %d" % num_sample_valid)
        print("------------------------------------")

        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_set_index[:-num_sample_valid]),
                num_workers=args.workers, pin_memory=True)
        if 'val' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_set_index[-num_sample_valid:]),
                num_workers=args.workers, pin_memory=True)
        if 'test' in args.splits:
            test_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
    else:
        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
        if 'val' or 'test' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
            test_loader = val_loader

    return train_loader, val_loader, test_loader
