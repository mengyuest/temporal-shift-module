# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os
import common
from os.path import join as ospj
ROOT_DATASET = None #'/store/datasets/'  # '/data/jilin/'

def return_ucf101(modality):
    filename_categories =  ospj(common.UCF101_META_PATH,'classInd.txt')
    if modality == 'RGB':
        root_data = common.UCF101_FRAMES
        filename_imglist_train = ospj(common.UCF101_META_PATH, 'ucf101_rgb_train_split_1.txt')
        filename_imglist_val = ospj(common.UCF101_META_PATH,'ucf101_rgb_val_split_1.txt')
        prefix = 'image_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'UCF101/jpg'
        filename_imglist_train = 'UCF101/file_list/ucf101_flow_train_split_1.txt'
        filename_imglist_val = 'UCF101/file_list/ucf101_flow_val_split_1.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_actnet(modality):
    filename_categories =  ospj(common.ACTNET_META_PATH, 'classInd.txt')
    if modality == 'RGB':
        root_data = common.ACTNET_FRAMES
        filename_imglist_train = ospj(common.ACTNET_META_PATH, 'actnet_train_split.txt')
        filename_imglist_val = ospj(common.ACTNET_META_PATH,'actnet_val_split.txt')
        prefix = 'image_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_fcvid(modality):
    filename_categories =  ospj(common.FCVID_META_PATH,'classInd.txt')
    if modality == 'RGB':
        root_data = common.FCVID_FRAMES
        filename_imglist_train = ospj(common.FCVID_META_PATH, 'fcvid_train_split.txt')
        filename_imglist_val = ospj(common.FCVID_META_PATH,'fcvid_val_split.txt')
        prefix = 'image_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_hmdb51(modality):
    filename_categories = ospj(common.HMDB51_META_PATH,'classInd.txt')
    if modality == 'RGB':
        root_data = common.HMDB51_FRAMES
        filename_imglist_train = ospj(common.HMDB51_META_PATH,'hmdb51_rgb_train_split_1.txt')
        filename_imglist_val = ospj(common.HMDB51_META_PATH,'hmdb51_rgb_val_split_1.txt')
        prefix = 'image_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'HMDB51/images'
        filename_imglist_train = 'HMDB51/splits/hmdb51_flow_train_split_1.txt'
        filename_imglist_val = 'HMDB51/splits/hmdb51_flow_val_split_1.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_something(modality):
    filename_categories = 'something/v1/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'something/v1/20bn-something-something-v1'
        filename_imglist_train = 'something/v1/train_videofolder.txt'
        filename_imglist_val = 'something/v1/val_videofolder.txt'
        prefix = '{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'something/v1/20bn-something-something-v1-flow'
        filename_imglist_train = 'something/v1/train_videofolder_flow.txt'
        filename_imglist_val = 'something/v1/val_videofolder_flow.txt'
        prefix = '{:06d}-{}_{:05d}.jpg'
    else:
        print('no such modality:'+modality)
        raise NotImplementedError
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_somethingv2(modality):
    filename_categories = ospj(common.STHV2_PATH,'category.txt')
    if modality == 'RGB':
        global ROOT_DATASET
        ROOT_DATASET = common.DATA_PATH
        #root_data = ROOT_DATASET + 'something/v2/20bn-something-something-v2-frames'
        #filename_imglist_train = 'something/v2/train_videofolder.txt'
        #filename_imglist_val = 'something/v2/val_videofolder.txt'
        root_data = common.STHV2_FRAMES
        filename_imglist_train = ospj(common.STHV2_PATH, "train_videofolder.txt")
        filename_imglist_val = ospj(common.STHV2_PATH, "val_videofolder.txt")
        prefix = '{:06d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'something/v2/20bn-something-something-v2-flow'
        filename_imglist_train = 'something/v2/train_videofolder_flow.txt'
        filename_imglist_val = 'something/v2/val_videofolder_flow.txt'
        prefix = '{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_ministh(modality):
    filename_categories = ospj(common.MINISTH_META_PATH, 'classIndMiniSth2.txt')
    if modality == 'RGB':
        root_data = common.MINISTH_FRAMES
        filename_imglist_train = ospj(common.MINISTH_META_PATH, 'mini_train_videofolder.txt')
        filename_imglist_val = ospj(common.MINISTH_META_PATH, 'mini_val_videofolder.txt')
        prefix = '{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_minik(modality):
    filename_categories = ospj(common.MINIK_META_PATH, 'classIndMiniK.txt')
    if modality == 'RGB':
        root_data = common.MINIK_FRAMES
        filename_imglist_train = ospj(common.MINIK_META_PATH, 'mini_train_videofolder.txt')
        filename_imglist_val = ospj(common.MINIK_META_PATH, 'mini_val_videofolder.txt')
        prefix = '{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_jester(modality):
    filename_categories = 'jester/category.txt'
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        root_data = ROOT_DATASET + 'jester/20bn-jester-v1'
        filename_imglist_train = 'jester/train_videofolder.txt'
        filename_imglist_val = 'jester/val_videofolder.txt'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_kinetics(modality):
    filename_categories = 400
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'kinetics/images'
        filename_imglist_train = 'kinetics/labels/train_videofolder.txt'
        filename_imglist_val = 'kinetics/labels/val_videofolder.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset, modality):
    dict_single = {'jester': return_jester, 'something': return_something, 'somethingv2': return_somethingv2,
                   'ucf101': return_ucf101, 'hmdb51': return_hmdb51,
                   'kinetics': return_kinetics,
                   'actnet': return_actnet, 'fcvid':return_fcvid,
                   'ministh': return_ministh, 'minik': return_minik} # TODO(yue)
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)

    #file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    #file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    if isinstance(file_categories, str):
        #file_categories = os.path.join(ROOT_DATASET, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  # number of categories
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix
