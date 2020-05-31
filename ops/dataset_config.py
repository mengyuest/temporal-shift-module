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



def return_charades(modality):
    filename_categories = ospj(common.CHARADES_META_PATH, 'categories.txt')
    if modality == 'RGB':
        root_data = common.CHARADES_FRAMES
        filename_imglist_train = ospj(common.CHARADES_META_PATH, 'train_segments.txt')
        filename_imglist_val = ospj(common.CHARADES_META_PATH, 'validation_segments.txt')
        prefix = '{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_epic(modality):
    filename_categories = "data/epic_kitchens2018/classInd.txt"  #ospj(common.EPIC_META_PATH, 'classIndActions.txt')
    if modality == 'RGB':
        root_data = common.EPIC_FRAMES
        filename_imglist_train = "data/epic_kitchens2018/training_split.txt" #ospj(common.EPIC_META_PATH, 'training_split.txt')
        filename_imglist_val = "data/epic_kitchens2018/validation_split.txt" #ospj(common.EPIC_META_PATH, 'validation_split.txt')
        prefix = 'frame_{:010d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_epic_verb(modality):
    filename_categories = ospj(common.EPIC_META_PATH, 'classIndVerbs.txt')
    if modality == 'RGB':
        root_data = common.EPIC_FRAMES
        filename_imglist_train = ospj(common.EPIC_META_PATH, 'training_verbs.txt')
        filename_imglist_val = ospj(common.EPIC_META_PATH, 'validation_verbs.txt')
        prefix = 'frame_{:010d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_epic_noun(modality):
    filename_categories = ospj(common.EPIC_META_PATH, 'classIndNouns.txt')
    if modality == 'RGB':
        root_data = common.EPIC_FRAMES
        filename_imglist_train = ospj(common.EPIC_META_PATH, 'training_nouns.txt')
        filename_imglist_val = ospj(common.EPIC_META_PATH, 'validation_nouns.txt')
        prefix = 'frame_{:010d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_something(modality):
    # filename_categories = 'something/v1/category.txt'
    filename_categories = ospj(common.STHV1_META_PATH,'classInd.txt')
    if modality == 'RGB':
        # root_data = ROOT_DATASET + 'something/v1/20bn-something-something-v1'
        # filename_imglist_train = 'something/v1/train_videofolder.txt'
        # filename_imglist_val = 'something/v1/val_videofolder.txt'
        root_data = common.STHV1_FRAMES
        filename_imglist_train = ospj(common.STHV1_META_PATH, "train_split.txt")
        filename_imglist_val = ospj(common.STHV1_META_PATH, "validation_split.txt")
        prefix = '{:05d}.jpg'
    # elif modality == 'Flow':
    #     root_data = ROOT_DATASET + 'something/v1/20bn-something-something-v1-flow'
    #     filename_imglist_train = 'something/v1/train_videofolder_flow.txt'
    #     filename_imglist_val = 'something/v1/val_videofolder_flow.txt'
    #     prefix = '{:06d}-{}_{:05d}.jpg'
    else:
        print('no such modality:'+modality)
        raise NotImplementedError
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_somethingv2(modality):
    filename_categories = ospj(common.STHV2_META_PATH,'classInd.txt')
    if modality == 'RGB':
        global ROOT_DATASET
        ROOT_DATASET = common.DATA_PATH
        #root_data = ROOT_DATASET + 'something/v2/20bn-something-something-v2-frames'
        #filename_imglist_train = 'something/v2/train_videofolder.txt'
        #filename_imglist_val = 'something/v2/val_videofolder.txt'
        root_data = common.STHV2_FRAMES
        filename_imglist_train = ospj(common.STHV2_META_PATH, "train_videofolder.txt")
        filename_imglist_val = ospj(common.STHV2_META_PATH, "val_videofolder.txt")
        prefix = '{:05d}.jpg'
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

def return_tinysth(modality):
    filename_categories = ospj(common.MINISTH_META_PATH, 'classIndTinySth2.txt')
    if modality == 'RGB':
        root_data = common.MINISTH_FRAMES
        filename_imglist_train = ospj(common.MINISTH_META_PATH, 'tiny_train_sth2.txt')
        filename_imglist_val = ospj(common.MINISTH_META_PATH, 'tiny_val_sth2.txt')
        prefix = '{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_teenysth(modality):
    filename_categories = ospj(common.MINISTH_META_PATH, 'classIndTeenySth2.txt')
    if modality == 'RGB':
        root_data = common.MINISTH_FRAMES
        filename_imglist_train = ospj(common.MINISTH_META_PATH, 'teeny_train_sth2.txt')
        filename_imglist_val = ospj(common.MINISTH_META_PATH, 'teeny_val_sth2.txt')
        prefix = '{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_low20sth(modality):
    filename_categories = ospj(common.MINISTH_META_PATH, 'classIndLow20Sth2.txt')
    if modality == 'RGB':
        root_data = common.MINISTH_FRAMES
        filename_imglist_train = ospj(common.MINISTH_META_PATH, 'low20_train_sth2.txt')
        filename_imglist_val = ospj(common.MINISTH_META_PATH, 'low20_val_sth2.txt')
        prefix = '{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_low42sth(modality):
    filename_categories = ospj(common.MINISTH_META_PATH, 'classIndLow42Sth2.txt')
    if modality == 'RGB':
        root_data = common.MINISTH_FRAMES
        filename_imglist_train = ospj(common.MINISTH_META_PATH, 'low42_train_sth2.txt')
        filename_imglist_val = ospj(common.MINISTH_META_PATH, 'low42_val_sth2.txt')
        prefix = '{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_minik(modality):
    filename_categories = 200
    #filename_categories = ospj(common.MINIK_META_PATH, 'classIndMiniK.txt')
    if modality == 'RGB':
        root_data = common.MINIK_FRAMES
        filename_imglist_train = ospj(common.MINIK_META_PATH, 'mini_train_videofolder.txt')
        filename_imglist_val = ospj(common.MINIK_META_PATH, 'mini_val_videofolder.txt')
        prefix = '{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_jester(modality):
    filename_categories = 'data/jester/classInd.txt'
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        root_data = common.JESTER_FRAMES
        filename_imglist_train = 'data/jester/train_split.txt'
        filename_imglist_val = 'data/jester/validation_split.txt'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_minijester(modality):
    filename_categories = "data/jester/classInd.txt"
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        root_data = common.JESTER_FRAMES
        filename_imglist_train = "data/jester/train_split_mini.txt"
        filename_imglist_val = "data/jester/validation_split_mini.txt"
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_kinetics(modality):
    filename_categories = 400
    if modality == 'RGB':
        root_data = common.K400_FRAMES
        filename_imglist_train = ospj(common.K400_META_PATH, 'train_400_toy.txt')  # TODO(changed)
        filename_imglist_val = ospj(common.K400_META_PATH, 'val_400_toy.txt')  # TODO(changed)
        prefix = '{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset, data_path):
    dict_single = {'jester': return_jester, 'something': return_something, 'somethingv2': return_somethingv2,
                   'ucf101': return_ucf101, 'hmdb51': return_hmdb51,
                   'kinetics': return_kinetics,
                   'actnet': return_actnet, 'fcvid':return_fcvid,
                   'charades': return_charades,
                   'minijester': return_minijester,
                   'epic': return_epic, 'epic_verb': return_epic_verb, 'epic_noun': return_epic_noun,
                   'ministh': return_ministh, 'minik': return_minik,
                   'tinysth': return_tinysth,
                   'teenysth': return_teenysth,
                   'low20sth': return_low20sth,
                   'low42sth': return_low42sth,
                   } # TODO(yue)
    common.set_manual_data_path(data_path, None)
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset]('RGB')
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
    # print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix
