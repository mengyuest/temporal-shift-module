# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
# ------------------------------------------------------
# Code adapted from https://github.com/metalbubble/TRN-pytorch/blob/master/process_dataset.py
# processing the raw data of the video Something-Something-V2

import os
import json
import sys
sys.path.append('../')
import common
from os.path import join as ospj

if __name__ == '__main__':
    dataset_splits_dir = ospj(common.UCF101_PATH, "splits_classification")
    categories = [x.strip().split()[1] for x in open(ospj(dataset_splits_dir,"classInd.txt")).readlines()]

    with open(ospj(common.UCF101_META_PATH, 'classInd.txt'), 'w') as f:
        f.write('\n'.join(categories))

    dict_categories = {x:i for i,x in enumerate(categories)}

    files_input = ['trainlist01.txt'] #['testlist01.txt', 'trainlist01.txt']
    files_output = ['ucf101_rgb_train_split_1.txt'] #['ucf101_rgb_val_split_1.txt', 'ucf101_rgb_train_split_1.txt']
    for (filename_input, filename_output) in zip(files_input, files_output):
        lines = open(ospj(dataset_splits_dir, filename_input)).readlines()
        folders = []
        idx_categories = []
        for line in lines:
            folder_name=line.strip().split()[0][:-4]
            label_name = folder_name.split("/")[0]
            folders.append(folder_name)
            idx_categories.append(dict_categories[label_name])
        output = []
        for i in range(len(folders)):
            curFolder = folders[i]
            curIDX = idx_categories[i]
            # counting the number of frames in each video folders
            dir_files = os.listdir(ospj(common.UCF101_FRAMES, curFolder))
            output.append('%s %d %d' % (curFolder, len(dir_files), curIDX))
            print('%d/%d' % (i, len(folders)))
        with open(ospj(common.UCF101_META_PATH,filename_output), 'w') as f:
            f.write('\n'.join(output))
