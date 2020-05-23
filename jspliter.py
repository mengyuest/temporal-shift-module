import os
from os.path import join as ospj
import tqdm
import common
common.set_manual_data_path(None, None)
downsample_rate=4

ls=open("data/jester/jester-v1-labels.csv").readlines()

class_idx_d = {}

with open("data/jester/classInd.txt", "w") as f:
    for i, l in enumerate(ls):
        class_idx_d[l.strip()] = i
        f.write("%d,%s"%(i,l))

sample_list={"train":{}, "validation":{}}
frames_list={"train":{}, "validation":{}}
for split in ["train", "validation"]:
    ls=open("data/jester/jester-v1-%s.csv"%split).readlines()
    with open("data/jester/%s_split.txt"%split, "w") as f:
        for l in tqdm.tqdm(ls):
            items = l.strip().split(";")
            class_name = items[1]
            class_id = class_idx_d[class_name]
            if class_name not in sample_list[split]:
                sample_list[split][class_name]=[]
                frames_list[split][class_name]={}
            sample_list[split][class_name].append(items[0])
            num_files = len(os.listdir(ospj(common.JESTER_FRAMES, items[0])))
            frames_list[split][class_name][items[0]]=num_files
            f.write("%s %d %d\n" % (items[0], num_files, class_id))

threshold={"train":{}, "validation":{}}
count={"train":{}, "validation":{}}
for split in ["train", "validation"]:
    for x in sample_list[split]:
        threshold[split][x] = len(sample_list[split][x])//downsample_rate
        count[split][x] = 0
    ls=open("data/jester/jester-v1-%s.csv"%split).readlines()
    with open("data/jester/%s_split_mini.txt" % split, "w") as f:
        for l in tqdm.tqdm(ls):
            items = l.strip().split(";")
            class_name = items[1]
            class_id = class_idx_d[class_name]
            count[split][class_name]+=1
            if count[split][class_name]>threshold[split][class_name]:
                continue
            num_files = frames_list[split][class_name][items[0]]
            f.write("%s %d %d\n" % (items[0], num_files, class_id))