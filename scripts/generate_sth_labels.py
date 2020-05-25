import os
from os.path import join as ospj
import tqdm

path="/nobackup/users/mengyue/datasets/sthv1"

ls = open(ospj(path,"something-something-v1-labels.csv")).readlines()
classes = sorted([l for l in ls if len(l) > 5])
class_d = {}
with open(ospj(path,"classInd.txt"), "w") as f:
    for i, l in enumerate(ls):
        class_d[l.strip()] = i
        f.write("%d,%s\n" % (i, l.strip()))

for split in ["train", "validation"]:
    ls = open(ospj(path,"something-something-v1-%s.csv"%split)).readlines()
    ls = [l for l in ls if len(l)>5]
    with open(ospj(path,"%s_split.txt"%(split)), "w") as f:
        for l in tqdm.tqdm(ls):
            items=l.strip().split(";")
            dirname = int(items[0])
            classname = items[1]
            class_idx = class_d[classname]
            num_files = len(os.listdir(ospj(path,"20bn-something-something-v1/%d" % dirname)))
            f.write("%d %d %d\n" % (dirname, num_files, class_idx))


