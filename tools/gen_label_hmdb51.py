import os
from os.path import join as ospj
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser(description="Split file generation for HMDB51")
# TODO(yue) we will generate three different splits (6 files in total(train+val))
# TODO(yue) load raw splits from "label_path", check file counts in "dataset_path", and save splits to "output_path"
parser.add_argument("dataset_path", type=str)
parser.add_argument("label_path", type=str)
parser.add_argument("output_path",type=str)
args = parser.parse_args()

if __name__ == '__main__':
    class_names = sorted(list(set([x.split("_test_split")[0] for x in os.listdir(args.label_path)])))
    assert len(class_names) == 51

    # TODO(yue) generate classInd.txt
    class_ind_file_path = ospj(args.output_path,"classInd.txt")
    if os.path.exists(class_ind_file_path):
        ans=input("classInd.txt already exists so do you want to overwrite it?")
        if ans.split()[0].upper() not in ["YES", "Y", "DO"]:
            exit("Please remove your classInd.txt if you still want to run this.")
    with open(class_ind_file_path,"w") as f:
        for i,class_name in enumerate(class_names):
            f.write("%d %s\n"%(i,class_name))

    # TODO(yue) write split files
    for split_idx in [1,2,3]:
        train_split_path=ospj(args.output_path, "hmdb51_rgb_train_split_%d.txt"%(split_idx))
        val_split_path=ospj(args.output_path, "hmdb51_rgb_val_split_%d.txt"%(split_idx))
        with open(train_split_path,"w") as f_train:
            with open(val_split_path,"w") as f_val:
                for curIDX, class_name in enumerate(tqdm(class_names)):
                    lines = open(ospj(args.label_path, "%s_test_split%d.txt"%(class_name, split_idx))).readlines()
                    for line in lines:
                        video_name=line.strip().split()[0]
                        sample_name = video_name.split(".")[0]
                        trainval_idx=int(line.strip().split()[1])
                        dir_files = os.listdir(ospj(args.dataset_path, class_name, sample_name))
                        f_todo = None
                        if trainval_idx==1: # TODO(yue) belong to train
                            f_todo = f_train
                        elif trainval_idx==2: # TODO(yue) belongs to val/test
                            f_todo = f_val
                        else:
                            continue
                        f_todo.write("%s %d %d\n"%(ospj(class_name, sample_name), len(dir_files), curIDX))