import numpy as np
import matplotlib.pyplot as plt
import pickle

import os, sys
from os.path import expanduser
from os.path import join as ospj
import socket
import getpass

from shutil import copyfile
# copyfile(src, dst)

host_name = socket.gethostname()
user_name = getpass.getuser()

topk = 20

if "mengyue" in user_name:
    exp_path = "/nobackup/users/mengyue/logs_tsm/"
    classInds={
        "sthv1": "/nobackup/users/mengyue/datasets/sthv1/classInd.txt",
        "sthv2": "/nobackup/users/mengyue/datasets/something2something-v2/classInd.txt",
        "minik": "/nobackup/users/mengyue/datasets/kinetics-qf/minik_classInd.txt",
        "jester": "/nobackup/users/mengyue/code/temporal-shift-module/data/jester/classInd.txt",
    }
    dataset_paths={
        "sthv1": "/nobackup/users/mengyue/datasets/sthv1/20bn-something-something-v1/",
        "sthv2": "/nobackup/users/mengyue/datasets/something2something-v2/frames/",
        "minik": "/nobackup/users/mengyue/datasets/kinetics-qf/",
        "jester": "/nobackup/users/mengyue/datasets/jester/20bn-jester-v1/",
    }

    prefixs={
        "sthv1": '{:05d}.jpg',
        "sthv2": '{:05d}.jpg',
        "minik": '{:05d}.jpg',
        "jester": '{:05d}.jpg',
    }

else:
    exp_path = "/Users/meng/Downloads/buffer/"

paths = [
    "g0609-231417_sthv1_8_bate50_q196ksha_gsmx_g.125_upbg_b64_lr.01step_SAVEMETA",
    "g0609-231823_sthv2_8_bate50_q196ksha_gsmx_g.125_upbg_b64_lr.02step_SAVEMETA",
    "g0609-232413_minik_8_bate50_q196ksha_gsmx_g.125_upbg_b64_lr.02step_SAVEMETA",
    "g0609-233305_jester_8_bate50_512sha_gsmx_g.05_real_b64_lr.02step_SAVEMETA"
]

gflops_list = [
    [0.0128, 0.1156, 0.0514, 0.0514, 0.0000],
    [0.0514, 0.1156, 0.0514, 0.0000, 0.0000],
    [0.0514, 0.1156, 0.0514, 0.0000, 0.0000],
    [0.1028, 0.1156, 0.0514, 0.1028, 0.0000],
    [0.0514, 0.1156, 0.0514, 0.0000, 0.0000],
    [0.0514, 0.1156, 0.0514, 0.0000, 0.0000],
    [0.0514, 0.1156, 0.0514, 0.0000, 0.0000],
    [0.1028, 0.1156, 0.0514, 0.1028, 0.0000],
    [0.0514, 0.1156, 0.0514, 0.0000, 0.0000],
    [0.0514, 0.1156, 0.0514, 0.0000, 0.0000],
    [0.0514, 0.1156, 0.0514, 0.0000, 0.0000],
    [0.0514, 0.1156, 0.0514, 0.0000, 0.0000],
    [0.0514, 0.1156, 0.0514, 0.0000, 0.0000],
    [0.1028, 0.1156, 0.0514, 0.1028, 0.0000],
    [0.0514, 0.1156, 0.0514, 0.0000, 0.0000],
    [0.0514, 0.1156, 0.0514, 0.0000, 0.0000],
]


def draw(path):
    with open(exp_path + path + "/record-path-val.pkl", "rb") as f:
        records = pickle.load(f)
    with open(exp_path + path + "/indices-val.pkl", "rb") as f:
        indices = pickle.load(f)
    with open(exp_path + path + "/gate-stat-val.pkl", "rb") as f:
        gates = pickle.load(f)
    data = np.load(exp_path + path + "/meta-gate-val.npy.npz")
    # print(path)
    # print(len(records))
    # print(indices.shape)
    # print(len(gates))
    # print(data.files)

    num_classes = len(np.unique(data["targets"])) - 1
    # print(np.unique(data["targets"]))
    print("classes:", num_classes)

    stat = {
        "skip": [[] for i in range(num_classes)],
        "reuse": [[] for i in range(num_classes)],
        "keep": [[] for i in range(num_classes)]
    }

    # collect
    reduced_ltc_gate = [0 for _ in range(3)]
    for layer_i in range(len(gates)):
        skip = np.sum(gates[layer_i] == 0, axis=(1, 2))
        reuse = np.sum(gates[layer_i] == 1, axis=(1, 2))
        keep = np.sum(gates[layer_i] == 2, axis=(1, 2))
        # reduced_tc_gate.append([skip, reuse, keep, skip+reuse+keep])
        total = skip + reuse + keep
        reduced_ltc_gate[0] += skip / total
        reduced_ltc_gate[1] += reuse / total
        reduced_ltc_gate[2] += keep / total

    reduced_ltc_gate[0] /= len(gates)
    reduced_ltc_gate[1] /= len(gates)
    reduced_ltc_gate[2] /= len(gates)

    predictions = np.argmax(np.sum(data["preds"][:, 0], axis=1), axis=-1)

    est_flops = np.array([4.1725 for _ in range(len(records))])

    # TODO compute flops
    for m_i, mask in enumerate(gates):
        upsave = np.zeros_like(mask)
        for t in range(mask.shape[1] - 1):
            upsave[:, t] = (np.logical_and(mask[:, t, :] != 2, mask[:, t+1, :] != 1)).astype(int)
        upsave[:, -1] = (mask[:, t, :] != 2).astype(int)
        upsave = np.mean(upsave, axis=(1, 2))
        downsave = np.mean((mask[:,:,:]==0).astype(int), axis=(1,2))

        conv_offset = 0
        real_count = 1.

        layer_i = m_i
        up_flops = gflops_list[layer_i][0 + conv_offset]
        down_flops = gflops_list[layer_i][1 + conv_offset] * real_count
        est_flops = est_flops - upsave * up_flops - downsave * down_flops

    samples = est_flops

    tmp_queue = [(i, x) for i, x in enumerate(samples)]
    tmp_queue = sorted(tmp_queue, key=lambda x: x[1])
    print(tmp_queue[:topk], tmp_queue[-topk:])

    # save the figures and named with dataset_idx_name_idices
    for qi, (idx, flops) in enumerate(tmp_queue[:topk] + tmp_queue[-topk:]):
        the_img_path = records[idx]
        the_indices = indices[idx]
        the_dataset=path.split("_")[1]
        the_label_idx=data["targets"][idx, 0]
        classes_list=[x.strip() for x in open(classInds[the_dataset]).readlines()]
        the_label_word = classes_list[the_label_idx]
        for ind in the_indices:
            copyfile(ospj(dataset_paths[the_dataset], the_img_path, (prefixs[the_dataset]).format(ind)),
                     ospj(exp_path, "visual", "%s_%d_%d_%.2f_%.2f_%s_%d.jpg"%
                          (the_dataset, qi, idx, flops, flops/4.1725*100, the_label_word.replace(" ", "_"), ind)))
    return


for path in paths:
    draw(path)
