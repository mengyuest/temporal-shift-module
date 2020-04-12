import os,sys
from os.path import join as ospj
import socket
import getpass
host_name = socket.gethostname()
user_name = getpass.getuser()


# TODO(yue) path
# TODO WSC (power machines)
if "c699" in host_name and user_name in ["sunxm"]:
    ROOT_DIR="/gpfs/wscgpfs02/sunxm"
    CODE_PREFIX = "snippets/"
    DATA_PREFIX="data"
    LOG_PREFIX="snippets"
    PRETRAIN_PATH =ospj(ROOT_DIR, LOG_PREFIX,"logs_tsm")
# TODO WSC (power machines)
elif "c699" in host_name and user_name in ["cvpr"]:
    ROOT_DIR="/gpfs/wscgpfs02/cvpr"
    CODE_PREFIX = "code/"
    DATA_PREFIX="datasets"
    LOG_PREFIX=""
    PRETRAIN_PATH =ospj(ROOT_DIR, LOG_PREFIX,"logs_tsm")

# TODO DIVA [01-05]
elif "diva0" in host_name and user_name in ["meng", "cclin", "rameswar"]:
    ROOT_DIR="/store"
    CODE_PREFIX = "workspaces/%s"%user_name
    DATA_PREFIX="datasets"
    LOG_PREFIX = "workspaces/%s"%user_name
    PRETRAIN_PATH = ospj(ROOT_DIR, LOG_PREFIX, "logs_tsm")

# TODO CCC (power machines)
elif "dcc" in host_name and user_name in ["cvpr", "ieee", "sc071139"]:
    ROOT_DIR = "/dccstor/longxun"
    CODE_PREFIX = "../../u/cvpr"
    DATA_PREFIX = "datasets"
    #DATA_PREFIX = "../multimodalvideo"
    LOG_PREFIX = "../multimodalvideo/yue"
    PRETRAIN_PATH =ospj(ROOT_DIR,"logs_tsm")

#TODO satori (power machines)
elif ("node" in host_name or "service" in host_name )and user_name == "mengyue":
    ROOT_DIR = "/nobackup/users/mengyue"
    CODE_PREFIX = "code"
    DATA_PREFIX = "datasets"
    # DATA_PREFIX = "../multimodalvideo"
    LOG_PREFIX = ""
    PRETRAIN_PATH = ospj(ROOT_DIR, "logs_tsm")

#TODO AIMOS (power machines)
elif user_name == "DPLDymng":
    ROOT_DIR = "/gpfs/u/home/DPLD/DPLDymng/scratch"
    CODE_PREFIX = "code"
    DATA_PREFIX = "datasets"
    LOG_PREFIX = ""
    PRETRAIN_PATH = ospj(ROOT_DIR, "logs_tsm")

else:
    exit("unauthorized user@host '%s@%s'" % (user_name, host_name))

CODE_PATH=ospj(ROOT_DIR, CODE_PREFIX,"temporal-shift-module")
DATA_PATH = ospj(ROOT_DIR, DATA_PREFIX)
STHV2_PATH=ospj(DATA_PATH, "something/v2")
STHV2_FRAMES=ospj(STHV2_PATH, '20bn-something-something-v2-frames')

#TODO(yue) UCF101
UCF101_PATH=ospj(DATA_PATH, "UCF101")
if "dcc" in host_name:
    UCF101_PATH = UCF101_PATH.replace("longxun/datasets", "multimodalvideo")
UCF101_META_PATH = ospj(UCF101_PATH, "file_list")
UCF101_FRAMES=ospj(UCF101_PATH, 'frame')

#TODO(yue) HMDB51
HMDB51_PATH=ospj(DATA_PATH, "HMDB51")
HMDB51_META_PATH = ospj(HMDB51_PATH, "split")
HMDB51_FRAMES=ospj(HMDB51_PATH, 'frame')

#TODO(yue) activity-net-v1.3
ACTNET_PATH=ospj(DATA_PATH, "activity-net-v1.3")
if "dcc" in host_name:
    ACTNET_PATH = ACTNET_PATH.replace("longxun/datasets","multimodalvideo")
ACTNET_META_PATH = ACTNET_PATH
ACTNET_FRAMES=ospj(ACTNET_PATH, 'frames')

#TODO(yue) FCVID
FCVID_PATH=ospj(DATA_PATH, "fcvid")
if "dcc" in host_name:
    FCVID_PATH = FCVID_PATH.replace("longxun/datasets","multimodalvideo")
FCVID_META_PATH = FCVID_PATH
FCVID_FRAMES=ospj(FCVID_PATH, 'frames')

#TODO(yue) mini-something
MINISTH_PATH=ospj(DATA_PATH, "something2something-v2")
if "dcc" in host_name:
    MINISTH_PATH = MINISTH_PATH.replace("longxun/datasets","multimodalvideo")
MINISTH_META_PATH = MINISTH_PATH
MINISTH_FRAMES=ospj(MINISTH_PATH, 'frames')

#TODO(yue) mini-kinetics
MINIK_PATH=ospj(DATA_PATH, "kinetics-qf")
if "dcc" in host_name:
    MINIK_PATH = MINIK_PATH.replace("longxun/datasets","multimodalvideo")
MINIK_META_PATH = MINIK_PATH
if "diva" in host_name:
    MINIK_META_PATH = MINIK_PATH.replace("kinetics-qf","kinetics-100")
# print(MINIK_META_PATH)
MINIK_FRAMES=ospj(MINIK_PATH)

EXPS_PATH=ospj(ROOT_DIR, LOG_PREFIX,"logs_tsm")

from os.path import expanduser
PYTORCH_CKPT_DIR = ospj(expanduser("~"), ".cache/torch/checkpoints")