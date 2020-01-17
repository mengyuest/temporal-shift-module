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

# TODO DIVA [01-05]
elif "diva0" in host_name and user_name in ["meng", "cclin", "rameswar"]:
    ROOT_DIR="/store"
    CODE_PREFIX = "workspaces/%s"%user_name
    DATA_PREFIX="datasets"
    LOG_PREFIX = "workspaces/%s"%user_name

# TODO CCC (power machines)
elif "dccpc" in host_name and user_name in ["cvpr", "ieee", "sc071139"]:
    ROOT_DIR = "/dccstor/longxun"
    CODE_PREFIX = "../../u/cvpr"
    DATA_PREFIX = "datasets"
    #DATA_PREFIX = "../multimodalvideo"
    LOG_PREFIX = ""

else:
    exit("unauthorized user@host '%s@%s'" % (user_name, host_name))

CODE_PATH=ospj(ROOT_DIR, CODE_PREFIX,"temporal-shift-module")
DATA_PATH = ospj(ROOT_DIR, DATA_PREFIX)
STHV2_PATH=ospj(DATA_PATH, "something/v2")
STHV2_FRAMES=ospj(STHV2_PATH, '20bn-something-something-v2-frames')
UCF101_PATH=ospj(DATA_PATH, "UCF101")
UCF101_META_PATH = ospj(UCF101_PATH, "file_list")
UCF101_FRAMES=ospj(UCF101_PATH, 'frame')

ACTNET_PATH=ospj(DATA_PATH, "activity-net-v1.3")
if "dccpc" in host_name:
    ACTNET_PATH = ACTNET_PATH.replace("longxun/datasets","multimodalvideo")
ACTNET_META_PATH = ACTNET_PATH
ACTNET_FRAMES=ospj(ACTNET_PATH, 'frames')

#TODO(yue) FCVID
FCVID_PATH=ospj(DATA_PATH, "fcvid")
FCVID_META_PATH = FCVID_PATH
FCVID_FRAMES=ospj(FCVID_PATH, 'frames')

HMDB51_PATH=ospj(DATA_PATH, "HMDB51")
HMDB51_META_PATH = ospj(HMDB51_PATH, "split")
HMDB51_FRAMES=ospj(HMDB51_PATH, 'frame')

EXPS_PATH=ospj(ROOT_DIR, LOG_PREFIX,"logs_tsm")