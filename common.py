import os, sys
from os.path import expanduser
from os.path import join as ospj
import socket
import getpass

host_name = socket.gethostname()
user_name = getpass.getuser()

# TODO check who you are and assign DATA_PATH, EXPS_PATH
if user_name in ["meng", "mengyue", "cvpr", "DPLDymng"]:
    import common_yue

    DATA_PATH, EXPS_PATH = common_yue.get_paths()

elif user_name in ["cclin", "DPLDchun"]:
    import common_cc

    DATA_PATH, EXPS_PATH = common_cc.get_paths()

elif user_name in ["rameswar", "DPLDpndr"]:
    import common_rpanda

    DATA_PATH, EXPS_PATH = common_rpanda.get_paths()


def inner_set_manual_data_path(data_path, exps_path):
    if data_path is not None:
        global DATA_PATH
        DATA_PATH = data_path

    if exps_path is not None:
        global EXPS_PATH
        EXPS_PATH = exps_path


def set_manual_data_path(data_path, exps_path):
    inner_set_manual_data_path(data_path, exps_path)

    global STHV2_PATH
    global STHV2_META_PATH
    global STHV2_FRAMES
    STHV2_PATH = ospj(DATA_PATH, "something2something-v2")
    STHV2_META_PATH = STHV2_PATH
    STHV2_FRAMES = ospj(STHV2_PATH, 'frames')
    # STHV2_PATH=ospj(DATA_PATH, "something/v2")
    # STHV2_META_PATH = STHV2_PATH
    # STHV2_FRAMES=ospj(STHV2_PATH, '20bn-something-something-v2-frames')

    # TODO(yue) UCF101
    global UCF101_PATH
    global UCF101_META_PATH
    global UCF101_FRAMES
    UCF101_PATH = ospj(DATA_PATH, "UCF101")
    UCF101_META_PATH = ospj(UCF101_PATH, "file_list")
    UCF101_FRAMES = ospj(UCF101_PATH, 'frame')

    # TODO(yue) HMDB51
    global HMDB51_PATH
    global HMDB51_META_PATH
    global HMDB51_FRAMES
    HMDB51_PATH = ospj(DATA_PATH, "HMDB51")
    HMDB51_META_PATH = ospj(HMDB51_PATH, "split")
    HMDB51_FRAMES = ospj(HMDB51_PATH, 'frame')

    # TODO(yue) activity-net-v1.3
    global ACTNET_PATH
    global ACTNET_META_PATH
    global ACTNET_FRAMES
    ACTNET_PATH = ospj(DATA_PATH, "activity-net-v1.3")
    ACTNET_META_PATH = ACTNET_PATH
    ACTNET_FRAMES = ospj(ACTNET_PATH, 'frames')

    # TODO(yue) FCVID
    global FCVID_PATH
    global FCVID_META_PATH
    global FCVID_FRAMES
    FCVID_PATH = ospj(DATA_PATH, "fcvid")
    FCVID_META_PATH = FCVID_PATH
    FCVID_FRAMES = ospj(FCVID_PATH, 'frames')

    # TODO(yue) mini-somethingV2
    global MINISTH_PATH
    global MINISTH_META_PATH
    global MINISTH_FRAMES
    MINISTH_PATH = ospj(DATA_PATH, "something2something-v2")
    MINISTH_META_PATH = MINISTH_PATH
    MINISTH_FRAMES = ospj(MINISTH_PATH, 'frames')

    # TODO(yue) charades
    global CHARADES_PATH
    global CHARADES_META_PATH
    global CHARADES_FRAMES
    CHARADES_PATH = ospj(DATA_PATH, "charades")
    CHARADES_META_PATH = CHARADES_PATH
    CHARADES_FRAMES = ospj(CHARADES_PATH, 'Charades_v1_rgb')

    # TODO(yue) epic
    global EPIC_PATH
    global EPIC_META_PATH
    global EPIC_FRAMES
    EPIC_PATH = ospj(DATA_PATH, "EPIC_KITCHENS_2018")
    EPIC_META_PATH = EPIC_PATH
    EPIC_FRAMES = ospj(EPIC_PATH, 'frames')

    # TODO(yue) mini-kinetics
    global MINIK_PATH
    global MINIK_META_PATH
    global MINIK_FRAMES
    MINIK_PATH = ospj(DATA_PATH, "kinetics-qf")
    MINIK_META_PATH = MINIK_PATH
    if "diva" in host_name:
        MINIK_META_PATH = MINIK_PATH.replace("kinetics-qf", "kinetics-100")
    MINIK_FRAMES = ospj(MINIK_PATH)

    # TODO (yue) jester
    global JESTER_PATH
    global JESTER_META_PATH
    global JESTER_FRAMES
    JESTER_PATH = ospj(DATA_PATH, "jester")
    JESTER_META_PATH = JESTER_PATH
    JESTER_FRAMES = ospj(JESTER_PATH, "20bn-jester-v1")
