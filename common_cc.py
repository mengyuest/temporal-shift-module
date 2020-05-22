from os.path import join as ospj
import socket
import getpass
host_name = socket.gethostname()
user_name = getpass.getuser()

def get_paths():
    # TODO(yue) path
    # TODO WSC (power machines)
    if "c699" in host_name:
        ROOT_DIR = "/gpfs/wscgpfs02/cvpr"
        DATA_PREFIX = "datasets"
        LOG_PREFIX = ""

    # TODO DIVA [01-05]
    elif "diva0" in host_name:
        ROOT_DIR = "/store"
        DATA_PREFIX = "datasets"
        LOG_PREFIX = "workspaces/%s" % user_name

    # TODO CCC (power machines)
    elif "dcc" in host_name:
        ROOT_DIR = "/dccstor/multimodalvideo"
        DATA_PREFIX = ""
        LOG_PREFIX = "yue"

    # TODO satori (power machines)
    elif "node" in host_name or "service" in host_name:
        ROOT_DIR = "/nobackup/users/mengyue"
        DATA_PREFIX = "datasets"
        LOG_PREFIX = ""

    # TODO AIMOS (power machines)
    elif "DPLD" in user_name:
        ROOT_DIR = "/gpfs/u/home/DPLD/DPLDymng/scratch"
        DATA_PREFIX = "datasets"
        LOG_PREFIX = ""

    else:
        exit("unauthorized user@host '%s@%s'" % (user_name, host_name))

    DATA_PATH = ospj(ROOT_DIR, DATA_PREFIX)
    EXPS_PATH = ospj(ROOT_DIR, LOG_PREFIX, "logs_tsm")

    return DATA_PATH, EXPS_PATH