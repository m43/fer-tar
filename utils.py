import os
import pathlib
import random
from datetime import datetime

import numpy as np
import torch

project_path = pathlib.Path(__file__).parent


# project_path = pathlib.Path(__file__).parent.parent


class Object(object):
    pass


def get_str_formatted_time() -> str:
    return datetime.now().strftime('%Y.%m.%d_%H.%M.%S')


def ensure_dir(dirname):
    dirname = pathlib.Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def ensure_dirs(dirs):
    for dir_ in dirs:
        ensure_dir(dir_)


def zipdir(path, ziph):
    """
    Usage example:
    zipf = zipfile.ZipFile(results_path + ".zip", 'w', zipfile.ZIP_DEFLATED)
    zipdir(results_path, zipf)
    zipf.close()

    Source: https://stackoverflow.com/questions/41430417/using-zipfile-to-create-an-archive

    :param path: Path to dir to zip
    :param ziph: zipfile handle
    :return:
    """
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(path, '..')))


def setup_torch_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_torch_device(print_logs=True):
    device = torch.device("cuda") if torch.cuda.device_count() else torch.device("cpu")
    device_ids = list(range(torch.cuda.device_count()))

    if print_logs:
        print(f"torch.cuda.device_count={torch.cuda.device_count()}")
        print(f"device_ids={device_ids}")
        print(f"Using device '{device}'\n")

    return device
