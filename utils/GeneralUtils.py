import os
import numpy as np
import subprocess
from importlib.machinery import SourceFileLoader
import json
from itertools import combinations
import math
import functools
import logging
import traceback
from collections.abc import MutableMapping
from typing import Union, Iterable
from pandas import DataFrame, read_csv, concat, json_normalize
import warnings
from sklearn.metrics import matthews_corrcoef
import torch

# from os.path import join as opj
# from shutil import copyfile, SameFileError
# import git  # pip install gitpython
# from sqlalchemy import create_engine


class CollectErrors:
    """
    This is meant to be used as a decorator around functions that you
    first try to run but if any errors arise, you collect them.

    See: https://realpython.com/primer-on-python-decorators/
    See: https://stackoverflow.com/questions/30904486/ ...
        python-wrapper-function-taking-arguments-inside-decorator
    """

    def __init__(self, logger=None, monitor="", debug=False):
        self.msgs = []
        self.logger = logger or logging.getLogger(__name__)
        self.monitor = monitor
        self._debug = debug

    def reset(self):
        self.msgs = []

    def __call__(self):
        def outer_wrapper(func):

            @functools.wraps(func)
            def inner_wrapper(*args, **kwargs):

                # maybe we don't want to catch the exception (debug)!
                if self._debug:
                    return func(*args, **kwargs)

                # otherwise try to run the function else log exception
                try:
                    return func(*args, **kwargs)
                except Exception as err:
                    self.logger.error(func.__name__ + "(): " + err.__repr__())
                    self.msgs.append(
                        {
                            "monitor": self.monitor,
                            "func": func.__name__,
                            "traceback": traceback.format_exception(
                                None, err, err.__traceback__
                            ),
                        }
                    )

            return inner_wrapper

        return outer_wrapper


def drop_duplicate_indices_for_df(df: DataFrame, keep="first") -> DataFrame:
    return df[~df.index.duplicated(keep=keep)]


def append_row_to_df_or_create_it(where: str, df: DataFrame):
    """If a Dataframe exists, append row to it, otherwise create anew.

    Parameters
    ----------
    where: str
        Path to dataframe.
    df: DataFrame
        a (1, n_columns) dataframe row to be appended.

    Returns
    -------
    None

    """
    if not os.path.isfile(where):
        df.to_csv(where)
        return

    # columns in saved df but not this row
    ordered_existing_cols = list(read_csv(where, nrows=1, index_col=0).columns)
    existing_cols = set(ordered_existing_cols)
    cols_to_append = set(df.columns)
    missing_cols = existing_cols.difference(cols_to_append)
    df.loc[:, missing_cols] = np.nan

    # columns in this row but not in saved df
    extra_cols = list(cols_to_append.difference(existing_cols))

    # important: preserve column order
    df = df.loc[:, ordered_existing_cols + extra_cols]

    if len(extra_cols) == 0:
        # just append to existing df
        df.to_csv(where, header=False, mode="a")
    else:
        # read, concat, then save whole thing
        df = concat([read_csv(where, index_col=0), df], axis=0)
        df.to_csv(where)


def calculate_mcc(truth, pred):
    """
    This is a wrapper around sklearn.metrics.mathews_corrcoef
    that returns nan when it's not possible to calulate mcc instead of 0.
    """
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("error", category=RuntimeWarning)
        try:
            return matthews_corrcoef(truth, pred)
        except RuntimeWarning:
            return np.nan


# noinspection PyPep8Naming
def calculate_4x4_statistics(TP, FP, FN, TN=None, add_eps_to_tn=True):
    """Calculate simple stistics"""
    ep = 1e-10
    if TP == 0:
        TP += ep
    if FP == 0:
        FP += ep
    if FN == 0:
        FN += ep
    TN = 0 if TN is None else TN

    stats = {"total": TP + FP + FN + TN}
    stats.update(
        {
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "accuracy": (TP + TN) / stats["total"],
            "precision": TP / (TP + FP),
            "recall": TP / (TP + FN),
        }
    )
    # add synonyms
    stats.update(
        {
            "TPR": stats["recall"],
            "sensitivity": stats["recall"],
            "F1": (2 * stats["precision"] * stats["recall"])
            / (stats["precision"] + stats["recall"]),
        }
    )
    if TN >= 0:
        if TN == 0:
            if add_eps_to_tn:
                TN += ep
            else:
                return stats
        stats.update({"TN": TN, "specificity": TN / (TN + FP)})
        # add synonyms
        stats["TNR"] = stats["specificity"]

        # mathiew's correlation coefficient
        numer = TP * TN - FP * FN
        denom = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        stats["MCC"] = numer / denom

    return stats


def flatten_dict(d: MutableMapping, sep: str = ".") -> MutableMapping:
    """Flatten a dictionary so that there are no nested dicts.

    Source:
    https://www.freecodecamp.org/news/how-to-flatten-a-dictionary-in-python-in-4-different-ways/

    Parameters
    ----------
    d: MutableMapping
        A dict with (potentially) some nested dicts.
    sep: str
        How to separate parent and child keys from parent and child dicts.
        For example {'a': 1, 'b': {'p': 2, 'q': 3}}
        would become {'a': 1, 'b.p': 2, 'b.q': 3}

    Returns
    -------
    MutableMapping

    """
    [flat_dict] = json_normalize(d, sep=sep).to_dict(orient="records")

    return flat_dict


def splitlist(lst, sz):
    """
    Split list into equal-ish portions each of size sz.
    https://stackoverflow.com/questions/4119070/how-to-divide-a-list-into-n-equal-parts-python
    """
    return [lst[i : i + sz] for i in range(0, len(lst), sz)]


def _divnonan(numer, denom):
    return 0 if denom < 1e-8 else numer / denom


# noinspection PyUnresolvedReferences
def unique_nonzero(arr: Iterable):
    unq = np.unique(arr).tolist()
    if 0 in unq:
        unq.remove(0)
    return unq


def combs_with_unique_products(low, high, k):
    prods = set()
    for comb in combinations(range(low, high), k):
        prod = np.prod(comb)
        if prod not in prods:
            yield comb
            prods.add(prod)


def save_json(what, path: str, mode="w", indent=4):
    with open(path, mode) as f:
        json.dump(what, f, indent=indent)


def load_json(path: str):
    with open(path) as f:
        what = json.load(f)
    return what


def write_or_append_json_list(what: dict, path: str, indent=4):
    """Append a dict to a saved json list on disk.

    This implimentation is specifically geared to prevent loading the entire
    file contents just to append a single element to the list, while still
    preseving the json file as a valid json format.

    Modified from:
      https://stackoverflow.com/questions/18087397/ ...
      append-list-of-python-dictionaries-to-a-file-without-loading-it/18088275
    """
    if not os.path.isfile(path):
        save_json([what], path=path, indent=indent)
        return
    with open(path, mode="r+") as file:
        # assumes existing json is a list to which we will append this item
        file.seek(0, 2)
        position = file.tell() - 2
        file.seek(position)
        tmp = json.dumps(what, indent=indent)
        ind = indent * " "
        tmp = ind + tmp.replace("\n", f"\n{ind}")
        file.write(",\n{}\n]".format(tmp))


def normalize_to_zero_one_range(x: Union[DataFrame, np.ndarray]):
    return (x - x.min()) / (x.max() - x.min())


def weighted_avg_and_std(values, weights):
    """Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    Source: https://stackoverflow.com/questions/2413522/ ...
        weighted-standard-deviation-in-numpy
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights)
    return average, math.sqrt(variance)


def _div(numer, denom):
    """Avoid ZeroDivisionError or RuntimeWarning."""
    if denom < 1e-9:
        return np.nan
    return numer / denom


def rmse(ypred, ytrue):
    """Root mean square error (numpy)."""
    return np.sqrt(np.mean((ypred - ytrue) ** 2))


def abserr(ypred, ytrue):
    """Absolute error (numpy)."""
    return np.abs(ypred - ytrue)


def load_configs(configs_path, assign_name="cfg"):
    """See: https://stackoverflow.com/questions/67631/how-to-import-a- ...
    ... module-given-the-full-path"""
    # noinspection PyArgumentList
    return SourceFileLoader(assign_name, configs_path).load_module()


# def save_configs(configs_path, results_path, warn=True):
#     """ save a copy of config file and last commit hash for reproducibility
#     see: https://stackoverflow.com/questions/14989858/ ...
#     get-the-current-git-hash-in-a-python-script
#     """
#     savename = opj(results_path, os.path.basename(configs_path))
#     if warn and os.path.exists(savename):
#         input(
#             f"This will OVERWRITE: {savename}\n"
#             "Are you SURE you want to continue?? (press Ctrl+C to abort)"
#         )
#     try:
#         copyfile(configs_path, savename)
#     except SameFileError:
#         pass
#     repo = git.Repo(search_parent_directories=True)
#     with open(opj(results_path, "last_commit_hash.txt"), 'w') as f:
#         f.write(repo.head.object.hexsha)


def reverse_dict(d, preserve=False):
    if not preserve:
        # only return one key is two keys share the same value
        return {v: k for k, v in d.items()}
    else:
        # values become list of keys that shared same value in original dict
        new = {}
        for k, v in d.items():
            if v not in new:
                new[v] = [k]
            else:
                new[v].append(k)
        return new


def ordered_vals_from_ordered_dict(d):
    vs = []
    for v in d.values():
        if v not in vs:
            vs.append(v)
    return vs


# def connect_to_sqlite(db_path: str):
#     sql_engine = create_engine('sqlite:///' + db_path, echo=False)
#     return sql_engine.connect()


def maybe_mkdir(folder):
    os.makedirs(folder, exist_ok=True)


# noinspection PyPep8Naming
def isGPUDevice():
    """Determine if device is an NVIDIA GPU device"""
    return os.system("nvidia-smi") == 0


# noinspection PyPep8Naming
def AllocateGPU(N_GPUs=1, GPUs_to_use=None, TOTAL_GPUS=4, verbose=True, N_trials=0):
    """Restrict GPU use to a set number or name.
    Args:
        N_GPUs - int, number of GPUs to restrict to.
        GPUs_to_use - optional, list of int ID's of GPUs to use.
                    if none, this will fetch GPU's with lowest
                    memory consumption
        verbose - bool, print to screen?
    """
    # only restrict if not a GPU machine or already restricted
    isGPU = isGPUDevice()

    assert TOTAL_GPUS == 4, "Only 4-GPU machines supported for now."

    try:
        AlreadyRestricted = os.environ["CUDA_VISIBLE_DEVICES"] is not None
    except KeyError:
        AlreadyRestricted = False

    if isGPU and (not AlreadyRestricted):
        try:
            if GPUs_to_use is None:

                if verbose:
                    print("Restricting GPU use to {} GPUs ...".format(N_GPUs))

                # If you did not specify what GPU to use, this will just
                # fetch the GPUs with lowest memory consumption.

                # Get processes from nvidia-smi command
                gpuprocesses = str(
                    subprocess.check_output("nvidia-smi", shell=True)
                ).split("\\n")
                # Parse out numbers, representing GPU no, PID and memory use
                start = 24
                gpuprocesses = gpuprocesses[start : len(gpuprocesses) - 2]
                gpuprocesses = [j.split("MiB")[0] for i, j in enumerate(gpuprocesses)]

                # Add "fake" zero-memory processes to represent all GPUs
                extrapids = np.zeros([TOTAL_GPUS, 3])
                extrapids[:, 0] = np.arange(TOTAL_GPUS)

                PIDs = []
                for p in range(len(gpuprocesses)):
                    pid = [int(s) for s in gpuprocesses[p].split() if s.isdigit()]
                    if len(pid) > 0:
                        PIDs.append(pid)
                # PIDs.pop(0)
                PIDs = np.array(PIDs)

                if len(PIDs) > 0:
                    PIDs = np.concatenate((PIDs, extrapids), axis=0)
                else:
                    PIDs = extrapids

                # Get GPUs memory consumption
                memorycons = np.zeros([TOTAL_GPUS, 2])
                for gpuidx in range(TOTAL_GPUS):
                    thisgpuidx = 1 * np.array(PIDs[:, 0] == gpuidx)
                    thisgpu = PIDs[thisgpuidx == 1, :]
                    memorycons[gpuidx, 0] = gpuidx
                    memorycons[gpuidx, 1] = np.sum(thisgpu[:, 2])

                # sort and get GPU's with lowest consumption
                memorycons = memorycons[memorycons[:, 1].argsort()]
                GPUs_to_use = list(np.int32(memorycons[0:N_GPUs, 0]))

            # Now restrict use to available GPUs
            gpus_list = GPUs_to_use.copy()
            GPUs_to_use = ",".join([str(j) for j in GPUs_to_use])
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
            os.environ["CUDA_VISIBLE_DEVICES"] = GPUs_to_use
            if verbose:
                print("Restricted GPU use to GPUs: " + GPUs_to_use)
            return gpus_list

        except ValueError:
            if N_trials < 2:
                if verbose:
                    print("Got value error, trying again ...")
                N = N_trials + 1
                AllocateGPU(N_GPUs=N_GPUs, N_trials=N)
            else:
                raise ValueError("Something is wrong, tried too many times and failed.")

    else:
        if verbose:
            if isGPU:
                print("No GPU allocation done.")
            if AlreadyRestricted:
                print("GPU devices already allocated.")


# noinspection PyPep8Naming
def Merge_dict_with_default(
    dict_given: dict, dict_default: dict, keys_Needed: list = None
):
    """Sets default values of dict keys not given"""

    keys_default = list(dict_default.keys())
    keys_given = list(dict_given.keys())

    # Optional: force user to unput some keys (eg. those without defaults)
    if keys_Needed is not None:
        for j in keys_Needed:
            if j not in keys_given:
                raise KeyError("Please provide the following key: " + j)

    keys_Notgiven = [j for j in keys_default if j not in keys_given]

    for j in keys_Notgiven:
        dict_given[j] = dict_default[j]

    return dict_given


def file_len(fname: str):
    """
    Given a filename, get number of lines it has efficiently. See:
    https://stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python
    """
    try:
        p = subprocess.Popen(
            ["wc", "-l", fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        result, err = p.communicate()
        if p.returncode != 0:
            raise IOError(err)
        return int(result.strip().split()[0])

    except FileNotFoundError:
        # on windows systems where subprocess and file paths are weird
        with open(fname) as fp:
            count = 0
            for _ in fp:
                count += 1
        return count


def kill_all_nvidia_processes():
    """
    Force kills all NVIDIA processes, even if they don't
    show up in NVIDIA_SMI (common when tensorflow gets crazy
    and you kill the kernel or it dies).
    """

    input(
        "Killing all gpu processes .. continue?"
        + "Press any button to continue, or Ctrl+C to quit ..."
    )

    # get gpu processes -- note that this
    # gets processes even if they don't show up
    # in the nvidia-smi command (which happens
    # often with tensorflow)
    gpuprocesses = str(
        subprocess.check_output("fuser -v /dev/nvidia*", shell=True)
    ).split("\\n")

    # preprocess process list
    gpuprocesses = gpuprocesses[0].split(" ")[1:]
    if "'" in gpuprocesses[-1]:
        gpuprocesses[-1] = gpuprocesses[-1].split("'")[0]

    # put into string form
    gpuprocesses_str = "{"
    for pr in gpuprocesses:
        gpuprocesses_str += str(pr) + ","
    gpuprocesses_str += "}"

    # now kill
    kill_command = "kill -9 %s" % gpuprocesses_str
    os.system(kill_command)

    print("killed the following processes: " + gpuprocesses_str)


def move_to_cpu_recursive(obj):
    """Recursively move tensors to CPU and convert to numpy arrays."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    elif isinstance(obj, dict):
        return {k: move_to_cpu_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_cpu_recursive(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_cpu_recursive(v) for v in obj)
    else:
        return obj
