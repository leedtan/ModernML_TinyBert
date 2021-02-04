NUM_TRIALS = 20

my_json = '{"access_token": null, "client_id": "32555940559.apps.googleusercontent.com", "client_secret": "ZmssLNjJy2998hD4CTg2ejr2", "refresh_token": "1//05jJdSPJetzqsCgYIARAAGAUSNwF-L9IrC9Z2KZtw_HEeZhH2aTBd8tpq3xmUCSbK_4ZBvRXMzpeQh7R28VjLV3laqoigF0qA4Rc", "token_expiry": null, "token_uri": "https://oauth2.googleapis.com/token", "user_agent": "Python client library", "revoke_uri": "https://oauth2.googleapis.com/revoke", "id_token": null, "id_token_jwt": null, "token_response": null, "scopes": [], "token_info_uri": null, "invalid": false, "_class": "GoogleCredentials", "_module": "oauth2client.client"}'

import numbers
import os
import pdb

import gspread
import numpy as np
import pandas as pd
from oauth2client.client import GoogleCredentials

gc = gspread.authorize(GoogleCredentials.from_json(my_json))
wb = gc.open_by_url(
    "https://docs.google.com/spreadsheets/d/1F8A5n3WVr9WXHDXZhv2w0BPaMC8uymUyxEWdI9UX6xg/edit#gid=0"
)

sheet = wb.worksheet("results")


def get_df(sheet):
    df = pd.DataFrame(sheet.get_all_values())
    df.columns = df.iloc[0, :]
    df.index = df.iloc[:, 0]
    df = df.iloc[1:, 1:]
    return df


def get_row_num(df, sheet, paramnames):
    if paramnames in df.index:
        rowidx = df.index.to_list().index(paramnames) + 1
    else:
        current_max_row = len(sheet.get_all_values())
        rowidx = int(current_max_row) + 1
    return rowidx


def check_run(paramnames, task, df=None, sheet=None):
    if 1:  # df is None:
        df = get_df(sheet)
    if paramnames in df.index:
        idx = df.columns.to_list().index(task)
        row = df.loc[paramnames, :]
        rowiloc = row.iloc[idx + 1]
        if rowiloc == "":
            return True
        if not isinstance(rowiloc, numbers.Number):
            if isinstance(rowiloc, str) and rowiloc.isnumeric():
                rowiloc = int(rowiloc)
            else:
                return False
        if rowiloc >= NUM_TRIALS:
            return False
    return True


def run(paramnames, task, df, sheet, params={}):
    df = get_df(sheet)
    if paramnames not in df.index:
        row_idx = get_row_num(df, sheet, paramnames)
        sheet.update("A" + str(row_idx), [[paramnames]])
        df = get_df(sheet)
    row_idx = get_row_num(df, sheet, paramnames)
    df = get_df(sheet)
    idx = df.columns.to_list().index(task)
    count = df.iloc[row_idx - 1, idx + 1]
    if count == "":
        count = 0
        avg = 0
    avg = df.iloc[row_idx - 1, idx]
    if avg == "":
        avg = 0
    avg = float(avg)
    count = int(count)

    all_datasets = ["rte", "sts-b", "mrpc", "cola"]
    metrics = ["test_acc", "test_pearson", "test_acc", "test_mcc"]
    metrics = dict(zip(all_datasets, metrics))
    result_key = metrics[args.task_name.lower()]

    args.seed = seed = count + 1
    args.output_dir = (
        output_dir + "_DATASET_" + args.task_name.lower() + "_SEED_" + str(seed)
    )
    results = run_glue_main(args)
    print(type(results))
    print(results)
    if "acc" in results:
        score = results["acc"]
    elif "pearson" in results:
        score = results["pearson"]
    print(
        "avg",
        type(avg),
        avg,
        "count",
        type(count),
        count,
        "score",
        type(score),
        score,
    )
    newavg = (avg * count + score) / (count + 1)
    sheet.update(str(chr(ord("a") + idx + 2)) + str(row_idx + 1), [[count + 1]])
    sheet.update(str(chr(ord("a") + idx + 1)) + str(row_idx + 1), [[newavg]])

    return True


def simulate(paramnames, tasks, df=None, sheet=None, params={}):
    if df is None:
        df = get_df(sheet)
    for task in tasks:
        i = 0
        while check_run(paramnames, task, df, sheet) and i < NUM_TRIALS:
            i += 1
            run(paramnames, task, df, sheet, params=params)
            df = get_df(sheet)


def dict2obj(d):
    if isinstance(d, list):
        d = [dict2obj(x) for x in d]
    if not isinstance(d, dict):
        return d

    class C(object):
        def __init__(self):
            pass

        def __call__(self):
            pass

        pass

    o = C()
    for k in d:
        o.__dict__[k] = dict2obj(d[k])
    return o


args = {
    "model_type": "bert",
    "model_name_or_path": "bert-large-uncased",
    "task_name": "RTE",
    "do_train": True,
    "data_dir": "/content/drive/My Drive/mixout/ModernML_TinyBert/glue_data",
    "max_seq_length": 64,
    "per_gpu_eval_batch_size": 8,
    "weight_decay": 0,
    "seed": 1,
    "overwrite_output_dir": True,
    "do_lower_case": True,
    "per_gpu_train_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "logging_steps": 0,
    "num_loggings": 10,
    "save_steps": 0,
    "test_val_split": True,
    "use_torch_adamw": True,
    "cache_dir": "/content/drive/My Drive/mixout/ModernML_TinyBert/hf-transformers-cache",
    "num_train_epochs": 3.0,
    "warmup_ratio": 0.1,
    "learning_rate": 2e-05,
    "output_dir": "tests/PAPER/BASELINE2",
    "all_datasets": True,
    "reinit_pooler": True,
    "normalize": True,
    "mixout_layers": 12,
    "reinit_layers": 0,
    "mixout": 0.3,
    "trials": 30,
    "l2_scaling": 0,
}

args["mixout_layers"] = 6
args["finetune_layers"] = 0
args["reinit_layers"] = 6
args["l2_reg_decay"] = 1
args["l2_reg_mult"] = 1e-2

for name, default_val in zip(
    [
        "data_dir",
        "model_type",
        "model_name_or_path",
        "task_name",
        "output_dir",
        "config_name",
        "tokenizer_name",
        "cache_dir",
        "max_seq_length",
        "do_train",
        "do_lower_case",
        "save_best",
        "save_last",
        "train_batch_size",
        "per_gpu_train_batch_size",
        "per_gpu_eval_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "layerwise_learning_rate_decay",
        "weight_decay",
        "adam_epsilon",
        "max_grad_norm",
        "num_train_epochs",
        "max_steps",
        "warmup_steps",
        "warmup_ratio",
        "weight_logging_steps",
        "logging_steps",
        "num_loggings",
        "save_steps",
        "no_cuda",
        "overwrite_output_dir",
        "overwrite_cache",
        "seed",
        "fp16",
        "fp16_opt_level",
        "local_rank",
        "server_ip",
        "server_port",
        "use_bertadam",
        "use_torch_adamw",
        "downsample_trainset",
        "resplit_val",
        "reinit_layers",
        "mixout_layers",
        "unfreeze_after_epoch",
        "reinit_pooler",
        "l2_scaling",
        "normalize",
        "all_datasets",
        "layer_mixout",
        "rezero_layers",
        "mixout",
        "mixout_decay",
        "trials",
        "prior_weight_decay",
        "test_val_split",
        "frozen_layers",
        "finetune_layers",
        "l2_reg_decay",
        "l2_reg_mult",
    ],
    [
        None,
        None,
        None,
        None,
        None,
        "",
        "",
        "",
        128,
        False,
        False,
        False,
        False,
        0,
        8,
        8,
        1,
        5e-5,
        1.0,
        0.0,
        1e-8,
        1.0,
        3.0,
        -1,
        0,
        0,
        10,
        0,
        0,
        500,
        False,
        False,
        False,
        42,
        False,
        "01",
        -1,
        "",
        "",
        False,
        False,
        -1,
        0,
        0,
        0,
        0,
        False,
        False,
        False,
        False,
        False,
        0,
        0.0,
        1.0,
        NUM_TRIALS,
        False,
        False,
        0,
        0,
        1.0,
        3e-3,
    ],
):
    if name not in args:
        args[name] = default_val

import os

from options import get_parser
from run_glue import main as run_glue_main

args = dict2obj(args)
output_dir = args.output_dir
data_dir = args.data_dir


def experiment(seeds):
    for seed in seeds:

        all_datasets = ["rte", "sts-b", "mrpc", "cola"]
        metrics = ["test_acc", "test_pearson", "test_acc", "test_mcc"]
        metrics = dict(zip(all_datasets, metrics))
        result_key = metrics[args.task_name.lower()]

        args.seed = seed
        args.output_dir = (
            output_dir + "_DATASET_" + args.task_name.lower() + "_SEED_" + str(seed)
        )
        results = run_glue_main(args)

        score = results[result_key]


# DATASETS = ["RTE", "MRPC", "STS-B"]
DATASETS = ["RTE"]


# for dataset in DATASETS:
#     seeds = range(args.trials)
#     args.task_name = dataset
#     args.data_dir = os.path.join(data_dir, args.task_name)
#     experiment(seeds)
def run_real(paramnames, tasks, df=None, sheet=None, params={}):
    if df is None:
        df = get_df(sheet)
    for task in tasks:
        args.task_name = task
        # params.data_dir = os.path.join(data_dir, args.task_name)
        args.data_dir = os.path.join(data_dir, args.task_name)
        i = 0
        while check_run(paramnames, task, df, sheet) and i < NUM_TRIALS:
            i += 1
            run(paramnames, task, df, sheet, params=params)
            df = get_df(sheet)


lnum = [
    args.frozen_layers,
    args.mixout_layers,
    args.finetune_layers,
    args.reinit_layers,
]
paramnames = (
    "Lee1_lay_"
    + str(lnum[0])
    + "_"
    + str(lnum[1])
    + "_"
    + str(lnum[2])
    + "_"
    + str(lnum[3])
    + "_reg_"
    + str(args.l2_reg_mult)
    + "_regdecay_"
    + str(args.l2_reg_decay)
    + "_ep_"
    + str(args.num_train_epochs)
    + "_decay_"
    + str(args.l2_reg_decay)
)
run_real(paramnames, DATASETS, df=None, sheet=sheet, params=args)
