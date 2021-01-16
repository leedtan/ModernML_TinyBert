import os
import sys

import pandas as pd

args = sys.argv[1:]
directory = args[0]
start_name = args[1]
try:
    datasets = args[2].split(",")
except:
    datasets = ["rte", "sts-b", "mrpc", "cola"]

files = [x for x in os.listdir(directory) if x.startswith(start_name)]

all_datasets = ["rte", "sts-b", "mrpc", "cola"]
metrics = ["test_acc", "test_pearson", "test_acc", "test_mcc"]
metrics = dict(zip(all_datasets, metrics))

pd_hold = []

for dataset in datasets:
    metric = metrics[dataset]
    for run in [x for x in files if "DATASET_" + dataset in x]:
        results = os.path.join(directory, run, "test_best_log.txt")
        try:
            result = pd.read_csv(results)[metric][0]
            pd_hold.append([dataset, result])
        except:
            continue

df = pd.DataFrame(pd_hold, columns=["dataset", "test_result"])
output = df.groupby("dataset").describe()
print(output)

# output.to_csv('recent_test.csv')
