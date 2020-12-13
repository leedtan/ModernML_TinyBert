import sys
import os
import pandas as pd
args = sys.argv[1:]
directory = args[0]
start_name = args[1]
try:
    accuracy = args[2]
except:
    accuracy = "test_acc"

files = [x for x in os.listdir(directory) if x.startswith(start_name)]

pd_hold = []

for run in files:
    results = os.path.join(directory,run,'test_best_log.txt')
    try:
        pd_hold.append(pd.read_csv(results))
    except:
        continue

df = pd.concat(pd_hold)
print(df[accuracy].describe())
