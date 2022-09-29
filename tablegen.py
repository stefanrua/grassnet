import pandas as pd
import os
import sys

folders = sys.argv[1:]

def split_name(folder):
    splits = folder.split('-')
    run = splits[1:-1]
    phase = splits[-1]
    if isinstance(run, list):
        run = '-'.join(run)
    if phase == 'train':
        phase = 'val'
    return run, phase

def read_nrmse(folder):
    res = pd.read_csv(f'{folder}/results.csv')
    return res.nrmse.item()

res = {'val':{}, 'test':{}}
for f in folders:
    run, phase = split_name(f)
    nrmse = read_nrmse(f)
    res[phase][run] = nrmse*100
table = pd.DataFrame(res)
table = table.sort_index()
table = table.to_markdown(floatfmt='.1f')
print(table)
