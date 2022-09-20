import os
import pandas as pd

imgdir     = 'images/rgb/'
labels_in  = 'labels/full.csv'
labels_out = 'labels/dvalue.csv'
target = 'D-value'

def getfid(fname):
    return int(fname.split('_id_')[1].split('.')[0])

df_in = pd.read_csv(labels_in, sep=';')[['id', target]].dropna()
df_out = pd.DataFrame()

c1 = (df_in['id'] < 30000) | (df_in['id'] >= 40000) # not rg test
c2 = (df_in['id'] < 50000) | (df_in['id'] >= 70000)  # not maaninka
df_in = df_in[c1 & c2]

fnames = os.listdir(imgdir)
fids = [getfid(f) for f in fnames]
fnames_by_fid = {}
for i in range(len(fnames)):
    fname = fnames[i]
    fid = fids[i]
    if fid in fnames_by_fid:
        fnames_by_fid[fid].append(fname)
    else:
        fnames_by_fid[fid] = [fname]

fnames_out = []
labels = []
for i, t in df_in[['id', target]].values:
    for f in fnames_by_fid[int(i)]:
        fnames_out.append(f)
        labels.append(t)

df_out['image'] = fnames_out
df_out[target] = labels

print(df_out)

df_out.to_csv(labels_out, index=False)
