import os
import pandas as pd

imgdir     = 'images/bad-general/'
labels_in  = 'labels/full.csv'
labels_out = 'labels/bad-general.csv'

def getfid(fname):
    return int(fname.split('_id_')[1].split('.')[0])

df_in = pd.read_csv(labels_in, sep=';')[['id', 'DMY']].dropna()
df_out = pd.DataFrame()

fnames = os.listdir(imgdir)
fids = [getfid(f) for f in fnames]

dmy = [int(df_in[df_in['id'] == i]['DMY']) for i in fids]

df_out['image'] = fnames
df_out['dmy'] = dmy

print(df_out)

df_out.to_csv(labels_out, index=False)
