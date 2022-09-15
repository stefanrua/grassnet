print('importing libraries...')

from PIL import Image
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import copy
import numpy as np
import os
import pandas as pd
import random
import timm
import torch
import signal
 
def handler(signum, frame):
    res = input(" exit? y/n ")
    if res == 'y':
        if len(err_val) > 0: save_results()
        exit(1)
 
signal.signal(signal.SIGINT, handler)

batch_size = 16
epochs = 5
imgsize = 224
learning_rate = 0.001
target = 'DMY'
valsplit = 0.2
weight_decay = 0.01

device = "cuda:0" if torch.cuda.is_available() else "cpu"
imgdir = 'images/rgb/'
labelfile = 'labels/combined.csv'
outdir = 'out/'

# fname: filename
# returns: id in filename
def getfid(fname):
    return int(fname.split('_id_')[1].split('.')[0])

def normalize_label(label):
    return label/10000

def normalize_image(image):
    return image/255

def denormalize_label(label):
    return label*10000

def denormalize_image(image):
    return image*255

# returns: [(image, label)]
def load_data():
    res = []
    labels = pd.read_csv(labelfile, sep=';')
    for fname in os.listdir(imgdir):
        fid = getfid(fname)
        img = Image.open(f"{imgdir}{fname}").convert('RGB')
        img = transforms.functional.pil_to_tensor(img)
        img = transforms.functional.center_crop(img, imgsize)
        img = normalize_image(img)
        label = int(labels[labels['id'] == fid][target])
        label = normalize_label(label)
        res.append((img, label))
    return res

def nrmse(true, pred):
    return mean_squared_error(true, pred) / np.mean(true)

print('loading data...')
data = load_data()
split = int(len(data)*valsplit)
random.shuffle(data)
data_train = data[split:]
data_val = data[:split]
dataloader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)
dataloader_val = DataLoader(data_val, batch_size=batch_size, shuffle=True)
model = timm.create_model('vgg16_bn',
        num_classes=1,
        pretrained=True,
        in_chans=3)
model = model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay)

w_best = copy.deepcopy(model.state_dict())
err_best = np.inf
err_train = []
err_val = []
err = 0

def save_results():
    # runid = largest runid in outdir + 1
    if outdir not in os.listdir():
        os.mkdir(outdir)
    dirs = os.listdir(outdir)
    if len(dirs) == 0:
        runid = 0
    else:
        runid = max([int(x) for x in dirs]) + 1
    rundir = f'{outdir}{runid}/'
    os.mkdir(rundir)
    print(f'saving results to {rundir}')
    err_df = pd.DataFrame({'err_train': err_train, 'err_val': err_val})
    err_df.to_csv(f'{rundir}nrmse.csv', index=False)
    torch.save(w_best, f'{rundir}w_best.pt')

# train: bool
def epoch_sub(train):
    if train: model.train()
    else: model.eval()
    labels_ep = []
    outputs_ep = []

    for images, labels in dataloader_train:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs.float(), labels.float())                  
        if train:
            loss.backward()
            optimizer.step()

        labels_ep += list(labels.detach().cpu().numpy())
        outputs_ep += list(outputs.detach().cpu().numpy())
    
    err = nrmse(labels_ep, outputs_ep)
    if train: err_train.append(err)
    else: err_val.append(err)
    phase = 'train' if train else 'val'
    print(f'{phase} nrmse: {err:.3}')


print('training...')
for epoch in range(epochs):
    print(f"epoch {epoch+1}/{epochs}")
    epoch_sub(train=True)
    epoch_sub(train=False)
    if err < err_best:
        err_best = err
        w_best = copy.deepcopy(model.state_dict())

save_results()
