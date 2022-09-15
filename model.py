print('importing libraries...')

from PIL import Image
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import copy
import numpy as np
import os
import pandas as pd
import random
import signal
import timm
import torch
 
batch_size = 16
epochs = 5
learning_rate = 0.001
valsplit = 0.2
weight_decay = 0.01

imgdir = 'images/rgb/'
labelfile = 'labels/dmy.csv'
outdir = 'out/'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
imgsize = 224

def handler(signum, frame):
    res = input(' exit? y/n ')
    if res == 'y':
        if len(err_val) > 0: save_results()
        exit(1)

signal.signal(signal.SIGINT, handler)

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

class GrassDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        image = transforms.functional.pil_to_tensor(image)
        image = normalize_image(image)
        label = self.img_labels.iloc[idx, 1]
        label = normalize_label(label)
        if self.transform:
            image = self.transform(image)
        return image, label

def nrmse(true, pred):
    return mean_squared_error(true, pred) / np.mean(true)

def save_results():
    # runid = largest runid in outdir + 1
    if not os.path.exists(outdir):
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

def epoch(train):
    global err_train, err_val

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
    return err

def train():
    global w_best, err_best
    print('training...')
    for e in range(epochs):
        print(f'epoch {e+1}/{epochs}')
        epoch(train=True)
        err = epoch(train=False)
        if err < err_best:
            err_best = err
            w_best = copy.deepcopy(model.state_dict())

# data
transform_train = torch.nn.Sequential(
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-180, 180)),
        transforms.CenterCrop(imgsize),
        )

transform_val = torch.nn.Sequential(
        transforms.CenterCrop(imgsize),
        )
data_train = GrassDataset(labelfile, imgdir, transform_train)
data_val = GrassDataset(labelfile, imgdir, transform_val)
N = len(data_train)
indices = list(range(N))
split = int(N*valsplit)
random.shuffle(indices)
train_idx, val_idx = indices[split:], indices[:split]
sampler_train = SubsetRandomSampler(train_idx)
sampler_val = SubsetRandomSampler(val_idx)
dataloader_train = DataLoader(data_train,
        batch_size=batch_size,
        sampler=sampler_train)
dataloader_val = DataLoader(data_val,
        batch_size=batch_size,
        sampler=sampler_val)

# model
model = timm.create_model('vgg16_bn',
        num_classes=1,
        pretrained=True,
        in_chans=3)
model = model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay)

# results
w_best = copy.deepcopy(model.state_dict())
err_best = np.inf
err_train = []
err_val = []

train()
save_results()
