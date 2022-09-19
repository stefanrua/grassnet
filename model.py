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
import sys
import timm
import torch
import matplotlib.pyplot as plt
import visualisation as vis

random.seed(0) # used for shuffling data before splitting to train/val

testing = False
show_examples = False
weight_file = None
batch_size = 16
epochs = 5
learning_rate = 0.001
valsplit = 0.2
weight_decay = 0.01
imgdir = 'images/subset1/'
labelfile = 'labels/subset1.csv'

i = 1
while i < len(sys.argv):
    match sys.argv[i]:
        case '--weights':
            weight_file = sys.argv[i+1]
            i += 2
        case '--batch-size':
            batch_size = int(sys.argv[i+1])
            i += 2
        case '--epochs':
            epochs = int(sys.argv[i+1])
            i += 2
        case '--learning-rate':
            learning_rate = float(sys.argv[i+1])
            i += 2
        case '--validation-split':
            valsplit = float(sys.argv[i+1])
            i += 2
        case '--weight-decay':
            weight_decay = float(sys.argv[i+1])
            i += 2
        case '--test':
            testing = True
            i += 1
        case '--show-examples':
            show_examples = True
            i += 1
        case '--image-dir':
            imgdir = sys.argv[i+1]
            i += 2
        case '--labels':
            labelfile = sys.argv[i+1]
            i += 2
        case _:
            print(f'unknown option: {sys.argv[i]}')
            exit(1)

outdir = 'out/'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
imgsize = 224

w_best = None
err_best = np.inf
errs = [[], []] # [[train], [val]]
pred = [[], []] # [[label], [pred]]

def handler(signum, frame):
    res = input(' exit? y/n ')
    if res == 'y':
        if w_best: save_results()
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

    with open(f'{rundir}options.txt', 'w') as f:
        f.write(' '.join(sys.argv))
    if err_best < np.inf:
        err_best_df = pd.DataFrame({'nrmse': [err_best]})
        err_best_df.to_csv(f'{rundir}results.csv', index=False)
    if w_best:
        torch.save(w_best, f'{rundir}w_best.pt')
    if len(errs[0]) > 0:
        err_df = pd.DataFrame({'err_train': errs[0], 'err_val': errs[1]})
        err_df.to_csv(f'{rundir}nrmsecurve.csv', index=False)
        vis.errcurve(rundir)
    if len(pred[0]) > 0:
        pred_df = pd.DataFrame({'label': pred[0], 'prediction': pred[1]})
        pred_df.to_csv(f'{rundir}predictions.csv', index=False)
        vis.predictions(rundir)
        if not testing: vis.labelhist(labelfile, rundir)

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
        label = self.img_labels.iloc[idx, 1]
        label = normalize_label(label)
        if self.transform:
            image = self.transform(image)
        image = transforms.functional.pil_to_tensor(image)
        image = normalize_image(image)
        return image, label

# returns nrmse, [[label], [prediction]]
def epoch(train):
    labels_ep = []
    outputs_ep = []

    if train:
        model.train()
        dataloader = dataloader_train
    else:
        model.eval()
        dataloader = dataloader_val

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).flatten()
        loss = criterion(outputs.float(), labels.float())                  
        if train:
            loss.backward()
            optimizer.step()

        labels_ep += list(labels.detach().cpu().numpy())
        outputs_ep += list(outputs.detach().cpu().numpy())

        if show_examples:
            img = images[0].detach().cpu()
            label = denormalize_label(labels[0].detach().cpu())
            pred = denormalize_label(outputs[0].detach().cpu())
            vis.imshow(img, f"pred: {pred:.0f}, label: {label:.0f}")
    
    err = nrmse(labels_ep, outputs_ep)
    if train:
        phase = 'train'
    elif testing:
        phase = 'test'
    else:
        phase = 'validation'
    print(f'{phase} nrmse: {err*100:.1f}%')
    labels_ep = [denormalize_label(l) for l in labels_ep]
    outputs_ep = [denormalize_label(l) for l in outputs_ep]
    predictions = [labels_ep, outputs_ep]
    return err, predictions

def train():
    print('training...')
    global w_best, err_best, errs, pred
    errs_train = []
    errs_val = []
    err_best = np.inf
    pred_best = []
    for e in range(epochs):
        print(f'epoch {e+1}/{epochs}')
        err_train, _ = epoch(train=True)
        err_val, pred_ep = epoch(train=False)
        errs[0].append(err_train)
        errs[1].append(err_val)
        if err_val < err_best:
            err_best = err_val
            w_best = copy.deepcopy(model.state_dict())
            pred = pred_ep

def test():
    print('calculating predictions...')
    global err_best, pred
    err_best, pred = epoch(train=False)

# data
transform_train = torch.nn.Sequential(
        transforms.RandomEqualize(1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-180, 180)),
        transforms.CenterCrop(imgsize),
        )
transform_val = torch.nn.Sequential(
        transforms.RandomEqualize(1),
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
print('loading model...')
model = timm.create_model('vgg16_bn',
        num_classes=1,
        pretrained=True,
        in_chans=3)
if weight_file:
    model.load_state_dict(torch.load(weight_file))
model = model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay)

# results
if testing:
    test()
    save_results()
else:
    train()
    save_results()
