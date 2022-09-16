import matplotlib.pyplot as plt
import pandas as pd

#plt.style.use('ggplot')
dpi=150

def errcurve(rundir, save=True):
    errs = pd.read_csv(f'{rundir}nrmse.csv')
    plt.plot(errs['err_train'], label='nrmse_train', color='gray')
    plt.plot(errs['err_val'], label='nrmse_val', color='black')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('nrmse')
    if save:
        plt.savefig(f'{rundir}nrmsecurve.png', dpi=dpi)
        plt.close()
    else:
        plt.show()

def predictions(rundir, save=True):
    pred = pd.read_csv(f'{rundir}predictions.csv')
    plt.plot([0,10000], [0,10000], color='gray')
    plt.plot(pred['label'], pred['prediction'],
            color='black',
            linestyle='',
            marker='o',
            markersize='2',
            alpha=0.1)
    plt.xlabel('label')
    plt.ylabel('prediction')
    if save:
        plt.savefig(f'{rundir}predictions.png', dpi=dpi)
        plt.close()
    else:
        plt.show()

def labelhist(labelfile, rundir, save=True):
    b = 100
    r = (0,10000)
    full = pd.read_csv(labelfile)
    pred = pd.read_csv(f'{rundir}predictions.csv')
    plt.title('labels')
    plt.hist(full.iloc[:,1], bins=b, range=r, 
            color='lightgray',
            label='train+val')
    plt.hist(pred['label'], bins=b, range=r,
            color='gray',
            label='val')
    plt.hist(pred['prediction'], bins=b, range=r,
            color='black',
            label='pred')
    plt.legend()
    if save:
        plt.savefig(f'{rundir}labelhist.png', dpi=dpi)
        plt.close()
    else:
        plt.show()

def imshow(img, title=None):
    img = img.numpy().transpose((1, 2, 0))
    img = np.clip(img, 0, 1)
    plt.axis('off')
    plt.imshow(img)
    if title:
        plt.title(title, loc='center', wrap=True)
    plt.show()

