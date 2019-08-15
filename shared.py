import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import empyrical as emp
import matplotlib.pyplot as plt
import seaborn as sns
cmap = sns.diverging_palette(220, 10, as_cmap=True)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from adamw import AdamW
from torchvision import datasets
import os, time, pdb, math, random
import _pickle as cPickle
from multiprocessing import Process, Queue, JoinableQueue
import torch.multiprocessing as _mp
mp = _mp.get_context('spawn')

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Import pickled data required for the training and testing
#X_train -> contains an observation of the environment, which consists of 512 features, at each round.
#y_train -> contains log returns returns of each asset within the portfolio at the end of each round
with open(root+'dataset/y_train.pkl', 'rb') as f: y_train = cPickle.load(f)
with open(root+'dataset/y_test.pkl', 'rb') as f: y_test = cPickle.load(f)

# Use AUDUSD, NZDUSD pairs and cash only as the portfolio
traincurs = []
testcurs = []
curr = [0, 3, 10]
base = [10, 10, 10]
for i in range(len(curr)):
    traincurs.append((y_train[:, curr[i]] - y_train[:, base[i]]).reshape(-1, 1))
    testcurs.append((y_test[:, curr[i]] - y_test[:, base[i]]).reshape(-1, 1))
y_train = np.concatenate(traincurs, axis=1)
y_test = np.concatenate(testcurs, axis=1)

y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)
with open(root+'dataset/X_train.pkl', 'rb') as f: X_train = cPickle.load(f)
with open(root+'dataset/X_test.pkl', 'rb') as f: X_test = cPickle.load(f)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
train_loader = torch.utils.data.DataLoader(train_dataset, 
    batch_size = 1, shuffle = False)
test_loader = torch.utils.data.DataLoader(test_dataset, 
    batch_size = 1, shuffle = False)

assets = ['AUDUSD', 'NZDUSD', 'USDUSD']

# Function for calculating risk-measures and plotting results
def plot_function(epoch_weights):
    ew = np.concatenate(epoch_weights).reshape(-1, len(assets))
    comm = np.sum(np.abs(ew[1:] - ew[:-1]), axis=1)
    ret = np.sum(np.multiply(ew, y_test.numpy()), axis=1)[1:]
    ind = pd.date_range("20180101", periods=len(ret), freq='H')
    ret = pd.DataFrame(ret - comm * cost, index = ind)
    ret.to_csv(root+'outputs/hourly_returns.csv')
    exp = np.exp(ret.resample('1D').sum()) - 1.0
    ggg = 'Drawdown:', emp.max_drawdown(exp).values[0], 'Sharpe:', emp.sharpe_ratio(exp)[0], \
    'Sortino:', emp.sortino_ratio(exp).values[0], 'Stability:', emp.stability_of_timeseries(exp), \
    'Tail:', emp.tail_ratio(exp), 'ValAtRisk:', emp.value_at_risk(exp)
    ttt = ' '.join(str(x) for x in ggg)
    print(ttt)
    plt.figure()
    np.exp(ret).cumprod().plot(figsize=(48, 12), title=ttt)
    plt.savefig(root+'graphs/cumulative_return', bbox_inches='tight', pad_inches=0)
    plt.close()
    ret = ret.resample('1W').sum()
    plt.figure(figsize=(48, 12))
    pal = sns.color_palette("Greens_d", len(ret))
    rank = ret.iloc[:,0].argsort()
    ax = sns.barplot(x=ret.index.strftime('%d-%m'), y=ret.values.reshape(-1), palette=np.array(pal[::-1])[rank])
    ax.text(0.5, 1.0, ttt, horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
    plt.savefig(root+'graphs/weekly_returns', bbox_inches='tight', pad_inches=0)
    plt.close()
    ew_df = pd.DataFrame(ew[1:], index = ind, columns =  assets)
    ew_df.to_csv(root+'outputs/portfolio_weights.csv')
    ew_df = ew_df.resample('1D').mean()
    #tr = np.diff(ew.T, axis=1)
    plt.figure(figsize=(48, 12))
    ax = sns.heatmap(ew_df.diff().T, cmap=cmap, center=0, robust=True, xticklabels=False)
    ax.text(0.5, 1.0, ttt, horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
    plt.savefig(root+'graphs/transactions', bbox_inches='tight', pad_inches=0)
    plt.close()
    ew_df = ew_df.resample('1D').mean()
    plt.figure(figsize=(48, 12))
    ax = sns.heatmap(ew_df.T, cmap=cmap, center=0, xticklabels=False, robust=True)
    ax.text(0.5, 1.0, ttt, horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
    plt.savefig(root+'graphs/portfolio_weights', bbox_inches='tight', pad_inches=0)
    plt.close()
