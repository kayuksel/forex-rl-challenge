import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets
import os, time, pdb, math, random
import _pickle as cPickle

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

root = '/home/kamer/Desktop/fx/'
with open(root+'dataset/y_train.pkl', 'rb') as f: y_train = cPickle.load(f)
with open(root+'dataset/y_test.pkl', 'rb') as f: y_test = cPickle.load(f)
with open(root+'dataset/X_train.pkl', 'rb') as f: X_train = cPickle.load(f)
with open(root+'dataset/X_test.pkl', 'rb') as f: X_test = cPickle.load(f)
#X_train = X_train[:, :-3]
#X_test = X_test[:, :-3]
train_dataset = torch.utils.data.TensorDataset(X_train)
train_loader = torch.utils.data.DataLoader(train_dataset,
    batch_size = len(train_dataset), shuffle = True)
aud = y_test[:, 0]
aud -= np.mean(aud)
nzd = y_test[:, 3]
nzd -= np.mean(nzd)

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc0 = nn.Linear(512, 256)
        self.fc1 = nn.Linear(256, 128)
        self.fc21 = nn.Linear(128, 2)
        self.fc22 = nn.Linear(128, 2)
        self.fc3 = nn.Linear(2, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 512)
        self.gelu = GELU()
    def encode(self, x):
        h1 = self.gelu(self.fc0(x))
        h2 = self.gelu(self.fc1(h1))
        return self.fc21(h2), self.fc22(h2)
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    def decode(self, z):
        h3 = self.gelu(self.fc3(z))
        h4 = self.gelu(self.fc4(h3))
        return torch.sigmoid(self.fc5(h4))
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

model = VAE()
if torch.cuda.is_available(): model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-2)
reconstruction_function = nn.MSELoss(size_average=False)
def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    return BCE + torch.sum(KLD_element).mul_(-0.5)

for epoch in range(500):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        if torch.cuda.is_available():
            data = data[0].cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))

mus = model(X_test.cuda())[1]
feat = mus.cpu().detach().numpy()
plt.figure(figsize=(24, 12))
mus = np.concatenate([feat, aud.reshape(-1, 1)], axis=1)
pd.DataFrame(mus, columns=['x', 'y', 'c']).plot.scatter(
x='x', y='c', c='c', colormap='seismic', s=0.1, ylim=(-0.02, 0.02))
plt.savefig(root+'graphs/vae_aud')
plt.close()
plt.figure(figsize=(24, 12))
mus = np.concatenate([feat, nzd.reshape(-1, 1)], axis=1)
pd.DataFrame(mus, columns=['x', 'y', 'c']).plot.scatter(
x='x', y='c', c='c', colormap='seismic', s=0.1, ylim=(-0.02, 0.02))
plt.savefig(root+'graphs/vae_nzd')
plt.close()
