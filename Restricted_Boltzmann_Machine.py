import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim


def load_data(filename):
    '''
  takes as input the filename and
  returns a tuple of (X,y) where X is of shape(n_samples,n_features)
  and y is of shape(n_samples,1)
  '''
    with open(filename, "r") as f:
        X = []
        y = []
        for line in f:
            test_data = [float(item) for item in line.split(" ") if
                         item != "\n"]  # cast every castable content to float
            if test_data[-1] == "\n":  # remove newline character if present
                test_data = test_data[:-1]
            y.append(test_data[0])
            X.append(test_data[1:])

    X = np.array(X)
    y = np.array(y)

    return X, y


X_train, y_train = load_data(
    "C:\\Users\\Dimitris Tzivrailis\\Desktop\\scientific books\\computer science\\patrec1_solution\\data\\train.txt")
X_test, y_test = load_data(
    "C:\\Users\\Dimitris Tzivrailis\\Desktop\\scientific books\\computer science\\patrec1_solution\\data\\test.txt")

plt.imshow(X_train[2].reshape(16, 16))
plt.colorbar()
plt.show()


def binary_color(matrix):
    m = matrix.copy()
    n, l = matrix.shape

    for i in range(n):
        for j in range(l):

            if m[i, j] <= 0:
                m[i, j] = 0
            else:
                m[i, j] = 1
    return m


plt.imshow(binary_color(X_train[2].reshape(16, 16)))
plt.show()


def train(model, train_loader, n_epochs, lr):
    train_op = optim.SGD(model.parameters(), lr)
    model.train()

    for epoch in range(n_epochs):
        loss_ = []
        for _, data in enumerate(train_loader):
            v, v_gibbs = model(data.view(-1, 16 * 16).bernoulli())

            loss = model.free_energy(v) - model.free_energy(v_gibbs)
            loss_.append(loss.item())
            train_op.zero_grad()
            loss.backward()
            train_op.step()

        print('Epoch %d\t Loss=%.4f' % (epoch, np.mean(loss_)))

    return model


class RBM(nn.Module):

    def __init__(self, n_vis=16 * 16, n_hid=128, k=5):
        super(RBM, self).__init__()
        self.v = nn.Parameter(torch.randn(1, n_vis))
        self.h = nn.Parameter(torch.randn(1, n_hid))
        self.W = nn.Parameter(torch.randn(n_hid, n_vis) * 1e-2)
        self.k = k

    def sample_from_p(self, p):
        return F.relu(torch.sign(p - torch.rand(p.size())))

    def visible_to_hidden(self, v):
        p = torch.sigmoid(F.linear(v.to(torch.float32), self.W.to(torch.float32), self.h.to(torch.float32)))
        bernoulli_new = self.sample_from_p(p)

        return p, bernoulli_new

    def hidden_to_visible(self, h):
        p = torch.sigmoid(F.linear(h, self.W.to(torch.float32).t(), self.v))
        bernoulli_new = self.sample_from_p(p)

        return p, bernoulli_new

    def free_energy(self, v):
        v_term = torch.matmul(v.to(torch.float32), self.v.to(torch.float32).t())
        w_x_h = F.linear(v.to(torch.float32), self.W, self.h)
        # h_term = torch.sum(F.softmax(w_x_h), dim=1)
        h_term = torch.sum(torch.log(1 + torch.exp(w_x_h)), dim=1)

        return (-h_term - v_term).mean()

    def forward(self, v):
        other_h, h = self.visible_to_hidden(v)
        h_1 = h
        for _ in range(self.k):
            _, v_gibb = self.hidden_to_visible(h)
            _, h_1 = self.visible_to_hidden(v_gibb)
        return v, v_gibb


xtrain = []
for i in range(np.shape(X_train)[0]):
    xtrain.append(binary_color(X_train[i].reshape(16, 16)))
torch_tensor = torch.tensor(np.array(xtrain))

batch_size = 4000
n_epochs = 10
lr = 0.01
n_hid = 200
n_vis = 16 * 16

rbm = RBM(n_vis=n_vis, n_hid=n_hid, k=5)
train_loader = torch.utils.data.DataLoader(torch_tensor, batch_size=batch_size)

model = train(rbm, torch_tensor, n_epochs=n_epochs, lr=lr)

import tensorflow
from fastai.basics import *

with torch.no_grad():
    vect = random.randint(0, 2000)
    v0 = torch_tensor[vect]
    print(v0.shape)

# print(v0)
plt.imshow(v0.reshape(16, 16))
plt.show()

for k in range(1):
    v_gibbs = model(v0.view(-1, 256))[1]
    v0 = v_gibbs

plt.imshow(v0.detach().numpy().reshape(16, 16))
plt.show()
