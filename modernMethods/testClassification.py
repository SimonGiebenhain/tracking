import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pyquaternion import Quaternion as Quaternion


import matplotlib.pyplot as plt

pattern = np.array([[0,0,0], [1,1,1], [0, 1, 0], [-0.5, -0.5, 0]])

def gen_datum():
    q = np.random.uniform(0, 1, 4)
    q_norm = np.linalg.norm(q)
    q = q / q_norm
    Q = Quaternion(q)
    rot_mat = Q.rotation_matrix
    rot_pat = (rot_mat @ pattern.T).T
    return rot_pat


def gen_datum2():
    theta = np.random.uniform(0, 6)
    dirx = np.sin(theta)
    diry = np.cos(theta)
    small = np.random.uniform(0.5,1.5)
    big = np.random.uniform(2,3)

    p1x = small*dirx
    p1y = small*diry
    p1 = np.stack([p1x, p1y])

    p2x = big*dirx
    p2y = big*diry
    p2 = np.stack([p2x, p2y])
    p3 = np.random.normal(0, 1, [2])

    return np.stack([p1, p2, p3], axis=0)


N = 50*100
data = np.zeros([N, 4, 3])
classes = np.zeros([N, 4])
for i in range(N):
    datum = gen_datum()
    perm = np.random.permutation(np.arange(0,4))
    classes[i, :] = perm[perm]
    datum = datum[perm, :]
    data[i, :, :] = datum

X_train = torch.from_numpy(np.reshape(data, [N, -1])).float()
Y_train = torch.from_numpy(classes).type(torch.LongTensor)

class simpleNet(nn.Module):
    def __init__(self):
        super(simpleNet, self).__init__()
        self.layer1 = nn.Linear(12, 20)
        self.layer2 = nn.Linear(20, 50)
        self.out1 = nn.Linear(50, 4)
        self.out2 = nn.Linear(50, 4)
        self.out3 = nn.Linear(50, 4)
        self.out4 = nn.Linear(50, 4)


    def forward(self, detections):
        x = F.relu(self.layer1(detections))
        x = F.relu(self.layer2(x))
        c1 = self.out1(x)
        c2 = self.out2(x)
        c3 = self.out3(x)
        c4 = self.out4(x)

        return c1, c2, c3, c4

CELoss = nn.CrossEntropyLoss()

net = simpleNet()
net.float()
optimizer = torch.optim.Adam(net.parameters())

def train(X_train, Y_train):
    for epoch in range(10):
        net.train()
        x_batches = torch.split(X_train, 50, 0)
        y_batches = torch.split(Y_train, 50, 0)

        avg_loss = 0
        for (x, y) in zip(x_batches, y_batches):
            net.zero_grad()
            c1, c2, c3, c4 = net(x)
            l1 = CELoss(c1, y[:, 0])
            l2 = CELoss(c2, y[:, 1])
            l3 = CELoss(c3, y[:, 2])
            l4 = CELoss(c4, y[:, 3])
            loss = l1 + l2 + l3 + l4

            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        avg_loss /= len(y_batches)
        print(avg_loss)


train(X_train, Y_train)

net.eval()
c1, c2, c3, c4 = net(X_train)
print(c2[:10, :])
print(Y_train[:10, 1])
#print(data)

#data_lin = np.reshape(data, [3*N, 2])
#print(data_lin)
#plt.scatter(data_lin[:, 0], data_lin[:, 1])
#plt.show()


