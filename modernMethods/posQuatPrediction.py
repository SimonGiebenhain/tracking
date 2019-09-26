import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from viz import visualize


N = 20000
T = 200
emb_dim1 = 150
emb_dim2 = 200
emb_dim3 = 150
hidden_dim = 30
input_dim = 3

NUM_EPOCHS = 30


#TODO 1  : generate data which is similar to birds trajectories?
#TODO 2  : add noise
#TODO 3  : add marker
#TODO 4  : make model predict marker association
#TODO 5  : handle missing detections


def Gen_Spirals(length, dims=2):
    theta_range = np.random.randint(1,10)
    theta = np.linspace(-theta_range * np.pi, theta_range * np.pi, length)
    z_range = np.random.randint(15,45)
    z = np.random.uniform(1,3)*np.sin(np.linspace(0, z_range, length))
    rx = np.abs(z) ** np.random.uniform(1.5,3)*np.abs(np.random.rand())  + 1
    ry = np.abs(z) ** np.random.uniform(1.5,3)*np.abs(np.random.rand())  + 1
    x = rx**1.5 * np.sin(theta)
    y = ry**1.5 * np.cos(theta)

    return np.stack([x,y,z], axis=1) + 5*np.random.uniform(low=-5, high=5, size=[1,dims])


def gen_training_data():
    #slopes = range(N)
    #xaxis = np.arange(T)

    pos_train = np.zeros([T, N, 3], dtype=np.float32)
    pos_test = np.zeros([T, N, 3], dtype=np.float32)
    #for n,a in enumerate(slopes):
    #    X[:, n ,0] = xaxis
    #    X[:, n ,1] = a/10 * xaxis
    #    X[:, n, 2] = np.sin(a/10 * xaxis/10)

    for n in range(N):
        pos_train[:, n, :] = Gen_Spirals(T, input_dim)
        pos_test[:, n, :] = Gen_Spirals(T, input_dim)

    maxi1 = max(np.max(pos_train[:,:,0]), np.max(pos_test[:,:,0]))/5
    maxi2 = max(np.max(pos_train[:,:,1]), np.max(pos_test[:,:,1]))/5
    maxi3 = max(np.max(pos_train[:,:,2]), np.max(pos_test[:,:,2]))/5
    pos_train[:,:,0] = pos_train[:,:,0] / maxi1
    pos_train[:,:,1] = pos_train[:,:,1] / maxi2
    pos_train[:,:,2] = pos_train[:,:,2] / maxi3
    pos_test[:, :, 0] = pos_test[:, :, 0] / maxi1
    pos_test[:, :, 1] = pos_test[:, :, 1] / maxi2
    pos_test[:, :, 2] = pos_test[:, :, 2] / maxi3

    pattern = 0.1 * np.array([[0, 0, 0], [0, 0, 0.5], [-0.7, -1, 0], [1.1, -1, 0.8]])

    #marker0 = np.tile(pattern)

    X_train = np.zeros([T, N, 12], dtype=np.float32)
    X_test = np.zeros([T, N, 12], dtype=np.float32)

    for n in range(N):
        for t in range(T):
            np.random.shuffle(pattern)
            X_train[t, n, :3] = pos_train[t, n, :] + pattern[0, :]
            X_train[t, n, 3:6] = pos_train[t, n, :] + pattern[1, :]
            X_train[t, n, 6:9] = pos_train[t, n, :] + pattern[2, :]
            X_train[t, n, 9:12] = pos_train[t, n, :] + pattern[3, :]

    for n in range(N):
        for t in range(T):
            np.random.shuffle(pattern)
            X_test[t, n, :3] = pos_test[t, n, :] + pattern[0, :]
            X_test[t, n, 3:6] = pos_test[t, n, :] + pattern[1, :]
            X_test[t, n, 6:9] = pos_test[t, n, :] + pattern[2, :]
            X_test[t, n, 9:12] = pos_test[t, n, :] + pattern[3, :]


    return X_train, X_test, pos_train, pos_test, maxi1, maxi2, maxi3


#lstm = nn.LSTM(input_dim, hidden_dim)  # Input dim is 3, output dim is 3
[X_train, X_test, pos_train, pos_test, maxi1, maxi2, maxi3] = gen_training_data()
#print(np.squeeze(X_train[:,2,:]))
train_in = torch.from_numpy(X_train)
train_out = torch.from_numpy(pos_train)
#visualize(train_data, T)
test_in = torch.from_numpy(X_test)
test_out = torch.from_numpy(pos_test)
#hidden = (torch.randn(1, N, hidden_dim), torch.randn(1, N, hidden_dim))
#out, hidden = lstm(train_data, hidden)


class LSTMTracker(nn.Module):

    def __init__(self, hidden_dim, input_dim):
        super(LSTMTracker, self).__init__()
        self.hidden_dim = hidden_dim

        self.emb1 = nn.Linear(12, emb_dim1)
        self.emb2 = nn.Linear(emb_dim1, emb_dim2)
        self.emb3 = nn.Linear(emb_dim2, emb_dim3)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(emb_dim3, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2pos = nn.Linear(hidden_dim, 3)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, trajectory):
        x = self.dropout(F.relu(self.emb1(trajectory)))
        x = self.dropout(F.relu(self.emb2(x)))
        x = self.dropout(F.relu(self.emb3(x)))
        lstm_out, _ = self.lstm(x)
        pos_space = self.hidden2pos(lstm_out)
        #next_pos = F.tanh(pos_space)
        return pos_space


model = LSTMTracker(hidden_dim, input_dim)
loss_function = nn.L1Loss() #nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

def train():
    model.train()
    for epoch in range(NUM_EPOCHS):
        batches = torch.split(train_in, 50, 1)
        truth_batches = torch.split(train_out, 50, 1)
        avg_loss = 0
        for batch, truth_batch in zip(batches, truth_batches):
            model.zero_grad()

            pred_pos = model(batch[:-1, :, :])

            loss = loss_function(pred_pos, truth_batch[1:, :, :])
            loss.backward()
            optimizer.step()
            avg_loss += loss
        avg_loss /= len(batches)

        model.eval()
        with torch.no_grad():
            preds = model(test_in[:-1,:,:])
            loss = loss_function(preds, test_out[1:,:,:])
            print("training loss: {ltrain:2.4f}, test loss: {ltest:2.4f}".format(ltrain=avg_loss.data, ltest=loss.data))
    torch.save(model.state_dict(), 'weights/lstm')



def eval():
    model.load_state_dict(torch.load('weights/lstm'))
    model.eval()
    with torch.no_grad():
        tag_scores = model(test_in[:-1, :, :])
        for n in range(10):
            center = torch.mean(tag_scores[:, n, :], dim=0)
            #print(tag_scores.shape)
            #print(torch.squeeze(tag_scores[:,n,:]))
            data = torch.stack((tag_scores[:,n,:]-center, test_out[1:,n,:]-center),1)
            #print(data.shape)
            visualize(data)

def eval_diff():
    model.load_state_dict(torch.load('weights/lstm'))
    model.eval()
    with torch.no_grad():
        X = np.zeros([T, 1, input_dim], dtype=np.float32)
        X[:, 0 ,0] = np.arange(T)
        X[:, 0 ,1] = 2 * np.arange(T)
        X[:, 0, 2] = np.sin(2 * np.arange(T)/10)

        maxi1 = np.max(X[:, :, 0])
        maxi2 = np.max(X[:, :, 1])
        maxi3 = np.max(X[:, :, 2])

        X[:, :, 0] = X[:, :, 0] / maxi1
        X[:, :, 1] = X[:, :, 1] / maxi2
        X[:, :, 2] = X[:, :, 2] / maxi3
        X = torch.from_numpy(X)
        pred_pos = model(X[:-1, : ,:])
        #print(X[1:, :, :])
        #print(pred_pos)
        #print(X[1:, : ,:] - pred_pos)
        #scale = torch.tensor([maxi1, maxi2, 1]).view(1,1,input_dim)
        X = X #* scale
        pred_pos = pred_pos #* scale
        #print(X)
        #print(pred_pos)
        #print(X[1:, : ,:] - pred_pos)
        data = torch.stack((pred_pos[:, 0, :], X[1:, 0, :]), 1)
        visualize(data)

def eval_diff2():
    model.load_state_dict(torch.load('weights/lstm'))
    model.eval()
    with torch.no_grad():
        X = np.zeros([T, 1, input_dim], dtype=np.float32)
        speedup = 3
        times = np.arange(T) * speedup
        X[:, 0, 0] = times / (T * speedup) * 5 * np.sin(times / 32)
        X[:, 0, 1] = np.cos(times / 68) ** 2 * 5
        X[:, 0, 2] = np.sin(times / 20) ** 2 * times / (T * speedup) * 3

        maxi1 = np.max(X[:, :, 0])
        maxi2 = np.max(X[:, :, 1])
        maxi3 = np.max(X[:, :, 2])

        X[:, :, 0] = X[:, :, 0] / maxi1
        X[:, :, 1] = X[:, :, 1] / maxi2
        X[:, :, 2] = X[:, :, 2] / maxi3
        X = torch.from_numpy(X)
        pred_pos = model(X[:-1, :, :])
        # print(X[1:, :, :])
        # print(pred_pos)
        # print(X[1:, : ,:] - pred_pos)
        # scale = torch.tensor([maxi1, maxi2, 1]).view(1,1,input_dim)
        X = X  # * scale
        pred_pos = pred_pos  # * scale
        # print(X)
        # print(pred_pos)
        # print(X[1:, : ,:] - pred_pos)
        data = torch.stack((pred_pos[:, 0, :], X[1:, 0, :]), 1)
        visualize(data)




train()
eval()
#eval_diff2()