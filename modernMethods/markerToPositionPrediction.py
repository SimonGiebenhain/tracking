import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from viz import visualize


N = 1000
T = 200
hidden_dim = 20
embedding_dim = 15
input_dim = 3

NUM_EPOCHS = 20
BATCH_SIZE = 32


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

    X_train = np.zeros([T, N, 3], dtype=np.float32)
    X_test = np.zeros([T, N, 3], dtype=np.float32)
    #for n,a in enumerate(slopes):
    #    X[:, n ,0] = xaxis
    #    X[:, n ,1] = a/10 * xaxis
    #    X[:, n, 2] = np.sin(a/10 * xaxis/10)

    for n in range(N):
        X_train[:, n, :] = Gen_Spirals(T, input_dim)
        X_test[:, n, :] = Gen_Spirals(T, input_dim)

    maxi1 = max(np.max(X_train[:,:,0]), np.max(X_test[:,:,0]))/5
    maxi2 = max(np.max(X_train[:,:,1]), np.max(X_test[:,:,1]))/5
    maxi3 = max(np.max(X_train[:,:,2]), np.max(X_test[:,:,2]))/5
    X_train[:,:,0] = X_train[:,:,0] / maxi1
    X_train[:,:,1] = X_train[:,:,1] / maxi2
    X_train[:,:,2] = X_train[:,:,2] / maxi3
    X_test[:, :, 0] = X_test[:, :, 0] / maxi1
    X_test[:, :, 1] = X_test[:, :, 1] / maxi2
    X_test[:, :, 2] = X_test[:, :, 2] / maxi3
    return X_train, X_test, maxi1, maxi2, maxi3


def add_markers_to_trajectories(trajectory, pattern):
    T = len(trajectory)
    N = np.shape(trajectory)[1]

    marker0 = trajectory + pattern[0, :]
    marker1 = trajectory + pattern[1, :]
    marker2 = trajectory + pattern[2, :]
    marker3 = trajectory + pattern[3, :]

    detections = np.concatenate([marker0, marker1, marker2, marker3], axis=2)
    print(np.shape(detections))
    return detections



[train, test, maxi1, maxi2, maxi3] = gen_training_data()
#TODO, check what would be the right scale of the pattern
pattern = np.array([[0,1,0], [0,0,1], [-1,-1,0], [1, -1, 1]])
train_dets = add_markers_to_trajectories(train, pattern)
test_dets = add_markers_to_trajectories(test, pattern)

#print(np.squeeze(X_train[:,2,:]))
X_train = torch.from_numpy(train_dets).float()
Y_train = torch.from_numpy(train).float()
##visualize(train_data, T)
X_test = torch.from_numpy(test_dets).float()
Y_test = torch.from_numpy(test).float()
#hidden = (torch.randn(1, N, hidden_dim), torch.randn(1, N, hidden_dim))


class LSTMTracker(nn.Module):

    def __init__(self, embedding_dim, hidden_dim):
        super(LSTMTracker, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.marker_embedding = nn.Linear(12, embedding_dim)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2pos = nn.Linear(hidden_dim, 3)

    def forward(self, marker_detections):
        embeddings = self.marker_embedding(marker_detections)
        lstm_out, _ = self.lstm(embeddings.view(len(marker_detections), -1, self.embedding_dim))
        pos_space = self.hidden2pos(lstm_out)
        #next_pos = F.tanh(pos_space)
        return pos_space


model = LSTMTracker(embedding_dim, hidden_dim)
model = model.float()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    for epoch in range(NUM_EPOCHS):
        X_batches = torch.split(X_train, BATCH_SIZE, 1)
        Y_batches = torch.split(Y_train, BATCH_SIZE, 1)
        avg_loss = 0
        for X_batch, Y_batch in zip(X_batches, Y_batches):
            model.zero_grad()

            pred_pos = model(X_batch[:-1,:,:])

            loss = loss_function(pred_pos, Y_batch[1:,:,:])
            loss.backward()
            optimizer.step()
            avg_loss += loss
        avg_loss /= len(Y_batches)

        model.eval()
        with torch.no_grad():
            preds = model(X_test[:-1,:,:])
            loss = loss_function(preds, Y_test[1:,:,:])
            print("training loss: {ltrain:2.4f}, test loss: {ltest:2.4f}".format(ltrain=avg_loss.data, ltest=loss.data))
    torch.save(model.state_dict(), 'weights/lstm_4marker_to_pos')



def eval():
    model.load_state_dict(torch.load('weights/lstm_4marker_to_pos'))
    model.eval()
    for n in range(N):
        with torch.no_grad():
            tag_scores = model(X_test[:-1, :, :])
            #print(tag_scores.shape)
            #print(torch.squeeze(tag_scores[:,n,:]))
            data = torch.stack((tag_scores[:,n,:], Y_test[1:,n,:]),1)
            center = torch.mean(data, dim=(0,1))
            data = data - centerm
            print(data.shape)
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




#train()
eval()
#eval_diff2()