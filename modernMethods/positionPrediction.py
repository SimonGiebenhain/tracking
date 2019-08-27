import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

N = 200
T = 50
hidden_dim = 20
input_dim = 3

NUM_EPOCHS = 2000


#TODO 1  : visualize trajectories
#TODO 2  : generate more complex data, training and testing data
#TODO 2.5: adapt architecture
#TODO 3  : generate data which is similar to birds trajectories?


def gen_training_data():
    slopes = range(N)
    xaxis = np.arange(T)

    X = np.zeros([T, N, 3], dtype=np.float32)
    for n,a in enumerate(slopes):
        X[:, n ,0] = xaxis
        X[:, n ,1] = a/10 * xaxis
        X[:, n, 2] = np.sin(a/10 * xaxis/10)

    maxi1 = np.max(X[:,:,0])
    maxi2 = np.max(X[:,:,1])
    X[:,:,0] = X[:,:,0] / maxi1
    X[:,:,1] = X[:,:,1] / maxi2
    return X, maxi1, maxi2

lstm = nn.LSTM(input_dim, hidden_dim)  # Input dim is 3, output dim is 3
[X_train, maxi1, maxi2] = gen_training_data()
print(np.squeeze(X_train[:,2,:]))
inputs = torch.from_numpy(X_train)
hidden = (torch.randn(1, N, hidden_dim), torch.randn(1, N, hidden_dim))
out, hidden = lstm(inputs, hidden)

#print(out)
#print(hidden)


class LSTMTracker(nn.Module):

    def __init__(self, hidden_dim, input_dim):
        super(LSTMTracker, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2pos = nn.Linear(hidden_dim, input_dim)

    def forward(self, trajectory):
        lstm_out, _ = self.lstm(trajectory)
        pos_space = self.hidden2pos(lstm_out)
        next_pos = F.tanh(pos_space)
        return next_pos


model = LSTMTracker(hidden_dim, input_dim)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#with torch.no_grad():
#    next_pos = model(inputs)
#    print(next_pos)


for epoch in range(NUM_EPOCHS):  # again, normally you would NOT do 300 epochs, it is toy data
    #for sentence, tags in training_data:
    batches = torch.split(inputs, 50, 1)
    for batch in batches:
        model.zero_grad()

        pred_pos = model(batch[:-1,:,:])

        loss = loss_function(pred_pos, batch[1:,:,:])
        loss.backward()
        optimizer.step()
        print(loss)

with torch.no_grad():
    tag_scores = model(inputs)
    print(tag_scores.shape)
    print(torch.squeeze(tag_scores[:,2,:]))
    #print(tag_scores)


with torch.no_grad():
    X = np.zeros([T, 1, input_dim], dtype=np.float32)
    X[:, 0 ,0] = np.arange(T)
    X[:, 0 ,1] = 2 * np.arange(T)
    X[:, 0, 2] = np.sin(2 * np.arange(T)/10)

    X[:, :, 0] = X[:, :, 0] / maxi1
    X[:, :, 1] = X[:, :, 1] / maxi2
    X = torch.from_numpy(X)
    pred_pos = model(X[:-1, : ,:])
    print(X[1:, :, :])
    print(pred_pos)
    print(X[1:, : ,:] - pred_pos)
    scale = torch.tensor([maxi1, maxi2, 1]).view(1,1,input_dim)
    X = X * scale
    pred_pos = pred_pos * scale
    print(X)
    print(pred_pos)
    print(X[1:, : ,:] - pred_pos)
