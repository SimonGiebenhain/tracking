import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pyquaternion import Quaternion as Quaternion

#from torchsummary import summary
import numpy as np
from viz import visualize
from vizTracking import visualize_tracking


# DONE: 1) visualize noisy detections as well
# DONE: 2) incorporate quaternion loss as well
# TODO: 2.5) Then also visualize the predicted marker locations
# TODO: 3) deal with different markers
# TODO: 4) handle missing detections
# TODO: 5) multimodal predictions?


TRAIN_SIZE = 10000
TEST_SIZE = int(TRAIN_SIZE / 10)
T = 200
NUM_EPOCHS = 15m
BATCH_SIZE = 64

NOISE_STD = 0.01

dim = 3
input_dim = 12
fc1_dim = 30
embedding_dim = 35
hidden_dim = 40
fc2_dim = 20
output_pos_dim = 3
output_quat_dim = 4



#TODO, check what would be the right scale of the pattern
pattern = 0.1 * np.array([[0,1,0], [0,0,1], [-1,-1,0], [1, -1, 1]])


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


def gen_data(size):
    X= np.zeros([T, int(size), 3], dtype=np.float32)

    for n in range(size):
        X[:, n, :] = Gen_Spirals(T, dim)

    maxi1 = np.max(X[:,:,0])/5
    maxi2 = np.max(X[:,:,1])/5
    maxi3 = np.max(X[:,:,2])/5
    X[:,:,0] = X[:,:,0] / maxi1
    X[:,:,1] = X[:,:,1] / maxi2
    X[:,:,2] = X[:,:,2] / maxi3

    return X


def gen_quats(size):
    quats = np.zeros([T, size, 4])


    def gen_quat():
        quat = np.zeros([T,4])
        xaxis = np.linspace(np.random.uniform(-1,1), np.random.uniform(2, 4), T)
        quat[:, 0] = np.random.uniform(0.5,4) * np.sin(xaxis/np.random.uniform(0.3,3))*np.random.uniform(1,10)
        quat[:, 1] = np.random.uniform(0.5,4) * xaxis / T * np.random.uniform(-3,3)
        quat[:, 2] = np.random.uniform(0.5,4) * np.sin(xaxis/np.random.uniform(0.3,3))**(np.random.randint(1,4))
        quat[:, 3] = np.random.uniform(0.5,4) * np.arctan(xaxis/np.random.uniform(0.3,3))*np.random.uniform(0,5)

        return quat / np.sqrt(np.sum(np.square(quat), axis=0))


    for n in range(size):
        quats[:, n, :] = gen_quat()

    return quats


def add_markers_to_trajectories(trajectory, quats, pattern):
    T = len(trajectory)
    N = np.shape(trajectory)[1]

    detections = np.zeros([T, N, 12])
    for t in range(T):
        for n in range(N):
            quat = Quaternion(quats[t,n,:])
            rot_mat = quat.rotation_matrix
            rotated_pattern = np.dot(rot_mat, pattern.T).T
            det = np.reshape(rotated_pattern + trajectory[t, n], -1)
            det = det + np.random.normal(0, NOISE_STD, np.shape(det))
            detections[t, n, :] = det

    #marker0 = trajectory + pattern[0, :] + np.random.normal(0, 0.02, np.shape(trajectory))
    #marker1 = trajectory + pattern[1, :] + np.random.normal(0, 0.02, np.shape(trajectory))
    #marker2 = trajectory + pattern[2, :] + np.random.normal(0, 0.02, np.shape(trajectory))
    #marker3 = trajectory + pattern[3, :] + np.random.normal(0, 0.02, np.shape(trajectory))

    #detections = np.concatenate([marker0, marker1, marker2, marker3], axis=2)
    #print(np.shape(detections))
    return detections


def complete_gen(pattern, train_size, test_size, both):
    if both:
        train_pos = gen_data(train_size)
        train_quats = gen_quats(train_size)
        train_dets = add_markers_to_trajectories(train_pos, train_quats, pattern)
        X_train = torch.from_numpy(train_dets).float()
        Y_pos_train = torch.from_numpy(train_pos).float()
        Y_quat_train = torch.from_numpy(train_quats).float()

    test_pos = gen_data(test_size)
    test_quats = gen_quats(test_size)
    test_dets = add_markers_to_trajectories(test_pos, test_quats, pattern)
    X_test = torch.from_numpy(test_dets).float()
    Y_pos_test = torch.from_numpy(test_pos).float()
    Y_quat_test = torch.from_numpy(test_quats).float()

    if both:
        return X_train, Y_pos_train, Y_quat_train, X_test, Y_pos_test, Y_quat_test
    else:
        return X_test, Y_pos_test, Y_quat_test


class LSTMTracker(nn.Module):

    def __init__(self, embedding_dim, hidden_dim):
        super(LSTMTracker, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim, fc1_dim)

        self.marker_embedding = nn.Linear(fc1_dim, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, fc2_dim)

        # The linear layer that maps from hidden state space to tag space

        self.map2pos = nn.Linear(fc2_dim, output_pos_dim)
        self.map2quat = nn.Linear(fc2_dim, output_quat_dim)

        self.dropout = nn.Dropout(p=0.2)


    def forward(self, marker_detections):
        x = self.dropout(F.relu(self.fc1(marker_detections)))
        embeddings = self.dropout(F.relu(self.marker_embedding(x)))
        lstm_out, _ = self.lstm(embeddings.view(len(marker_detections), -1, self.embedding_dim))

        x = self.dropout(F.relu(self.fc2(self.dropout(lstm_out))))
        pos_space = self.map2pos(x)
        quat_space = F.softmax(self.map2quat(x), dim=2)
        #next_pos = F.tanh(pos_space)
        return pos_space, quat_space


model = LSTMTracker(embedding_dim, hidden_dim)
model = model.float()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#summary(model, input_size=())

def train():

    [X_train, Y_pos_train, Y_quat_train, X_test, Y_pos_test, Y_quat_test] = complete_gen(pattern, TRAIN_SIZE, TEST_SIZE, both=True)

    model.train()
    for epoch in range(NUM_EPOCHS):
        X_batches = torch.split(X_train, BATCH_SIZE, 1)
        Y_pos_batches = torch.split(Y_pos_train, BATCH_SIZE, 1)
        Y_quat_batches = torch.split(Y_quat_train, BATCH_SIZE, 1)
        avg_loss = 0
        for X_batch, Y_pos_batch, Y_quat_batch in zip(X_batches, Y_pos_batches, Y_quat_batches):
            model.zero_grad()

            pred_pos, pred_quat = model(X_batch[:-1,:,:])

            #TODO scale loss properly! when position number are big then quat loss basically does nothing!
            # maybe scale by average norm of positions then maybe weight constantly e.g. 1/2 since the focus is on position,
            # but that should make a difference anyways
            loss_pos = loss_function(pred_pos, Y_pos_batch[1:,:,:])
            loss_quat = loss_function(pred_quat, Y_quat_batch[1:, :, :])
            loss = loss_pos + loss_quat
            loss.backward()
            optimizer.step()
            avg_loss += loss
        avg_loss /= len(Y_pos_batches)

        model.eval()
        with torch.no_grad():
            pred_pos, pred_quat = model(X_test[:-1,:,:])
            loss_pos = loss_function(pred_pos, Y_pos_test[1:, :, :])
            loss_quat = loss_function(pred_quat, Y_quat_test[1:, :, :])
            loss = loss_pos + loss_quat
            print("training loss: {ltrain:2.4f}, test loss: {ltest:2.4f}".format(ltrain=avg_loss.data, ltest=loss.data))
    torch.save(model.state_dict(), 'weights/lstm_4marker_to_pos')



def eval():
    test_size = 100
    [X_test, Y_pos_test, Y_quat_test] = complete_gen(pattern, 0, test_size, both=False)

    model.load_state_dict(torch.load('weights/lstm_4marker_to_pos'))
    model.eval()

    predicted_pos, _ = model(X_test[:-1, :, :])

    for n in range(10):
        with torch.no_grad():
            data = torch.stack((predicted_pos[:,n,:], Y_pos_test[1:,n,:]),1)
            center = torch.mean(data, dim=(0,1)).numpy()
            visualize_tracking(predicted_pos[:, n, :].detach().numpy() - center, None, Y_pos_test[1:, n, :].numpy() - center, None, X_test[:-1, n, :].numpy(), pattern)

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