import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from vizTracking import visualize_tracking



N_train = 1000
N_eval = 100
T = 200
hidden_dim = 25
input_dim = 4

NUM_EPOCHS = 10

weight_file = 'weights/quat_lstm.npy'

#TODO: pose loss
#TODO adapt architecture, e.g. is proper output net needed since quaternion have strange behaviour?
#TODO: shuffle markers! then embedding net is necessary


def gen_quats(length, dims):
    theta_range = np.random.uniform(1,2)
    theta = np.linspace(-theta_range * np.pi, theta_range * np.pi, length)
    z_range = np.random.randint(1,10)
    z = np.random.uniform(1,3)*np.sin(np.linspace(0, z_range, length))
    rx = np.abs(z) ** np.random.uniform(1.5,3)*np.abs(np.random.rand())  + 1
    ry = np.abs(z) ** np.random.uniform(1.5,3)*np.abs(np.random.rand())  + 1
    rw = np.abs(z)
    x = rx**1.5 * np.sin(theta)
    y = ry**1.5 * np.cos(theta)
    w = 1 + np.random.uniform(0.5, 4)*np.sin(theta)*np.cos(theta)**2
    quats = np.stack([w, x, y, z], axis=1)
    quats = quats / np.expand_dims(np.sqrt(np.sum(np.square(quats), axis=1)), axis=1)
    return quats


def qrot(q, v):
    #TODO can I change this function to also work with constant v and changing quaternions?
    # if not just tile/stack v accordingly
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).+

    source: https://github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def gen_training_data(N):

    quat_train = np.zeros([T, N, 4], dtype=np.float32)
    quat_test = np.zeros([T, N, 4], dtype=np.float32)

    for n in range(N):
        quat_train[:, n, :] = gen_quats(T, input_dim)
        quat_test[:, n, :] = gen_quats(T, input_dim)

    #maxi1 = max(np.max(quat_train[:, :, 0]), np.max(quat_test[:, :, 0])) / 5
    #maxi2 = max(np.max(quat_train[:, :, 1]), np.max(quat_test[:, :, 1])) / 5
    #maxi3 = max(np.max(quat_train[:, :, 2]), np.max(quat_test[:, :, 2])) / 5
    #maxi4 = max(np.max(quat_train[:, :, 3]), np.max(quat_test[:, :, 3])) / 5

    quat_train[:, :, 0] = quat_train[:, :, 0]# / maxi1
    quat_train[:, :, 1] = quat_train[:, :, 1]# / maxi2
    quat_train[:, :, 2] = quat_train[:, :, 2]# / maxi3
    quat_train[:, :, 3] = quat_train[:, :, 3]# / maxi4

    quat_test[:, :, 0] = quat_test[:, :, 0]# / maxi1
    quat_test[:, :, 1] = quat_test[:, :, 1]# / maxi2
    quat_test[:, :, 2] = quat_test[:, :, 2]# / maxi3
    quat_test[:, :, 3] = quat_test[:, :, 3]# / maxi4

    marker1 = np.array([0, 0, 0])
    marker2 = np.array([0, 0, 0.5])
    marker3 = np.array([-0.7, -1, 0])
    marker4 = np.array([1.1, -1, 0.8])
    pattern = np.stack([marker1, marker2, marker3, marker4], axis=0)

    stacked_marker1 = np.tile(marker1, reps=(T, N, 1))
    stacked_marker2 = np.tile(marker2, reps=(T, N, 1))
    stacked_marker3 = np.tile(marker3, reps=(T, N, 1))
    stacked_marker4 = np.tile(marker4, reps=(T, N, 1))

    print(np.shape(stacked_marker1))
    print(np.shape(quat_train))
    assert np.shape(stacked_marker1)[:2] == np.shape(quat_train)[:2]

    quat_train = torch.from_numpy(quat_train).float()
    quat_test = torch.from_numpy(quat_test).float()
    stacked_marker1 = torch.from_numpy(stacked_marker1).float()
    stacked_marker2 = torch.from_numpy(stacked_marker2).float()
    stacked_marker3 = torch.from_numpy(stacked_marker3).float()
    stacked_marker4 = torch.from_numpy(stacked_marker4).float()


    rotated_marker1_train = qrot(quat_train, stacked_marker1)
    rotated_marker2_train = qrot(quat_train, stacked_marker2)
    rotated_marker3_train = qrot(quat_train, stacked_marker3)
    rotated_marker4_train = qrot(quat_train, stacked_marker4)
    X_train = torch.cat([rotated_marker1_train,
                        rotated_marker2_train,
                        rotated_marker3_train,
                        rotated_marker4_train], dim=2)

    rotated_marker1_test = qrot(quat_test, stacked_marker1)
    rotated_marker2_test = qrot(quat_test, stacked_marker2)
    rotated_marker3_test = qrot(quat_test, stacked_marker3)
    rotated_marker4_test = qrot(quat_test, stacked_marker4)
    X_test = torch.cat( [rotated_marker1_test,
                        rotated_marker2_test,
                        rotated_marker3_test,
                        rotated_marker4_test], dim=2)

    return X_train, quat_train, X_test, quat_test, pattern


class LSTMTracker(nn.Module):

    def __init__(self, hidden_dim, input_dim):
        super(LSTMTracker, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(12, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2quat = nn.Linear(hidden_dim, 4)

        self.dropout = nn.Dropout(p=0.15)

    def forward(self, detections):
        #x = self.dropout(F.relu(self.emb1(trajectory)))
        #x = self.dropout(F.relu(self.emb2(x)))
        #x = self.dropout(F.relu(self.emb3(x)))
        lstm_out, _ = self.lstm(detections)
        quat_space = self.hidden2quat(lstm_out)
        #quat_norm = torch.sqrt(torch.sum(torch.pow(quat_space, 2, ), dim=2))
        #quat_space = quat_space / torch.unsqueeze(quat_norm, dim=2)
        return quat_space


model = LSTMTracker(hidden_dim, input_dim)
# TODO: try pose error!!
# TODO: respect antipodal pair as well!
loss_function = nn.L1Loss() #nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

def train():
    [X_train, Y_train, X_test, Y_test, pattern] = gen_training_data(N_train)
    model.train()
    for epoch in range(NUM_EPOCHS):
        batches = torch.split(X_train, 50, 1)
        truth_batches = torch.split(Y_train, 50, 1)
        avg_loss = 0
        for batch, truth_batch in zip(batches, truth_batches):
            model.zero_grad()

            pred_quat = model(batch[:-1, :, :])

            loss = loss_function(pred_quat, truth_batch[1:, :, :])
            loss.backward()
            optimizer.step()
            avg_loss += loss
        avg_loss /= len(batches)

        model.eval()
        with torch.no_grad():
            preds = model(X_test[:-1,:,:])
            loss = loss_function(preds, Y_test[1:,:,:])
            print("training loss: {ltrain:2.4f}, test loss: {ltest:2.4f}".format(ltrain=avg_loss.data, ltest=loss.data))
    torch.save(model.state_dict(), weight_file)



def eval():
    [X_train, Y_train, X_test, Y_test, pattern] = gen_training_data(N_eval)

    model.load_state_dict(torch.load(weight_file))
    model.eval()


    with torch.no_grad():
        quat_preds = model(X_test[:-1, :, :])

        print(X_test[:10, 0,:])
        print(quat_preds[:10,0,:])
        print(Y_test[:10, 0, :])
        for n in range(10):
            visualize_tracking(None,
                               quat_preds[:, n, :].detach().numpy(),
                               None,
                               Y_test[1:, n, :].detach().numpy(),
                               X_test[:-1, n, :].numpy(),
                               pattern)


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