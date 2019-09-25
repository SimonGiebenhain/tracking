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
# DONE2.5) Then also visualize the predicted marker locations
# TODO: 3) deal with different markers
# TODO: 4) handle missing detections and false positives
# TODO: 5) multimodal predictions?


TRAIN_SIZE = 10000
TEST_SIZE = int(TRAIN_SIZE / 10)
T = 200
NUM_EPOCHS = 20
BATCH_SIZE = 64

NOISE_STD = 0.001

dim = 3
input_dim = 12
fc1_dim = 50
embedding_dim = 70
hidden_dim = 40
fc2_dim = 50
output_pos_dim = 3
output_quat_dim = 4



#TODO, check what would be the right scale of the pattern
pattern = 0.1 * np.array([[0,0,0], [0,0,0.5], [-0.7,-1,0], [1.1, -1, 0.8]])

# OLD!!!
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
            detections[t, n, :] = det

    det_no_noise = detections
    detections = detections + np.random.normal(0, NOISE_STD, np.shape(detections))

    #marker0 = trajectory + pattern[0, :] + np.random.normal(0, 0.02, np.shape(trajectory))
    #marker1 = trajectory + pattern[1, :] + np.random.normal(0, 0.02, np.shape(trajectory))
    #marker2 = trajectory + pattern[2, :] + np.random.normal(0, 0.02, np.shape(trajectory))
    #marker3 = trajectory + pattern[3, :] + np.random.normal(0, 0.02, np.shape(trajectory))

    #detections = np.concatenate([marker0, marker1, marker2, marker3], axis=2)
    #print(np.shape(detections))
    return detections, det_no_noise


def complete_gen(pattern, train_size, test_size, both):
    if both:
        train_pos = gen_data(train_size)
        train_quats = gen_quats(train_size)
        train_dets, Y_marker_train = add_markers_to_trajectories(train_pos, train_quats, pattern)
        X_train = torch.from_numpy(train_dets).float()
        Y_pos_train = torch.from_numpy(train_pos).float()
        Y_quat_train = torch.from_numpy(train_quats).float()
        Y_marker_train = torch.from_numpy(Y_marker_train).float()

    test_pos = gen_data(test_size)
    test_quats = gen_quats(test_size)
    test_dets, Y_marker_test = add_markers_to_trajectories(test_pos, test_quats, pattern)
    X_test = torch.from_numpy(test_dets).float()
    Y_pos_test = torch.from_numpy(test_pos).float()
    Y_quat_test = torch.from_numpy(test_quats).float()
    Y_marker_test = torch.from_numpy(Y_marker_test).float()

    if both:
        return X_train, Y_pos_train, Y_quat_train, Y_marker_train, X_test, Y_pos_test, Y_quat_test, Y_marker_test
    else:
        return X_test, Y_pos_test, Y_quat_test, Y_marker_test


#TODO compare speed of qrot and stack, with rotation matrix
# qrot is probably better
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


def quat2mat(q):
    """
    Takes tensors with quaternions in last dimension, i.e. q has shape (*,4),
        where * can be any positive number of dimensions.

    Original code from: https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
    """

    original_shape = list(q.shape)
    q = q.view(-1, 4)

    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMats = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1)
    rotMats = rotMats.view(original_shape[:-1]+ [9])
    rotMats = rotMats.view(original_shape[:-1]+ [3, 3])
    return rotMats


def quat_rot(q, v):
    return torch.matmul(quat2mat(q), torch.from_numpy(v.T).float()).permute([0, 1, 3, 2])

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
        predicted_markers = quat_rot(quat_space, pattern)
        predicted_markers = predicted_markers + pos_space.unsqueeze(2)
        predicted_markers = predicted_markers.contiguous().view(len(marker_detections), -1, 12)

        return predicted_markers, pos_space, quat_space


model = LSTMTracker(embedding_dim, hidden_dim)
model = model.float()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#summary(model, input_size=())

def train():

    [X_train, Y_pos_train, Y_quat_train, Y_marker_train, X_test, Y_pos_test, Y_quat_test, Y_marker_test] = complete_gen(pattern, TRAIN_SIZE, TEST_SIZE, both=True)

    model.train()
    for epoch in range(NUM_EPOCHS):
        X_batches = torch.split(X_train, BATCH_SIZE, 1)
        Y_pos_batches = torch.split(Y_pos_train, BATCH_SIZE, 1)
        Y_quat_batches = torch.split(Y_quat_train, BATCH_SIZE, 1)
        Y_marker_batches = torch.split(Y_marker_train, BATCH_SIZE, 1)
        avg_loss = 0
        for X_batch, Y_pos_batch, Y_quat_batch, Y_marker_batch in zip(X_batches, Y_pos_batches, Y_quat_batches, Y_marker_batches):
            model.zero_grad()

            pred_marker, pred_pos, pred_quat = model(X_batch[:-1,:,:])

            #OLD!!! TODO scale loss properly! when position number are big then quat loss basically does nothing!
            #        maybe scale by average norm of positions then maybe weight constantly e.g. 1/2 since the focus is on position,
            #        but that should make a difference anyways
            #loss_pos = loss_function(pred_pos, Y_pos_batch[1:,:,:])
            #loss_quat = loss_function(pred_quat, Y_quat_batch[1:, :, :])
            #loss = loss_pos + loss_quat
            loss = loss_function(pred_marker, Y_marker_batch[1:, :, :])
            loss.backward()
            optimizer.step()
            avg_loss += loss
        avg_loss /= len(Y_pos_batches)

        model.eval()
        with torch.no_grad():
            pred_marker, pred_pos, pred_quat = model(X_test[:-1,:,:])
            #loss_pos = loss_function(pred_pos, Y_pos_test[1:, :, :])
            #loss_quat = loss_function(pred_quat, Y_quat_test[1:, :, :])
            #loss = loss_pos + loss_quat
            loss = loss_function(pred_marker, Y_marker_test[1:, :, :])
            print("training loss: {ltrain:2.4f}, test loss: {ltest:2.4f}".format(ltrain=avg_loss.data, ltest=loss.data))
    torch.save(model.state_dict(), 'weights/lstm_4marker_to_pos')



def eval():
    test_size = 100
    [X_test, Y_pos_test, Y_quat_test, Y_marker_test] = complete_gen(pattern, 0, test_size, both=False)

    model.load_state_dict(torch.load('weights/lstm_4marker_to_pos'))
    model.eval()

    marker_preds, predicted_pos, predicted_quats = model(X_test[:-1, :, :])

    for n in range(10):
        with torch.no_grad():
            data = torch.stack((predicted_pos[:,n,:], Y_pos_test[1:,n,:]),1)
            center = torch.mean(data, dim=(0,1)).numpy()
            visualize_tracking(predicted_pos[:, n, :].detach().numpy() - center,
                               predicted_quats[:, n, :].detach().numpy(),
                               Y_pos_test[1:, n, :].detach().numpy() - center,
                               Y_quat_test[1:, n, :].detach().numpy(),
                               X_test[:-1, n, :].numpy() - np.tile(center, 4),
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