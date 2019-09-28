import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from pyquaternion import Quaternion as Quaternion


from vizTracking import visualize_tracking



BATCH_SIZE = 50

N_train = 100*BATCH_SIZE
N_eval = 10*BATCH_SIZE
T = 200
fc1_dim = 150
fc2_dim = 200
fc3_dim = 100
hidden_dim = 25
fc_out_1_size = 35
input_dim = 4

NUM_EPOCHS = 10

weight_file = 'weights/quat_lstm.npy'


####################################################################################
######### REPORT #########
# Results of experiments without shuffling and the following architecture:
# hidden_dim=25, fc_out_1_size=35, no dropout
# It seems that a combination of both the quat loss(l1-loss) and the pose loss(l2-loss) increases performance over either individual loss

#TODO: redo experiment, bc. pose loss has to be adapted when shuffled!!
#TODO: think about if its possible to predict order of markers or something

# With shuffling the detections:
# breaks down with architecture above.
# although the quat loss is still relatively low, the pose error is quite large

# After adding embedding network:
# adding (150,200,100) embedding network didn't do the trick (with quat loss only). It was still a bit too wabbley
#
####################################################################################




marker1 = np.array([0, 0, 0])
marker2 = np.array([0, 0, 0.5])
marker3 = np.array([-0.7, -1, 0])
marker4 = np.array([1.1, -1, 0.8])

pattern = np.stack([marker1, marker2, marker3, marker4], axis=0)

stacked_marker1 = np.tile(marker1, reps=(T-1, BATCH_SIZE, 1))
stacked_marker2 = np.tile(marker2, reps=(T-1, BATCH_SIZE, 1))
stacked_marker3 = np.tile(marker3, reps=(T-1, BATCH_SIZE, 1))
stacked_marker4 = np.tile(marker4, reps=(T-1, BATCH_SIZE, 1))

stacked_marker1 = torch.from_numpy(stacked_marker1).float()
stacked_marker2 = torch.from_numpy(stacked_marker2).float()
stacked_marker3 = torch.from_numpy(stacked_marker3).float()
stacked_marker4 = torch.from_numpy(stacked_marker4).float()


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
    if not q.shape[:-1] == v.shape[:-1]:
        q_batch_size = list(q.shape)[1]
        size = int(q_batch_size/BATCH_SIZE)
        v = v.repeat([1, size, 1])

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

    X_train = np.zeros([T, N, 12])
    X_train_shuffled = np.zeros([T, N, 12])
    X_test = np.zeros([T, N, 12])
    X_test_shuffled = np.zeros([T, N, 12])


    p = np.copy(patternq)

    for t in range(T):
        for n in range(N):
            q = Quaternion(quat_train[t, n, :])
            np.random.shuffle(p)
            rotated_pattern = (q.rotation_matrix @ p.T).T
            X_train_shuffled[t, n, :] = np.reshape(rotated_pattern, -1)

            rotated_pattern = (q.rotation_matrix @ pattern.T).T
            X_train[t, n, :] = np.reshape(rotated_pattern, -1)

            q = Quaternion(quat_test[t, n, :])
            np.random.shuffle(p)
            rotated_pattern = (q.rotation_matrix @ p.T).T
            X_test_shuffled[t, n, :] = np.reshape(rotated_pattern, -1)

            rotated_pattern = (q.rotation_matrix @ pattern.T).T
            X_test[t, n, :] = np.reshape(rotated_pattern, -1)

    #maxi1 = max(np.max(quat_train[:, :, 0]), np.max(quat_test[:, :, 0])) / 5
    #maxi2 = max(np.max(quat_train[:, :, 1]), np.max(quat_test[:, :, 1])) / 5
    #maxi3 = max(np.max(quat_train[:, :, 2]), np.max(quat_test[:, :, 2])) / 5
    #maxi4 = max(np.max(quat_train[:, :, 3]), np.max(quat_test[:, :, 3])) / 5

   #quat_train[:, :, 0] = quat_train[:, :, 0]# / maxi1
   #quat_train[:, :, 1] = quat_train[:, :, 1]# / maxi2
   #quat_train[:, :, 2] = quat_train[:, :, 2]# / maxi3
   #quat_train[:, :, 3] = quat_train[:, :, 3]# / maxi4

   #quat_test[:, :, 0] = quat_test[:, :, 0]# / maxi1
   #quat_test[:, :, 1] = quat_test[:, :, 1]# / maxi2
   #quat_test[:, :, 2] = quat_test[:, :, 2]# / maxi3
   #quat_test[:, :, 3] = quat_test[:, :, 3]# / maxi4

   #stacked_marker1 = np.tile(marker1, reps=(T, N, 1))
   #stacked_marker2 = np.tile(marker2, reps=(T, N, 1))
   #stacked_marker3 = np.tile(marker3, reps=(T, N, 1))
   #stacked_marker4 = np.tile(marker4, reps=(T, N, 1))




   ##print(np.shape(stacked_marker1))
   ##print(np.shape(quat_train))
   #assert np.shape(stacked_marker1)[:2] == np.shape(quat_train)[:2]

   #stacked_marker1 = torch.from_numpy(stacked_marker1).float()
   #stacked_marker2 = torch.from_numpy(stacked_marker2).float()
   #stacked_marker3 = torch.from_numpy(stacked_marker3).float()
   #stacked_marker4 = torch.from_numpy(stacked_marker4).float()

   #quat_train = torch.from_numpy(quat_train).float()
   #quat_test = torch.from_numpy(quat_test).float()

   #rotated_marker1_train = qrot(quat_train, stacked_marker1)
   #rotated_marker2_train = qrot(quat_train, stacked_marker2)
   #rotated_marker3_train = qrot(quat_train, stacked_marker3)
   #rotated_marker4_train = qrot(quat_train, stacked_marker4)
   #X_train = torch.cat([rotated_marker1_train,
   #                    rotated_marker2_train,
   #                    rotated_marker3_train,
   #                    rotated_marker4_train], dim=2)

   #rotated_marker1_test = qrot(quat_test, stacked_marker1)
   #rotated_marker2_test = qrot(quat_test, stacked_marker2)
   #rotated_marker3_test = qrot(quat_test, stacked_marker3)
   #rotated_marker4_test = qrot(quat_test, stacked_marker4)
   #X_test = torch.cat( [rotated_marker1_test,
   #                    rotated_marker2_test,
   #                    rotated_marker3_test,
   #                    rotated_marker4_test], dim=2)
   #

    return torch.from_numpy(X_train_shuffled).float(), torch.from_numpy(quat_train).float(), \
           torch.from_numpy(X_train).float(), \
           torch.from_numpy(X_test_shuffled).float(), torch.from_numpy(quat_test).float(), \
           torch.from_numpy(X_test).float() , \
           pattern


class LSTMTracker(nn.Module):

    def __init__(self, hidden_dim, input_dim):
        super(LSTMTracker, self).__init__()
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(12, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, fc3_dim)
        self.lstm = nn.LSTM(fc3_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2quat1 = nn.Linear(hidden_dim, fc_out_1_size)
        self.hidden2quat2 = nn.Linear(fc_out_1_size, 4)

        self.dropout = nn.Dropout(p=0.15)

    def forward(self, detections):
        x = self.dropout(F.relu(self.fc1(detections)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        lstm_out, _ = self.lstm(x)
        x = F.relu(self.hidden2quat1(lstm_out))
        quat_space = self.hidden2quat2(x)
        # maybe leave out wenn not using pose error
        quat_norm = torch.sqrt(torch.sum(torch.pow(quat_space, 2, ), dim=2))
        quat_space = quat_space / torch.unsqueeze(quat_norm, dim=2)
        #print(quat_space.shape)
        #print(stacked_marker1.shape)

        rotated_marker1 = qrot(quat_space, stacked_marker1)
        #print('HALL=')
        #print(rotated_marker1.shape)
        rotated_marker2 = qrot(quat_space, stacked_marker2)
        rotated_marker3 = qrot(quat_space, stacked_marker3)
        rotated_marker4 = qrot(quat_space, stacked_marker4)
        rotated_pattern = torch.cat([rotated_marker1,
                             rotated_marker2,
                             rotated_marker3,
                             rotated_marker4], dim=2)
        return quat_space, rotated_pattern


model = LSTMTracker(hidden_dim, input_dim)
# TODO: try pose error!!
# TODO: respect antipodal pair as well!
loss_function_pose = nn.MSELoss() #nn.L1Loss() #nn.MSELoss()
loss_function_quat = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

def train():
    [X_train_shuffled, Y_train, X_train, X_test_shuffled, Y_test, X_test, pattern] = gen_training_data(N_train)
    model.train()
    for epoch in range(NUM_EPOCHS):
        batches = torch.split(X_train_shuffled, BATCH_SIZE, 1)
        truth_batches = torch.split(Y_train, BATCH_SIZE, 1)
        batches_not_shuffled = torch.split(X_train, BATCH_SIZE, 1)
        avg_loss_pose = 0
        avg_loss_quat = 0
        for batch, truth_batch, batch_not_shuffled in zip(batches, truth_batches, batches_not_shuffled):
            model.zero_grad()

            pred_quat, pred_markers = model(batch[:-1, :, :])

            loss_pose = loss_function_pose(pred_markers, batch_not_shuffled[1:, :, :])
            loss_quat = loss_function_quat(pred_quat, truth_batch[1:, :, :])
            loss = loss_pose + loss_quat
            loss.backward()
            optimizer.step()
            avg_loss_pose += loss_pose
            avg_loss_quat += loss_quat
        avg_loss_pose /= len(batches)
        avg_loss_quat /= len(batches)

        model.eval()
        with torch.no_grad():
            pred_quat, preds  = model(X_test_shuffled[:-1,:,:])
            loss_pose = loss_function_pose(preds, X_test[1:,:,:])
            loss_quat = loss_function_quat(pred_quat, Y_test[1:, :, :])
            print("TrainPoseLoss: {train_pose:2.4f}, TrainQuatLoss: {train_quat:2.4f} \t TestPoseLoss: {test_pose:2.4f}, TestQuatLoss: {test_quat:2.4f}".format(
                train_pose=avg_loss_pose.data, train_quat=avg_loss_quat.data, test_pose=loss_pose, test_quat=loss_quat))
    torch.save(model.state_dict(), weight_file)



def eval():
    [X_train, Y_train, _, X_test, Y_test, _, pattern] = gen_training_data(N_eval)

    model.load_state_dict(torch.load(weight_file))
    model.eval()


    with torch.no_grad():
        quat_preds, _ = model(X_test[:-1, :, :])

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