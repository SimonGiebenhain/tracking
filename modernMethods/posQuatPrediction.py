import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from pyquaternion import Quaternion as Quaternion


from vizTracking import visualize_tracking



BATCH_SIZE = 50

N_train = 200*BATCH_SIZE
N_eval = 20*BATCH_SIZE
T = 200
fc1_dim = 200
fc2_dim = 250
fc3_dim = 100
hidden_dim = 35
fc_out_1_size = 50
input_dim = 4

NUM_EPOCHS = 20

weight_file = 'weights/pos_quat_lstm.npy'


# TODO: pattern as additional input! in order to learn with arbitrary patterns
# TODO: try with bird behaviour and simulated pattern
# TODO: missing dtections and false positives
# TODO: easier data or more powerful net? It cannot even really overfit...

####################################################################################
######### REPORT #########
#
####################################################################################




#marker1 = np.array([0, 0, 0])
#marker2 = np.array([0, 0, 0.5])
#marker3 = np.array([-0.7, -1, 0])
#marker4 = np.array([1.1, -1, 0.8])
#
#pattern = np.stack([marker1, marker2, marker3, marker4], axis=0)
#
#stacked_marker1 = np.tile(marker1, reps=(T-1, BATCH_SIZE, 1))
#stacked_marker2 = np.tile(marker2, reps=(T-1, BATCH_SIZE, 1))
#stacked_marker3 = np.tile(marker3, reps=(T-1, BATCH_SIZE, 1))
#stacked_marker4 = np.tile(marker4, reps=(T-1, BATCH_SIZE, 1))
#
#stacked_marker1 = torch.from_numpy(stacked_marker1).float()
#stacked_marker2 = torch.from_numpy(stacked_marker2).float()
#stacked_marker3 = torch.from_numpy(stacked_marker3).float()
#stacked_marker4 = torch.from_numpy(stacked_marker4).float()

def gen_pattern(N):
    # one marker is always the origin
    marker1 = np.zeros([T, N, 3])

    # The others have to be generated such that they span a 3-dim space
    marker2 = np.random.uniform(-1, 1, [T, N, 3])

    marker3 = np.random.uniform(-1, 1, [T, N, 3])
    ortho_marker2 = np.stack([marker2[:,:,1] + marker2[:, :, 2], -marker2[:, :, 0], -marker2[:, :, 0]], axis=2)
    marker3 = (marker3 + ortho_marker2) / 2

    ortho_marker23 = np.cross(marker2, marker3)
    scale_marker2 = np.random.uniform(-1, 1, [T, N, 1])
    scale_marker3 = np.random.uniform(-1, 1, [T, N, 1])
    scale_ortho = np.random.uniform(0.1, 1, [T, N, 1]) * np.random.choice([-1, 1], size=[T, N, 1], replace=True)
    marker4 = scale_marker2 * marker2 + scale_marker3 + marker3 + scale_ortho * ortho_marker23

    pattern = np.stack([marker1, marker2, marker3, marker4], axis=2)

    return pattern, marker1, marker2, marker3, marker4


def gen_quats(length, dims):
    theta_range = np.random.uniform(1,2)
    theta = np.linspace(-theta_range * np.pi, theta_range * np.pi, length)
    z_range = np.random.randint(1,10)
    z = np.random.uniform(1,3)*np.sin(np.linspace(0, z_range, length))
    rx = np.abs(z) ** np.random.uniform(1.5,3)*np.abs(np.random.rand())  + 1
    ry = np.abs(z) ** np.random.uniform(1.5,3)*np.abs(np.random.rand())  + 1
    x = rx**1.5 * np.sin(theta)
    y = ry**1.5 * np.cos(theta)
    w = 1 + np.random.uniform(0.5, 4)*np.sin(theta)*np.cos(theta)**2
    quats = np.stack([w, x, y, z], axis=1)
    quats = quats / np.expand_dims(np.sqrt(np.sum(np.square(quats), axis=1)), axis=1)
    #quats = np.tile(np.array([1, 0, 0, 0]), [length, 1])
    return quats


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


def scale_trajectory(trajectory):
    max_pos = np.max(trajectory, axis=0)
    min_pos = np.min(trajectory, axis=0)
    movement_range = max_pos - min_pos
    return  5 * (trajectory / movement_range)


def center_trajectory(trajectory):
    center = np.mean(trajectory, axis=0)
    return trajectory - center


def gen_pos(N):

    pos_train = np.zeros([T, N, 3], dtype=np.float32)
    pos_test = np.zeros([T, N, 3], dtype=np.float32)

    for n in range(N):
        trajectory = Gen_Spirals(T, 3)
        trajectory = center_trajectory(trajectory)
        trajectory = scale_trajectory(trajectory)
        pos_train[:, n, :] = trajectory

    for n in range(N):
        trajectory = Gen_Spirals(T, 3)
        trajectory = center_trajectory(trajectory)
        trajectory = scale_trajectory(trajectory)
        pos_test[:, n, :] = trajectory

    return pos_train, pos_test


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


# TODO: vecotrize with qrot() and by shuffling markers while generating them
def gen_training_data(N):

    quat_train = np.zeros([T, N, 4], dtype=np.float32)
    quat_test = np.zeros([T, N, 4], dtype=np.float32)

    for n in range(N):
        quat_train[:, n, :] = gen_quats(T, input_dim)
        quat_test[:, n, :] = gen_quats(T, input_dim)

    [pos_train, pos_test] = gen_pos(N)
    pos_train_stacked = np.tile(pos_train, [1, 1, 4])
    pos_test_stacked = np.tile(pos_test, [1, 1, 4])

    pattern_train, _, _, _, _ = gen_pattern(N)
    pattern_test, _, _, _, _ = gen_pattern(N)


    X_train = np.zeros([T, N, 12])
    X_train_shuffled = np.zeros([T, N, 12])
    X_test = np.zeros([T, N, 12])
    X_test_shuffled = np.zeros([T, N, 12])


    for t in range(T):
        for n in range(N):
            p_train = pattern_train[t, n, :, :]
            p_train_copy = np.copy(p_train)

            q = Quaternion(quat_train[t, n, :])
            np.random.shuffle(p_train_copy)
            rotated_pattern = (q.rotation_matrix @ p_train_copy.T).T
            X_train_shuffled[t, n, :] = np.reshape(rotated_pattern, -1)

            rotated_pattern = (q.rotation_matrix @ p_train.T).T
            X_train[t, n, :] = np.reshape(rotated_pattern, -1)

            p_test = pattern_test[t, n, :, :]
            p_test_copy = pattern_test[t, n, :, :]

            q = Quaternion(quat_test[t, n, :])
            np.random.shuffle(p_test_copy)
            rotated_pattern = (q.rotation_matrix @ p_test_copy.T).T
            X_test_shuffled[t, n, :] = np.reshape(rotated_pattern, -1)

            rotated_pattern = (q.rotation_matrix @ p_test.T).T
            X_test[t, n, :] = np.reshape(rotated_pattern, -1)

    X_train = X_train + pos_train_stacked
    X_train_shuffled = X_train_shuffled + pos_train_stacked
    X_test = X_test + pos_test_stacked
    X_test_shuffled = X_test_shuffled + pos_test_stacked


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

    return torch.from_numpy(X_train_shuffled).float(), \
           torch.from_numpy(quat_train).float(), \
           torch.from_numpy(X_train).float(), \
           torch.from_numpy(pos_train).float(), \
           torch.from_numpy(X_test_shuffled).float(), \
           torch.from_numpy(quat_test).float(), \
           torch.from_numpy(X_test).float() , \
           torch.from_numpy(pos_test).float(), \
           torch.from_numpy(pattern_train).float(), \
           torch.from_numpy(pattern_test).float()


class LSTMTracker(nn.Module):

    def __init__(self, hidden_dim, input_dim):
        super(LSTMTracker, self).__init__()
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(12+12, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, fc3_dim)
        self.lstm = nn.LSTM(fc3_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2quat1 = nn.Linear(hidden_dim, fc_out_1_size)
        self.hidden2quat2 = nn.Linear(fc_out_1_size, 4)

        self.hidden2pos = nn.Linear(hidden_dim, 3)

        self.dropout = nn.Dropout(p=0.15)

    def forward(self, detections, patterns):
        marker1 = patterns[:, :, 0, :].contiguous()
        marker2 = patterns[:, :, 1, :].contiguous()
        marker3 = patterns[:, :, 2, :].contiguous()
        marker4 = patterns[:, :, 3, :].contiguous()
        x = torch.cat([detections, patterns.view(T-1, -1, 12)], dim=2)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        lstm_out, _ = self.lstm(x)
        x = F.relu(self.hidden2quat1(lstm_out))
        quat_space = self.hidden2quat2(x)
        pos_space = self.hidden2pos(lstm_out)
        # maybe leave out wenn not using pose error
        quat_norm = torch.sqrt(torch.sum(torch.pow(quat_space, 2, ), dim=2))
        quat_space = quat_space / torch.unsqueeze(quat_norm, dim=2)

        rotated_marker1 = qrot(quat_space, marker1) + pos_space
        rotated_marker2 = qrot(quat_space, marker2) + pos_space
        rotated_marker3 = qrot(quat_space, marker3) + pos_space
        rotated_marker4 = qrot(quat_space, marker4) + pos_space
        rotated_pattern = torch.cat([rotated_marker1,
                             rotated_marker2,
                             rotated_marker3,
                             rotated_marker4], dim=2)
        return quat_space, pos_space, rotated_pattern


model = LSTMTracker(hidden_dim, input_dim)
# TODO: try pose error!!
# TODO: respect antipodal pair as well!
loss_function_pose = nn.MSELoss() #nn.L1Loss() #nn.MSELoss()
loss_function_quat = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train():
    #TODO: store patterns
    #TODO: adapt architecture to acceot patterns as well
    [X_train_shuffled, Y_quat_train, X_train, Y_pos_train, X_test_shuffled, Y_quat_test, X_test, Y_pos_test, pattern_train, pattern_test] = gen_training_data(N_train)
    model.train()
    for epoch in range(NUM_EPOCHS):
        batches = torch.split(X_train_shuffled, BATCH_SIZE, 1)
        quat_truth_batches = torch.split(Y_quat_train, BATCH_SIZE, 1)
        pos_truth_batches = torch.split(Y_pos_train, BATCH_SIZE, 1)
        batches_not_shuffled = torch.split(X_train, BATCH_SIZE, 1)
        pattern_batches = torch.split(pattern_train, BATCH_SIZE, 1)
        avg_loss_pose = 0
        avg_loss_quat = 0
        avg_loss_pos = 0
        for batch, quat_truth_batch, pos_truth_batch, batch_not_shuffled, pattern_batch in zip(batches, quat_truth_batches, pos_truth_batches, batches_not_shuffled, pattern_batches):
            model.zero_grad()

            pred_quat, pred_pos, pred_markers = model(batch[:-1, :, :], pattern_batch[:-1, :, :, :])

            loss_pose = loss_function_pose(pred_markers, batch_not_shuffled[1:, :, :])
            loss_quat = loss_function_quat(pred_quat, quat_truth_batch[1:, :, :])
            loss_pos = loss_function_pose(pred_pos, pos_truth_batch[1:, :, :])

            loss = loss_pose + loss_quat + loss_pos
            loss.backward()
            optimizer.step()
            avg_loss_pose += loss_pose
            avg_loss_quat += loss_quat
            avg_loss_pos += loss_pos
        avg_loss_pose /= len(batches)
        avg_loss_quat /= len(batches)
        avg_loss_pos /= len(batches)

        model.eval()
        with torch.no_grad():
            pred_quat, pred_pos, preds  = model(X_test_shuffled[:-1,:,:], pattern_test[:-1, :, :, :])
            loss_pose = loss_function_pose(preds, X_test[1:,:,:])
            loss_quat = loss_function_quat(pred_quat, Y_quat_test[1:, :, :])
            loss_pos = loss_function_pose(pred_pos, Y_pos_test[1:, :, :])
            print("TrainPoseLoss: {train_pose:2.4f}, TrainQuatLoss: {train_quat:2.4f}  TrainPosLoss: {train_pos:2.4f}\t TestPoseLoss: {test_pose:2.4f}, TestQuatLoss: {test_quat:2.4f}, TestPosLoss: {test_pos:2.4f}".format(
                train_pose=avg_loss_pose.data, train_quat=avg_loss_quat.data, train_pos=avg_loss_pos.data, test_pose=loss_pose, test_quat=loss_quat, test_pos=loss_pos.data))
    torch.save(model.state_dict(), weight_file)



def eval():
    [X_train, Y_quat_train, _, Y_pos_train, X_test, Y_quat_test, _, Y_pos_test, pattern] = gen_training_data(N_eval)

    model.load_state_dict(torch.load(weight_file))
    model.eval()


    with torch.no_grad():
        quat_preds, pos_preds, _ = model(X_test[:-1, :, :])

        for n in range(10):
            visualize_tracking(pos_preds[:, n, :].detach().numpy(),
                               quat_preds[:, n, :].detach().numpy(),
                               Y_pos_test[1:, n, :].detach().numpy(),
                               Y_quat_test[1:, n, :].detach().numpy(),
                               X_test[:-1, n, :].numpy(),
                               pattern)


train()
#eval()