use_colab = False



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from math import sqrt

import gc
import os
import pandas as pd
import re
from datetime import datetime


import matplotlib.pyplot as plt
colab_path_prefix = '/content/gdrive/My Drive/'
if use_colab:
    from google.colab import drive
    drive.mount('/content/gdrive')

    import sys
    sys.path.append(colab_path_prefix + 'pyquaternion')
from pyquaternion import Quaternion as Quaternion

if not use_colab:
    from vizTracking import visualize_tracking




if use_colab:
    generated_data_dir = colab_path_prefix + 'generated_data'
else:
    generated_data_dir = 'generated_training_data'

MODEL_NAME = 'LSTM'
TASK = 'PosQuatPred; '

drop_some_dets = True
add_false_positives = False
add_noise = False
NOISE_STD = 0.01
use_const_pat = True

if drop_some_dets:
    TASK += 'Drop some detections; '
if add_false_positives:
    TASK += 'Add false positives; '
if add_noise:
    TASK += 'Noisy; '
if use_const_pat:
    TASK += 'ConstantPattern; '
else:
    TASK += 'ChangingPattern; '



BATCH_SIZE = 50
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
STRONG_DROPOUT_RATE = 0.25
WEAK_DROPOUT_RATE = 0.1

CHECKPOINT_INTERVAL = 10
save_model_every_interval = False
save_best_model = True

N_train = 20*BATCH_SIZE
N_eval = int(N_train/10)
N_test = int(N_train/2)

T = 200

fc1_det_dim = 300
fc2_det_dim = 400
fc3_det_dim = 500
fc4_det_dim = 350
fc5_det_dim = 300

fc1_pat_dim = 200
fc2_pat_dim = 300
fc3_pat_dim = 300
fc4_pat_dim = 200

hidden_dim = 150

fc1_quat_size = 100
fc2_quat_size = 50

fc1_pos_size = 150
fc2_pos_size = 50

#fc1_det_dim = 250
#fc2_det_dim = 300
#fc3_det_dim = 300
#fc4_det_dim = 250
#fc1_pat_dim = 200
#fc2_pat_dim = 250
#fc3_pat_dim = 150
#hidden_dim = 75
#fc_out_1_size = 30


# TODO: improve false positives, i.e. roll where last real detection is, delete rest, roll where to put fp between real detections

# TODO: compare LSTM to simple RNN, then design custom cell, maybe with recurrent dropout and batch normalization

# TODO: why does eval() not work with old model???


# TODO: proper noise model, look at noise behaviour for individual markers inside pattern

# TODO: generate bird behaviour from VICON predictions, if not enough use kalman filter predictions

# TODO: ask for pidgeon data?

# TODO: write goldlücke a mail



# TODO: at some point figure out good dropout rate, and other hyper parms

# TODO: try relative inupts instead of absolute detections

# TODO: look at hand notes for more ideas, e.g. multi modal predictions

# TODO: read some papers for more ideas


####################################################################################
######### REPORT #########
#
####################################################################################


def normalize_vector(v):
    return v / np.sum(v)


def gen_folder_name(task, name):
    if len(task) > 20:
        short_task = re.sub('[^A-Z]', '', task)
    else:
        short_task = task
    if len(name) > 20:
        short_name = re.sub('[^A-Z]', '', name)
    else:
        short_name = name
    now = datetime.now()
    dt_str = now.strftime("%d.%m.%Y@%H:%M:%S")
    return short_name + '_' + short_task + '_' + dt_str


class TrainingData():
    def __init__(self):
        self.X_train = None
        self.X_train_shuffled = None
        self.quat_train = None
        self.pos_train = None
        self.pattern_train = None

        self.X_test = None
        self.X_test_shuffled = None
        self.quat_test = None
        self.pos_test = None
        self.pattern_test = None

        self.X_eval = None
        self.X_eval_shuffled = None
        self.quat_eval = None
        self.pos_eval = None
        self.pattern_eval = None

    def set_train_data(self, data_dict):
        self.X_train = data_dict['X']
        self.X_train_shuffled = data_dict['X_shuffled']
        self.quat_train = data_dict['quat']
        self.pos_train = data_dict['pos']
        self.pattern_train = data_dict['pattern']

    def set_test_data(self, data_dict):
        self.X_test = data_dict['X']
        self.X_test_shuffled = data_dict['X_shuffled']
        self.quat_test = data_dict['quat']
        self.pos_test = data_dict['pos']
        self.pattern_test = data_dict['pattern']

    def set_eval_data(self, data_dict):
        self.X_eval = data_dict['X']
        self.X_eval_shuffled = data_dict['X_shuffled']
        self.quat_eval = data_dict['quat']
        self.pos_eval = data_dict['pos']
        self.pattern_eval = data_dict['pattern']

    def load_data(self, dir_name, N_train, N_test, N_eval):
        self.X_train = np.load(dir_name + '/X_train' + str(N_train) + '.npy')
        self.X_train_shuffled = np.load(dir_name + '/X_train_shuffled' + str(N_train) + '.npy')
        self.quat_train = np.load(dir_name + '/quat_train' + str(N_train) + '.npy')
        self.pos_train = np.load(dir_name + '/pos_train' + str(N_train) + '.npy')
        self.pattern_train = np.load(dir_name + '/pattern_train' + str(N_train) + '.npy')

        self.X_test = np.load(dir_name + '/X_test' + str(N_test) + '.npy')
        self.X_test_shuffled = np.load(dir_name + '/X_test_shuffled' + str(N_test) + '.npy')
        self.quat_test = np.load(dir_name + '/quat_test' + str(N_test) + '.npy')
        self.pos_test = np.load(dir_name + '/pos_test' + str(N_test) + '.npy')
        self.pattern_test = np.load(dir_name + '/pattern_test' + str(N_test) + '.npy')

        self.X_eval = np.load(dir_name + '/X_eval' + str(N_eval) + '.npy')
        self.X_eval_shuffled = np.load(dir_name + '/X_eval_shuffled' + str(N_eval) + '.npy')
        self.quat_eval = np.load(dir_name + '/quat_eval' + str(N_eval) + '.npy')
        self.pos_eval = np.load(dir_name + '/pos_eval' + str(N_eval) + '.npy')
        self.pattern_eval = np.load(dir_name + '/pattern_eval' + str(N_eval) + '.npy')

        print('Loaded data successfully!')

    def save_data(self, dir_name):
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        N = np.shape(self.X_train)[1]
        np.save(dir_name + '/X_train' + str(N) + '.npy', self.X_train)
        np.save(dir_name + '/X_train_shuffled' + str(N) + '.npy', self.X_train_shuffled)
        np.save(dir_name + '/quat_train' + str(N) + '.npy', self.quat_train)
        np.save(dir_name + '/pos_train' + str(N) + '.npy', self.pos_train)
        np.save(dir_name + '/pattern_train' + str(N) + '.npy', self.pattern_train)

        N = np.shape(self.X_test)[1]
        np.save(dir_name + '/X_test' + str(N) + '.npy', self.X_test)
        np.save(dir_name + '/X_test_shuffled' + str(N) + '.npy', self.X_test_shuffled)
        np.save(dir_name + '/quat_test' + str(N) + '.npy', self.quat_test)
        np.save(dir_name + '/pos_test' + str(N) + '.npy', self.pos_test)
        np.save(dir_name + '/pattern_test' + str(N) + '.npy', self.pattern_test)

        N = np.shape(self.X_eval)[1]
        np.save(dir_name + '/X_eval' + str(N) + '.npy', self.X_eval)
        np.save(dir_name + '/X_eval_shuffled' + str(N) + '.npy', self.X_eval_shuffled)
        np.save(dir_name + '/quat_eval' + str(N) + '.npy', self.quat_eval)
        np.save(dir_name + '/pos_eval' + str(N) + '.npy', self.pos_eval)
        np.save(dir_name + '/pattern_eval' + str(N) + '.npy', self.pattern_eval)

        print('Saved data successfully!')

    def convert_to_torch(self):
        if use_colab:
            self.X_train = torch.from_numpy(self.X_train).float().cuda()
            self.X_train_shuffled = torch.from_numpy(self.X_train_shuffled).float().cuda()
            self.quat_train = torch.from_numpy(self.quat_train).float().cuda()
            self.pos_train = torch.from_numpy(self.pos_train).float().cuda()
            self.pattern_train = torch.from_numpy(self.pattern_train).float().cuda()

            self.X_test = torch.from_numpy(self.X_test).float().cuda()
            self.X_test_shuffled = torch.from_numpy(self.X_test_shuffled).float().cuda()
            self.quat_test = torch.from_numpy(self.quat_test).float().cuda()
            self.pos_test = torch.from_numpy(self.pos_test).float().cuda()
            self.pattern_test = torch.from_numpy(self.pattern_test).float().cuda()

            self.X_eval = torch.from_numpy(self.X_eval).float().cuda()
            self.X_eval_shuffled = torch.from_numpy(self.X_eval_shuffled).float().cuda()
            self.quat_eval = torch.from_numpy(self.quat_eval).float().cuda()
            self.pos_eval = torch.from_numpy(self.pos_eval).float().cuda()
            self.pattern_eval = torch.from_numpy(self.pattern_eval).float().cuda()
        else:
            self.X_train = torch.from_numpy(self.X_train).float()
            self.X_train_shuffled = torch.from_numpy(self.X_train_shuffled).float()
            self.quat_train = torch.from_numpy(self.quat_train).float()
            self.pos_train = torch.from_numpy(self.pos_train).float()
            self.pattern_train = torch.from_numpy(self.pattern_train).float()

            self.X_test = torch.from_numpy(self.X_test).float()
            self.X_test_shuffled = torch.from_numpy(self.X_test_shuffled).float()
            self.quat_test = torch.from_numpy(self.quat_test).float()
            self.pos_test = torch.from_numpy(self.pos_test).float()
            self.pattern_test = torch.from_numpy(self.pattern_test).float()

            self.X_eval = torch.from_numpy(self.X_eval).float()
            self.X_eval_shuffled = torch.from_numpy(self.X_eval_shuffled).float()
            self.quat_eval = torch.from_numpy(self.quat_eval).float()
            self.pos_eval = torch.from_numpy(self.pos_eval).float()
            self.pattern_eval = torch.from_numpy(self.pattern_eval).float()

        print('Converted data to torch format.')

    def convert_to_numpy(self):
        self.X_train = self.X_train.numpy()
        self.X_train_shuffled = self.X_train_shuffled.numpy()
        self.quat_train = self.quat_train.numpy()
        self.pos_train = self.pos_train.numpy()
        self.pattern_train = self.pattern_train.numpy()

        self.X_test = self.X_test.numpy()
        self.X_test_shuffled = self.X_test_shuffled.numpy()
        self.quat_test = self.quat_test.numpy()
        self.pos_test = self.pos_test.numpy()
        self.pattern_test = self.pattern_test.numpy()

        self.X_eval = self.X_eval.numpy()
        self.X_eval_shuffled = self.X_eval_shuffled.numpy()
        self.quat_eval = self.quat_eval.numpy()
        self.pos_eval = self.pos_eval.numpy()
        self.pattern_eval = self.pattern_eval.numpy()

        print('Converted data to numpy format.')


class HyperParams():
    def __init__(self, n_train, n_test, T, batch_size, optimizer, init_learning_rate, lr_scheduler, lr_scheduler_params,
                 dropout_rate, batch_norm_type, loss_type, comments):
        self.batch_size = batch_size
        self.n_train = n_train
        self.n_test = n_test
        self.T = T
        self.optimizer = type(optimizer).__name__
        self.init_learning_rate = init_learning_rate
        self.lr_scheduler = type(lr_scheduler).__name__
        self.lr_scheduler_params = ''
        for key in lr_scheduler_params:
            self.lr_scheduler_params += key + ': ' + str(lr_scheduler_params[key]) + ', '
        self.dropout_rate = dropout_rate
        self.batch_norm_type = batch_norm_type
        self.loss_type = loss_type
        self.comments = comments

    def gen_string(self):
        description = 'Description of hyper parameters of the model: \n'
        description = description + 'Batch size: \t {} \n'.format(self.batch_size)
        description = description + 'Train size: \t {} \n'.format(self.n_train)
        description = description + 'Test size: \t {} \n'.format(self.n_test)
        description = description + 'Sequence length: \t {} \n'.format(self.T)
        description = description + 'Optimizer: \t ' + self.optimizer + '\n'
        description = description + 'Initial LR: \t {} \n'.format(self.init_learning_rate)
        description = description + 'LR scheduler: \t ' + self.lr_scheduler + '\n'
        description = description + 'Scheduler params: \t' + self.lr_scheduler_params + '\n'
        description = description + 'Dropout rate: \t {} \n'.format(self.dropout_rate)
        description = description + 'Batch norm: \t ' + self.batch_norm_type + '\n'
        description = description + 'Loss type: \t ' + self.loss_type + '\n'
        description = description + 'Comments: \t ' + self.comments + '\n'
        return description


class TrainingLogger():
    def __init__(self, name, task, hyper_params):
        self.name = name
        self.task = task
        self.best_val_loss = np.Inf
        self.best_pose_loss = np.Inf
        self.hyper_params = hyper_params

        self.progress_dict = {'train_pose': [], 'train_quat': [], 'train_pos': [],
                              'test_pose': [], 'test_quat': [], 'test_pos': [],
                              'learning_rate': []}

        if use_colab:
            self.folder_name = colab_path_prefix + gen_folder_name(task, name)
        else:
            self.folder_name = gen_folder_name(task, name)

        working_dir = os.getcwd()
        if not use_colab:
            path = working_dir + '/' + self.folder_name
        else:
            path = self.folder_name
        os.mkdir(path)

    def log_epoch(self, train_pose, train_quat, train_pos, test_pose, test_quat, test_pos, model, lr):
        self.progress_dict['train_pose'].append(train_pose)
        self.progress_dict['train_quat'].append(train_quat)
        self.progress_dict['train_pos'].append(train_pos)
        self.progress_dict['test_pose'].append(test_pose)
        self.progress_dict['test_quat'].append(test_quat)
        self.progress_dict['test_pos'].append(test_pos)
        self.progress_dict['learning_rate'].append(lr)

        if self.best_pose_loss > test_pose:
            self.best_pose_loss = test_pose

        val_loss = test_pos + test_quat
        if save_best_model and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(model, self.folder_name + '/model_best.npy')
        epoch = len(self.progress_dict['train_pose'])
        if save_model_every_interval and epoch % CHECKPOINT_INTERVAL == 0:
            torch.save(model, self.folder_name + '/model_epoch_{}'.format(epoch))

    def save_log(self):
        description = self.hyper_params.gen_string()
        description = 'MODEL: \t' + self.name + '\n TASK: \t' + self.task + '\n' + description
        description = description + 'Best pose loss: \t {loss:1.8f} \n'.format(loss=self.best_pose_loss)
        file = open(self.folder_name + '/' + 'hyper_params.txt', 'w+')
        file.write(description)
        file.close()

        training_progress_df = pd.DataFrame.from_dict(self.progress_dict)
        training_progress_df.to_csv(self.folder_name + '/' + 'training_progress.csv')

        if not use_colab:
            training_progress = training_progress_df.to_numpy()
            print(np.shape(training_progress))
            for k, key in enumerate(self.progress_dict):
                if key == 'learning_rate':
                    continue
                if k < 3:
                    c_train = np.array([1, k * 0.2, 0, 1])
                    plt.plot(training_progress[:, k], c=c_train, label=key)
                else:
                    c_test = np.array([0, k * 0.2, 1, 1])
                    plt.plot(training_progress[:, k], c=c_test, label=key)
            plt.legend()
            plt.savefig(self.folder_name + '/training_progress.png', format='png')


def gen_pattern_constant(N):
    marker1 = np.array([0, 0, 0])
    marker2 = np.array([0, 0, 0.5])
    marker3 = np.array([-0.7, -1, 0])
    marker4 = np.array([1.1, -1, 0.8])

    pattern = np.stack([marker1, marker2, marker3, marker4], axis=0)

    stacked_marker1 = np.tile(marker1, reps=(T, N, 1))
    stacked_marker2 = np.tile(marker2, reps=(T, N, 1))
    stacked_marker3 = np.tile(marker3, reps=(T, N, 1))
    stacked_marker4 = np.tile(marker4, reps=(T, N, 1))

    pattern = np.stack([stacked_marker1, stacked_marker2, stacked_marker3, stacked_marker4], axis=2)

    # stacked_marker1 = torch.from_numpy(stacked_marker1).float()
    # stacked_marker2 = torch.from_numpy(stacked_marker2).float()
    # stacked_marker3 = torch.from_numpy(stacked_marker3).float()
    # stacked_marker4 = torch.from_numpy(stacked_marker4).float()

    return pattern, stacked_marker1, stacked_marker2, stacked_marker3, stacked_marker4


def gen_pattern(N):
    # one marker is always the origin
    marker1 = np.zeros([T, N, 3])

    # The others have to be generated such that they span a 3-dim space
    marker2 = np.random.uniform(-1, 1, [1, N, 3])

    marker3 = np.random.uniform(-1, 1, [1, N, 3])
    ortho_marker2 = np.stack([marker2[:, :, 1] + marker2[:, :, 2], -marker2[:, :, 0], -marker2[:, :, 0]], axis=2)
    marker3 = (marker3 + ortho_marker2) / 2

    ortho_marker23 = np.cross(marker2, marker3)
    scale_marker2 = np.random.uniform(-1, 1, [1, N, 1])
    scale_marker3 = np.random.uniform(-1, 1, [1, N, 1])
    scale_ortho = np.random.uniform(0.1, 1, [1, N, 1]) * np.random.choice([-1, 1], size=[1, N, 1], replace=True)
    marker4 = scale_marker2 * marker2 + scale_marker3 + marker3 + scale_ortho * ortho_marker23

    marker2 = np.tile(marker2 / 10, [T, 1, 1])
    marker3 = np.tile(marker3 / 10, [T, 1, 1])
    marker4 = np.tile(marker4 / 10, [T, 1, 1])

    pattern = np.stack([marker1, marker2, marker3, marker4], axis=2)

    return pattern, marker1, marker2, marker3, marker4


def gen_quats(length):
    theta_range = np.random.uniform(1, 2)
    theta = np.linspace(-theta_range * np.pi, theta_range * np.pi, length)
    z_range = np.random.randint(1, 10)
    z = np.random.uniform(1, 3) * np.sin(np.linspace(0, z_range, length))
    rx = np.abs(z) ** np.random.uniform(1.5, 3) * np.abs(np.random.rand()) + 1
    ry = np.abs(z) ** np.random.uniform(1.5, 3) * np.abs(np.random.rand()) + 1
    x = rx ** 1.5 * np.sin(theta)
    y = ry ** 1.5 * np.cos(theta)
    w = 1 + np.random.uniform(0.5, 4) * np.sin(theta) * np.cos(theta) ** 2
    quats = np.stack([w, x, y, z], axis=1)
    quats = quats / np.expand_dims(np.sqrt(np.sum(np.square(quats), axis=1)), axis=1)
    # quats = np.tile(np.array([1, 0, 0, 0]), [length, 1])
    return quats


def Gen_Spirals(length, dims=2):
    theta_range = np.random.randint(1, 10)
    theta = np.linspace(-theta_range * np.pi, theta_range * np.pi, length)
    z_range = np.random.randint(15, 45)
    z = np.random.uniform(1, 3) * np.sin(np.linspace(0, z_range, length))
    rx = np.abs(z) ** np.random.uniform(1.5, 3) * np.abs(np.random.rand()) + 1
    ry = np.abs(z) ** np.random.uniform(1.5, 3) * np.abs(np.random.rand()) + 1
    x = rx ** 1.5 * np.sin(theta)
    y = ry ** 1.5 * np.cos(theta)

    return np.stack([x, y, z], axis=1) + 5 * np.random.uniform(low=-5, high=5, size=[1, dims])


def scale_trajectory(trajectory):
    max_pos = np.max(trajectory, axis=0)
    min_pos = np.min(trajectory, axis=0)
    movement_range = max_pos - min_pos
    return 5 * (trajectory / movement_range)


def center_trajectory(trajectory):
    center = np.mean(trajectory, axis=0)
    return trajectory - center


def gen_pos(N):
    pos = np.zeros([T, N, 3], dtype=np.float32)

    for n in range(N):
        trajectory = Gen_Spirals(T, 3)
        trajectory = center_trajectory(trajectory)
        trajectory = scale_trajectory(trajectory)
        pos[:, n, :] = trajectory

    return pos


def qrot(q, v):
    # TODO can I change this function to also work with constant v and changing quaternions?
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
        size = int(q_batch_size / BATCH_SIZE)
        v = v.repeat([1, size, 1])

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


# TODO: vecotrize with qrot() and by shuffling markers while generating them
def gen_data(N):
    quat = np.zeros([T, N, 4], dtype=np.float32)

    for n in range(N):
        quat[:, n, :] = gen_quats(T)

    pos = gen_pos(N)

    pos_stacked = np.tile(pos, [1, 1, 4])
    if add_false_positives:
        pos_stacked_fp = np.tile(pos, [1, 1, 5])
    else:
        pos_stacked_fp = np.tile(pos, [1, 1, 4])

    if use_const_pat:
        pattern, _, _, _, _ = gen_pattern_constant(N)
    else:
        pattern, _, _, _, _ = gen_pattern(N)

    X = np.zeros([T, N, 12])
    if add_false_positives:
        X_shuffled = np.zeros([T, N, 15])
    else:
        X_shuffled = np.zeros([T, N, 12])

    for t in range(T):
        for n in range(N):
            p = pattern[t, n, :, :]
            p_copy = np.copy(p)

            q = Quaternion(quat[t, n, :])
            np.random.shuffle(p_copy)
            rotated_pattern = (q.rotation_matrix @ p_copy.T).T
            if add_false_positives:
                rotated_pattern = np.concatenate([rotated_pattern, np.ones([1, 3]) * -1000], axis=0)
                if np.random.uniform(0, 1) < 0.1:
                    if drop_some_dets:
                        if np.random.uniform(0, 1) < 0.5:
                            if np.random.uniform(0, 1) < 0.5:
                                rotated_pattern[3, :] = np.ones([1, 3]) * -1000
                                rotated_pattern[2, :] = np.random.uniform(-2, 2, [1, 3])
                            else:
                                rotated_pattern[3, :] = np.random.uniform(-2, 2, [1, 3])
                        else:
                            rotated_pattern[4, :] = np.random.uniform(-2, 2, [1, 3])
                    else:
                        rotated_pattern[4, :] = np.random.uniform(-2, 2, [1, 3])
                else:
                    if drop_some_dets and np.random.uniform(0, 1) < 0.5:
                        rotated_pattern[3, :] = np.array([-1000, -1000, -1000])
                        if drop_some_dets and np.random.uniform(0, 1) < 0.5:
                            rotated_pattern[2, :] = np.array([-1000, -1000, -1000])
            else:
                if drop_some_dets and np.random.uniform(0, 1) < 0.5:
                    rotated_pattern[3, :] = np.array([-1000, -1000, -1000])
                    if drop_some_dets and np.random.uniform(0, 1) < 0.5:
                        rotated_pattern[2, :] = np.array([-1000, -1000, -1000])
            dets = np.reshape(rotated_pattern, -1)
            if add_noise:
                noise = np.random.normal(0, NOISE_STD, np.shape(dets))
                dets = dets + noise
            X_shuffled[t, n, :] = dets

            rotated_pattern = (q.rotation_matrix @ p.T).T
            X[t, n, :] = np.reshape(rotated_pattern, -1)
    X = X + pos_stacked
    X_shuffled = X_shuffled + pos_stacked_fp

    return {'X': X, 'X_shuffled': X_shuffled, 'quat': quat, 'pos': pos, 'pattern': pattern}


class customLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(customLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):

        if hidden is None:
            hidden = self._init_hidden(x)

        h, c = hidden
        h = h.view(h.size(1), -1)
        c = c.view(c.size(1), -1)
        x = x.view(x.size(1), -1)

        # Linear mappings
        preact = self.i2h(x) + self.h2h(h)

        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]

        c_t = torch.mul(c, f_t) + torch.mul(i_t, g_t)

        h_t = torch.mul(o_t, c_t.tanh())

        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)
        return (h_t, c_t)
        #return h_t, (h_t, c_t)

    @staticmethod
    def _init_hidden(input_):
        #h = torch.zeros_like(input_.view(1, input_.size(1), -1))
        #c = torch.zeros_like(input_.view(1, input_.size(1), -1))
        h = torch.zeros([1, input_.size(1), hidden_dim]).cuda()
        c = torch.zeros([1, input_.size(1), hidden_dim]).cuda()
        return h, c


class customLSTM(nn.Module):

    def __init__(self, hidden_size, bias=True):
        super().__init__()

        if use_const_pat:
            self.lstm_cell = customLSTMCell(fc5_det_dim, hidden_dim, bias)
        else:
            self.lstm_cell = customLSTMCell(fc5_det_dim + fc4_pat_dim, hidden_dim, bias)

        if add_false_positives:
            self.fc1_det = nn.Linear(15, fc1_det_dim)
        else:
            self.fc1_det = nn.Linear(12, fc1_det_dim)

        self.fc2_det = nn.Linear(fc1_det_dim, fc2_det_dim)
        self.fc3_det = nn.Linear(fc2_det_dim, fc3_det_dim)
        self.fc4_det = nn.Linear(fc3_det_dim, fc4_det_dim)
        self.fc5_det = nn.Linear(fc4_det_dim, fc5_det_dim)

        if not use_const_pat:
            self.fc1_pat = nn.Linear(12, fc1_pat_dim)
            self.fc2_pat = nn.Linear(fc1_pat_dim, fc2_pat_dim)
            self.fc3_pat = nn.Linear(fc2_pat_dim, fc3_pat_dim)
            self.fc4_pat = nn.Linear(fc3_pat_dim, fc4_pat_dim)

        # self.fc1_combo = nn.Linear(fc2_pat_dim + fc3_det_dim, fc1_combo_dim)
        # self.fc2_combo = nn.Linear(fc1_combo_dim, fc2_combo_dim)

        if use_const_pat:
            self.lstm = nn.LSTM(fc5_det_dim, hidden_dim)
        else:
            self.lstm = nn.LSTM(fc5_det_dim + fc4_pat_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2quat1 = nn.Linear(hidden_dim, fc1_quat_size)
        self.hidden2quat2 = nn.Linear(fc1_quat_size, fc2_quat_size)
        self.hidden2quat3 = nn.Linear(fc2_quat_size, 4)

        self.hidden2pos1 = nn.Linear(hidden_dim, fc1_pos_size)
        self.hidden2pos2 = nn.Linear(fc1_pos_size, fc2_pos_size)
        self.hidden2pos3 = nn.Linear(fc2_pos_size, 3)

        self.strong_dropout = nn.Dropout(p=STRONG_DROPOUT_RATE)
        self.weak_dropout = nn.Dropout(p=WEAK_DROPOUT_RATE)


    def forward(self, detections, patterns, hidden=None):
        # input is of dimensionalty (time, input_size, ...)
        marker1 = patterns[:, :, 0, :].contiguous()
        marker2 = patterns[:, :, 1, :].contiguous()
        marker3 = patterns[:, :, 2, :].contiguous()
        marker4 = patterns[:, :, 3, :].contiguous()

        x = self.weak_dropout(F.relu(self.fc1_det(detections)))
        x = self.strong_dropout(F.relu(self.fc2_det(x)))
        x = self.strong_dropout(F.relu(self.fc3_det(x)))
        x = self.strong_dropout(F.relu(self.fc4_det(x)))
        x = self.strong_dropout(F.relu(self.fc5_det(x)))

        if not use_const_pat:
            x_pat = self.weak_dropout(F.relu(self.fc1_pat(patterns.view(T - 1, -1, 12))))
            x_pat = self.strong_dropout(F.relu(self.fc2_pat(x_pat)))
            x_pat = self.strong_dropout(F.relu(self.fc3_pat(x_pat)))
            x_pat = self.strong_dropout(F.relu(self.fc4_pat(x_pat)))
            x = torch.cat([x, x_pat], dim=2)

        x = torch.unsqueeze(x, dim=0)
        lstm_out = []

        for a in torch.unbind(x, dim=1):
            hidden = self.lstm_cell(a, hidden)
            lstm_out.append(hidden[0].clone())

        lstm_out = torch.squeeze(torch.stack(lstm_out, dim=1))


        x_quat = self.weak_dropout(F.relu(self.hidden2quat1(lstm_out)))
        x_quat = self.weak_dropout(F.relu(self.hidden2quat2(x_quat)))
        x_quat = self.hidden2quat3(x_quat)

        x_pos = self.weak_dropout(F.relu(self.hidden2pos1(lstm_out)))
        x_pos = self.weak_dropout(F.relu(self.hidden2pos2(x_pos)))
        x_pos = self.hidden2pos3(x_pos)

        quat_norm = torch.sqrt(torch.sum(torch.pow(x_quat, 2, ), dim=2))
        x_quat = x_quat / torch.unsqueeze(quat_norm, dim=2)

        rotated_marker1 = qrot(x_quat, marker1) + x_pos
        rotated_marker2 = qrot(x_quat, marker2) + x_pos
        rotated_marker3 = qrot(x_quat, marker3) + x_pos
        rotated_marker4 = qrot(x_quat, marker4) + x_pos
        rotated_pattern = torch.cat([rotated_marker1,
                                     rotated_marker2,
                                     rotated_marker3,
                                     rotated_marker4], dim=2)

        return x_quat, x_pos, rotated_pattern


class LSTMTracker(nn.Module):
    def __init__(self, hidden_dim):
        super(LSTMTracker, self).__init__()
        self.hidden_dim = hidden_dim
        if add_false_positives:
            self.fc1_det = nn.Linear(15, fc1_det_dim)
        else:
            self.fc1_det = nn.Linear(12, fc1_det_dim)
        self.fc2_det = nn.Linear(fc1_det_dim, fc2_det_dim)
        self.fc3_det = nn.Linear(fc2_det_dim, fc3_det_dim)
        self.fc4_det = nn.Linear(fc3_det_dim, fc4_det_dim)
        self.fc5_det = nn.Linear(fc4_det_dim, fc5_det_dim)

        if not use_const_pat:
            self.fc1_pat = nn.Linear(12, fc1_pat_dim)
            self.fc2_pat = nn.Linear(fc1_pat_dim, fc2_pat_dim)
            self.fc3_pat = nn.Linear(fc2_pat_dim, fc3_pat_dim)
            self.fc4_pat = nn.Linear(fc3_pat_dim, fc4_pat_dim)

        # self.fc1_combo = nn.Linear(fc2_pat_dim + fc3_det_dim, fc1_combo_dim)
        # self.fc2_combo = nn.Linear(fc1_combo_dim, fc2_combo_dim)

        if use_const_pat:
            self.lstm = nn.LSTM(fc5_det_dim, hidden_dim)
        else:
            self.lstm = nn.LSTM(fc5_det_dim + fc4_pat_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2quat1 = nn.Linear(hidden_dim, fc1_quat_size)
        self.hidden2quat2 = nn.Linear(fc1_quat_size, fc2_quat_size)
        self.hidden2quat3 = nn.Linear(fc2_quat_size, 4)

        self.hidden2pos1 = nn.Linear(hidden_dim, fc1_pos_size)
        self.hidden2pos2 = nn.Linear(fc1_pos_size, fc2_pos_size)
        self.hidden2pos3 = nn.Linear(fc2_pos_size, 3)

        self.strong_dropout = nn.Dropout(p=STRONG_DROPOUT_RATE)
        self.weak_dropout = nn.Dropout(p=WEAK_DROPOUT_RATE)

    def forward(self, detections, patterns):
        marker1 = patterns[:, :, 0, :].contiguous()
        marker2 = patterns[:, :, 1, :].contiguous()
        marker3 = patterns[:, :, 2, :].contiguous()
        marker4 = patterns[:, :, 3, :].contiguous()

        x = self.weak_dropout(F.relu(self.fc1_det(detections)))
        x = self.strong_dropout(F.relu(self.fc2_det(x)))
        x = self.strong_dropout(F.relu(self.fc3_det(x)))
        x = self.strong_dropout(F.relu(self.fc4_det(x)))
        x = self.strong_dropout(F.relu(self.fc5_det(x)))

        if not use_const_pat:
            x_pat = self.weak_dropout(F.relu(self.fc1_pat(patterns.view(T - 1, -1, 12))))
            x_pat = self.strong_dropout(F.relu(self.fc2_pat(x_pat)))
            x_pat = self.strong_dropout(F.relu(self.fc3_pat(x_pat)))
            x_pat = self.strong_dropout(F.relu(self.fc4_pat(x_pat)))
            x = torch.cat([x, x_pat], dim=2)

        # x_combo = self.dropout(F.relu(self.fc1_combo(x_combo)))
        # x_combo = self.dropout(F.relu(self.fc2_combo(x_combo)))

        # x = torch.cat([x_det, x_pat], dim=2)

        # x = x_det - x_pat

        lstm_out, _ = self.lstm(x)

        x_quat = self.weak_dropout(F.relu(self.hidden2quat1(lstm_out)))
        x_quat = self.weak_dropout(F.relu(self.hidden2quat2(x_quat)))
        x_quat = self.hidden2quat3(x_quat)

        x_pos = self.weak_dropout(F.relu(self.hidden2pos1(lstm_out)))
        x_pos = self.weak_dropout(F.relu(self.hidden2pos2(x_pos)))
        x_pos = self.hidden2pos3(x_pos)

        quat_norm = torch.sqrt(torch.sum(torch.pow(x_quat, 2, ), dim=2))
        x_quat = x_quat / torch.unsqueeze(quat_norm, dim=2)

        rotated_marker1 = qrot(x_quat, marker1) + x_pos
        rotated_marker2 = qrot(x_quat, marker2) + x_pos
        rotated_marker3 = qrot(x_quat, marker3) + x_pos
        rotated_marker4 = qrot(x_quat, marker4) + x_pos
        rotated_pattern = torch.cat([rotated_marker1,
                                     rotated_marker2,
                                     rotated_marker3,
                                     rotated_marker4], dim=2)

        return x_quat, x_pos, rotated_pattern


model = customLSTM(hidden_dim, bias=True)
#model = LSTMTracker(hidden_dim)
if use_colab and torch.cuda.is_available():
    print('USING CUDA DEVICE')
    model.cuda()
    device = torch.device('cuda')
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

# TODO: respect antipodal pair as well!
loss_function_pose = nn.MSELoss()
loss_function_quat = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
lr_scheduler_params = {'mode': 'min', 'factor': 0.5, 'patience': 3, 'min_lr': 1e-06, 'cooldown': 4}
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=lr_scheduler_params['mode'],
                                                       factor=lr_scheduler_params['factor'],
                                                       patience=lr_scheduler_params['patience'],
                                                       cooldown= lr_scheduler_params['cooldown'],
                                                       verbose=True, min_lr=lr_scheduler_params['min_lr'])

hyper_params = HyperParams(N_train, N_eval, T, BATCH_SIZE, optimizer, LEARNING_RATE, scheduler, lr_scheduler_params,
                           STRONG_DROPOUT_RATE, 'NONE', 'l2 on pos + 5* l1 on quat', '')
logger = TrainingLogger(MODEL_NAME, TASK, hyper_params)
name = logger.folder_name + '/model_best.npy'


def train(data):
    for gci in range(10):
        gc.collect()

    for epoch in range(1, NUM_EPOCHS + 1):
        gc.collect()
        model.train()

        batches = torch.split(data.X_train_shuffled, BATCH_SIZE, 1)
        quat_truth_batches = torch.split(data.quat_train, BATCH_SIZE, 1)
        pos_truth_batches = torch.split(data.pos_train, BATCH_SIZE, 1)
        batches_not_shuffled = torch.split(data.X_train, BATCH_SIZE, 1)
        pattern_batches = torch.split(data.pattern_train, BATCH_SIZE, 1)
        avg_loss_pose = 0
        avg_loss_quat = 0
        avg_loss_pos = 0
        n_batches_per_epoch = len(batches)

        for k, [batch, quat_truth_batch, pos_truth_batch, batch_not_shuffled, pattern_batch] in enumerate(
                zip(batches, quat_truth_batches, pos_truth_batches, batches_not_shuffled, pattern_batches)):
            model.zero_grad()

            pred_quat, pred_pos, pred_markers = model(batch[:-1, :, :], pattern_batch[:-1, :, :, :])

            loss_pose = loss_function_pose(pred_markers, batch_not_shuffled[1:, :, :])
            loss_quat = loss_function_quat(pred_quat, quat_truth_batch[1:, :, :])
            loss_pos = loss_function_pose(pred_pos, pos_truth_batch[1:, :, :])

            loss = loss_pos + loss_quat
            loss.backward()
            optimizer.step()
            avg_loss_pose += loss_pose
            avg_loss_quat += loss_quat
            avg_loss_pos += loss_pos

            #if k % int(n_batches_per_epoch / 10) == 0:
            #    model.eval()
            #    with torch.no_grad():
            #        pred_quat, pred_pos, preds = model(data.X_eval_shuffled[:-1, :, :], data.pattern_eval[:-1, :, :, :])
            #        loss_pose = loss_function_pose(preds, data.X_eval[1:, :, :])
            #        loss_quat = loss_function_quat(pred_quat, data.quat_eval[1:, :, :])
            #        loss_pos = loss_function_pose(pred_pos, data.pos_eval[1:, :, :])
            #        val_loss = loss_pos + loss_quat
            #        print(val_loss)
            #        scheduler.step(val_loss)
            #    model.train()

        avg_loss_pose /= len(batches)
        avg_loss_quat /= len(batches)
        avg_loss_pos /= len(batches)

        model.eval()
        with torch.no_grad():
            pred_quat, pred_pos, preds = model(data.X_test_shuffled[:-1, :, :], data.pattern_test[:-1, :, :, :])
            loss_pose = loss_function_pose(preds, data.X_test[1:, :, :])
            loss_quat = loss_function_quat(pred_quat, data.quat_test[1:, :, :])
            loss_pos = loss_function_pose(pred_pos, data.pos_test[1:, :, :])
            val_loss = loss_pos + loss_quat
            for param_group in optimizer.param_groups:
                learning_rate = param_group['lr']
                print("Epoch: {epoch:2d}, Learning Rate: {learning_rate:1.8f} \n TrainPoseLoss: {train_pose:2.4f}, "
                      "TrainQuatLoss: {train_quat:2.4f}  TrainPosLoss: {train_pos:2.4f} \t "
                      "TestPoseLoss: {test_pose:2.4f}, TestQuatLoss: {test_quat:2.4f}, TestPosLoss: {test_pos:2.4f}".format(
                    epoch=epoch, learning_rate=learning_rate,
                    train_pose=avg_loss_pose.data, train_quat=avg_loss_quat.data, train_pos=avg_loss_pos.data,
                    test_pose=loss_pose, test_quat=loss_quat, test_pos=loss_pos.data))
            scheduler.step(val_loss)
            logger.log_epoch(avg_loss_pose.data, avg_loss_quat.data, avg_loss_pos.data,
                             loss_pose.data, loss_quat.data, loss_pos.data,
                             model,
                             learning_rate)
        if use_colab:
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

    logger.save_log()


def eval(name):
    data = TrainingData()
    data.load_data(generated_data_dir, N_train, N_test, N_eval)
    data.convert_to_torch()
    model = torch.load(name)
    model.eval()

    with torch.no_grad():
        quat_preds, pos_preds, _ = model(data.X_test[:-1, :, :], data.pattern_test[:-1, :, :])

        for n in range(10):
            visualize_tracking(pos_preds[:, n, :].detach().numpy(),
                               quat_preds[:, n, :].detach().numpy(),
                               data.pos_test[1:, n, :].detach().numpy(),
                               data.quat_test[1:, n, :].detach().numpy(),
                               data.X_test_shuffled[:-1, n, :].numpy(),
                               data.pattern_test[0, n, :].numpy())


#######################################################################################################################

if use_colab:
    data = TrainingData()
    data.set_train_data(gen_data(N_train))
    data.set_test_data(gen_data(N_test))
    data.set_eval_data(gen_data(N_eval))
    data.save_data(generated_data_dir)
    data.load_data(generated_data_dir, N_train, N_test, N_eval)
    data.convert_to_torch()

else:
    data = TrainingData()
    #data.set_train_data(gen_data(N_train))
    #data.set_test_data(gen_data(N_test))
    #data.set_eval_data(gen_data(N_eval))
    #data.save_data(generated_data_dir)
    data.load_data(generated_data_dir, N_train, N_test, N_eval)
    data.convert_to_torch()

gc.collect()
train(data)

gc.collect()
if not use_colab:
    eval(name)
    #eval('LSTM_baseline/model_best.npy')