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

from math import ceil


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


MODEL_NAME = 'LSTM_BIRDS'
TASK = 'PosQuatPred; '

drop_some_dets = False
add_false_positives = False
add_noise = False
NOISE_STD = 0.01
use_const_pat = True
generate_data = False

if use_colab:
    if use_const_pat:
      generated_data_dir = colab_path_prefix + 'generated_data' + '_const_pat'
    else:
      generated_data_dir = colab_path_prefix + 'generated_data'

else:
    if use_const_pat:
      generated_data_dir = 'generated_data' + '_const_pat'
    else:
      generated_data_dir = 'generated_training_data'

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



BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
STRONG_DROPOUT_RATE = 0.05
WEAK_DROPOUT_RATE = 0.01

CHECKPOINT_INTERVAL = 10
save_model_every_interval = False
save_best_model = True

N_train = 600*BATCH_SIZE
N_test = int(N_train/10)

T = 100

fc1_det_dim = 20
fc2_det_dim = 30
fc3_det_dim = 30
fc4_det_dim = 30
fc5_det_dim = 20

fc1_pat_dim = 20
fc2_pat_dim = 30
fc3_pat_dim = 30
fc4_pat_dim = 15

hidden_dim = 20

fc1_quat_size = 20
fc2_quat_size = 5

fc1_pos_size = 20
fc2_pos_size = 5

#fc1_det_dim = 250
#fc2_det_dim = 300
#fc3_det_dim = 300
#fc4_det_dim = 250
#fc1_pat_dim = 200
#fc2_pat_dim = 250
#fc3_pat_dim = 150
#hidden_dim = 75
#fc_out_1_size = 30


# TODO: MATLAB: HOW TO REINITIALIZE LOST TRACKS??

# TODO: make position independent,

# TODO: use noise model to drop detections
# TODO: think about noise model for false positives

# TODO: write some preprocessing methods, i.e. pattern should have std. dev of 1 of norm of markers or something

# TODO: improve false positives, i.e. roll where last real detection is, delete rest, roll where to put fp between real detections

# TODO: be patient when working with varying patterns, compare custom architecture(may be too slow!) to in-built LSTM
# TODO: compare LSTM to simple RNN, and GRU, then design custom cell,
# TODO: experiment with recurrent dropout and batch normalization
# TODO: peephole lstm

# TODO: at some point figure out good dropout rate, and other hyper parms

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

    def set_train_data(self, data_dict):
        self.X_train = data_dict['X']
        self.X_train_shuffled = data_dict['X_shuffled']
        self.quat_train = data_dict['quat']
        self.pos_train = data_dict['pos']
        self.pattern_train = data_dict['pattern']
        self.make_relative()

    def set_test_data(self, data_dict):
        self.X_test = data_dict['X']
        self.X_test_shuffled = data_dict['X_shuffled']
        self.quat_test = data_dict['quat']
        self.pos_test = data_dict['pos']
        self.pattern_test = data_dict['pattern']
        self.make_relative()

    def make_relative(self):
        self.X_train_shuffled = make_detections_relative(self.X_train_shuffled, self.pos_train)
        self.X_test_shuffled = make_detections_relative(self.X_test_shuffled, self.pos_test)
        self.delta_pos_train = make_pos_relative(self.pos_train)
        self.delta_pos_test = make_pos_relative(self.pos_test)
        self.pattern_train = self.pattern_train[1:, :, :]
        self.pattern_test = self.pattern_test[1:, :, :]
        self.X_train = self.X_train[1:, :, :]
        self.X_test = self.X_test[1:, :, :]
        self.quat_train = self.quat_train[1:, :, :]
        self.quat_test = self.quat_test[1:, :, :]
        self.pos_train = self.pos_train[1:, :, :]
        self.pos_test = self.pos_test[1:, :, :]

    def load_data(self, dir_name, N_train, N_test, name):
        dname = dir_name + '_' + name
        if generate_data:
            postfix = str(N_train) + '.npy'
        else:
            postfix = '.npy'
        self.X_train = np.load(dname + '/X_train' + postfix)
        self.X_train_shuffled = np.load(dname + '/X_train_shuffled' + postfix)
        self.quat_train = np.load(dname + '/quat_train' + postfix)
        self.pos_train = np.load(dname + '/pos_train' + postfix)
        self.pattern_train = np.load(dname + '/pattern_train' + postfix)

        if generate_data:
            postfix = str(N_test) + '.npy'
        else:
            postfix = '.npy'
        self.X_test = np.load(dname + '/X_test' + postfix)
        self.X_test_shuffled = np.load(dname + '/X_test_shuffled' + postfix)
        self.quat_test = np.load(dname + '/quat_test' + postfix)
        self.pos_test = np.load(dname + '/pos_test' + postfix)
        self.pattern_test = np.load(dname + '/pattern_test' + postfix)
        self.make_relative()

        print('Loaded data successfully!')

    def save_data(self, dir_name, name):
        dname = dir_name + '_' + name
        if not os.path.exists(dname):
            os.mkdir(dname)
        N = np.shape(self.X_train)[1]
        if generate_data:
            postfix = str(N) + '.npy'
        else:
            postfix = '.npy'
        np.save(dname + '/X_train' + postfix, self.X_train)
        np.save(dname + '/X_train_shuffled' + postfix, self.X_train_shuffled)
        np.save(dname + '/quat_train' + postfix, self.quat_train)
        np.save(dname + '/pos_train' + postfix, self.pos_train)
        np.save(dname + '/pattern_train' + postfix, self.pattern_train)

        N = np.shape(self.X_test)[1]
        np.save(dname + '/X_test' + postfix, self.X_test)
        np.save(dname + '/X_test_shuffled' + postfix, self.X_test_shuffled)
        np.save(dname + '/quat_test' + postfix, self.quat_test)
        np.save(dname + '/pos_test' + postfix, self.pos_test)
        np.save(dname + '/pattern_test' + postfix, self.pattern_test)

        print('Saved data successfully!')

    def convert_to_torch(self):
        if use_colab:
            self.X_train = torch.from_numpy(self.X_train).float().cuda()
            self.X_train_shuffled = torch.from_numpy(self.X_train_shuffled).float().cuda()
            self.quat_train = torch.from_numpy(self.quat_train).float().cuda()
            self.pos_train = torch.from_numpy(self.pos_train).float().cuda()
            self.pattern_train = torch.from_numpy(self.pattern_train).float().cuda()
            self.delta_pos_train = torch.from_numpy(self.delta_pos_train).float().cuda()

            self.X_test = torch.from_numpy(self.X_test).float().cuda()
            self.X_test_shuffled = torch.from_numpy(self.X_test_shuffled).float().cuda()
            self.quat_test = torch.from_numpy(self.quat_test).float().cuda()
            self.pos_test = torch.from_numpy(self.pos_test).float().cuda()
            self.pattern_test = torch.from_numpy(self.pattern_test).float().cuda()
            self.delta_pos_test = torch.from_numpy(self.delta_pos_test).float().cuda()

        else:
            self.X_train = torch.from_numpy(self.X_train).float()
            self.X_train_shuffled = torch.from_numpy(self.X_train_shuffled).float()
            self.quat_train = torch.from_numpy(self.quat_train).float()
            self.pos_train = torch.from_numpy(self.pos_train).float()
            self.pattern_train = torch.from_numpy(self.pattern_train).float()
            self.delta_pos_train = torch.from_numpy(self.delta_pos_train).float()


            self.X_test = torch.from_numpy(self.X_test).float()
            self.X_test_shuffled = torch.from_numpy(self.X_test_shuffled).float()
            self.quat_test = torch.from_numpy(self.quat_test).float()
            self.pos_test = torch.from_numpy(self.pos_test).float()
            self.pattern_test = torch.from_numpy(self.pattern_test).float()
            self.delta_pos_test = torch.from_numpy(self.delta_pos_test).float()

        print('Converted data to torch format.')

    def convert_to_numpy(self):
        self.X_train = self.X_train.numpy()
        self.X_train_shuffled = self.X_train_shuffled.numpy()
        self.quat_train = self.quat_train.numpy()
        self.pos_train = self.pos_train.numpy()
        self.pattern_train = self.pattern_train.numpy()
        self.delta_pos_train = self.delta_pos_train.numpy()

        self.X_test = self.X_test.numpy()
        self.X_test_shuffled = self.X_test_shuffled.numpy()
        self.quat_test = self.quat_test.numpy()
        self.pos_test = self.pos_test.numpy()
        self.pattern_test = self.pattern_test.numpy()
        self.delta_pos_test = self.delta_pos_test.numpy()

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


def gen_pattern_(N):
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

    marker2 = np.tile(marker2, [T, 1, 1])
    marker3 = np.tile(marker3, [T, 1, 1])
    marker4 = np.tile(marker4, [T, 1, 1])

    pattern = np.stack([marker1, marker2, marker3, marker4], axis=2)

    return pattern, marker1, marker2, marker3, marker4


def gen_pattern(N):
    if use_colab:
        patterns = np.load(colab_path_prefix + 'patterns.npy')
    else:
        patterns = np.load('data/patterns.npy')

    marker1 = np.zeros([T, N, 3])
    marker2 = np.zeros([T, N, 3])
    marker3 = np.zeros([T, N, 3])
    marker4 = np.zeros([T, N, 3])

    chosen_pat_idx = np.random.choice(np.arange(0, 9), [N])

    for n in range(N):
        marker1[:, n, :] = np.squeeze(np.tile(patterns[chosen_pat_idx[n], 0, :], [T, 1, 1])) / 10
        marker2[:, n, :] = np.squeeze(np.tile(patterns[chosen_pat_idx[n], 1, :], [T, 1, 1])) / 10
        marker3[:, n, :] = np.squeeze(np.tile(patterns[chosen_pat_idx[n], 2, :], [T, 1, 1])) / 10
        marker4[:, n, :] = np.squeeze(np.tile(patterns[chosen_pat_idx[n], 3, :], [T, 1, 1])) / 10

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
def gen_data(N_train, N_test):

    print(N_train)
    print(N_test)
    if not generate_data:
        ratio = N_train / (N_train + N_test)
        pos = np.load('data/cleaned_kalman_pos_all.npy')
        true_N = pos.shape[1]
        N_train = ceil(true_N * ratio)
        pos_train = pos[:, :N_train, :]
        pos_test = pos[:, N_train:, :]
        N_test = pos_test.shape[1]
        print(N_train)
        print(N_test)

    def gen_datum(N, pos_data=None):

        if generate_data:
            pos = gen_pos(N)
        else:
            pos = pos_data


        quat = np.zeros([T, N, 4], dtype=np.float32)

        for n in range(N):
            quat[:, n, :] = gen_quats(T)


        pos_stacked = np.tile(pos, [1, 1, 4])
        if add_false_positives:
            pos_stacked_fp = np.tile(pos, [1, 1, 5])
        else:
            pos_stacked_fp = np.tile(pos, [1, 1, 4])

        if use_const_pat:
            pattern, _, _, _, _ = gen_pattern_constant(N)
        else:
            pattern, _, _, _, _ = gen_pattern(N)
        pattern = pattern / 10

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

    if generate_data:
        return (gen_datum(N_train), gen_datum(N_test))
    else:
        return (gen_datum(N_train, pos_train), gen_datum(N_test, pos_test)), N_train, N_test


def make_detections_relative(dets, pos):
    tiled_pos = np.tile(pos, [1, 1, 4])
    return dets[1:, :, :] - tiled_pos[:-1, :, :]

def make_pos_relative(pos):
    return pos[1:, :, :] - pos[:-1, :, :]

class customLSTMCell(nn.Module):
    def __init__(self, input_size_det, input_size_pat, hidden_size, bias=True):
        super(customLSTMCell, self).__init__()
        self.input_size_det = input_size_det
        self.input_size_pat = input_size_pat
        self.hidden_size = hidden_size
        self.bias = bias
        self.i2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)

        self.det2vec = nn.Linear(input_size_det, self.hidden_size)
        if not use_const_pat:
            self.pat2vec = nn.Linear(input_size_pat, self.hidden_size)

        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)

        self.reset_parameters()


    def reset_parameters(self):
        std = 1.0 / sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x_det, x_pat, hidden):

        if hidden is None:
            hidden = self._init_hidden(x_det)

        h, c = hidden

        hh = self.fc1(h)
        if use_const_pat:
           x = F.relu(self.det2vec(x_det) + hh)
        else:
            x = F.relu(self.det2vec(x_det) + hh) - F.relu(self.pat2vec(x_pat) + hh)

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
        if use_colab:
            h = torch.zeros([1, input_.size(1), hidden_dim]).cuda()
            c = torch.zeros([1, input_.size(1), hidden_dim]).cuda()
        else:
            h = torch.zeros([1, input_.size(1), hidden_dim])
            c = torch.zeros([1, input_.size(1), hidden_dim])
        return h, c


class customLSTM(nn.Module):

    def __init__(self, hidden_size, bias=True):
        super().__init__()

        if use_const_pat:
            self.lstm_cell = customLSTMCell(fc5_det_dim, None, hidden_dim, bias)
        else:
            self.lstm_cell = customLSTMCell(fc5_det_dim, fc4_pat_dim, hidden_dim, bias)

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

        # The linear layer that maps from hidden state space to tag space
        self.hidden2quat1 = nn.Linear(hidden_dim, fc1_quat_size)
        self.hidden2quat2 = nn.Linear(fc1_quat_size, fc2_quat_size)
        self.hidden2quat3 = nn.Linear(fc2_quat_size, 4)

        self.hidden2pos1 = nn.Linear(hidden_dim, fc1_pos_size)
        self.hidden2pos2 = nn.Linear(fc1_pos_size, fc2_pos_size)
        self.hidden2pos3 = nn.Linear(fc2_pos_size, 3)

        self.strong_dropout = nn.Dropout(p=STRONG_DROPOUT_RATE)
        self.weak_dropout = nn.Dropout(p=WEAK_DROPOUT_RATE)


    def forward(self, x, patterns, hidden=None):
        # input is of dimensionalty (time, input_size, ...)

        x = self.weak_dropout(F.relu(self.fc1_det(x)))
        x = self.strong_dropout(F.relu(self.fc2_det(x)))
        x = self.strong_dropout(F.relu(self.fc3_det(x)))
        x = self.strong_dropout(F.relu(self.fc4_det(x)))
        x = self.strong_dropout(F.relu(self.fc5_det(x)))

        if not use_const_pat:
            x_pat = self.weak_dropout(F.relu(self.fc1_pat(patterns.view(T - 1, -1, 12))))
            x_pat = self.strong_dropout(F.relu(self.fc2_pat(x_pat)))
            x_pat = self.strong_dropout(F.relu(self.fc3_pat(x_pat)))
            x_pat = self.strong_dropout(F.relu(self.fc4_pat(x_pat)))
            #x = torch.cat([x, x_pat], dim=2)

        x = torch.unsqueeze(x, dim=0)
        x_pat = torch.unsqueeze(x_pat, dim=0)
        lstm_out = []

        for det, pat in zip(torch.unbind(x, dim=1), torch.unbind(x_pat, dim=1)):
            hidden = self.lstm_cell(det, pat, hidden)
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

        rotated_marker1 = qrot(x_quat, patterns[:, :, 0, :].contiguous()) + x_pos
        rotated_marker2 = qrot(x_quat, patterns[:, :, 1, :].contiguous()) + x_pos
        rotated_marker3 = qrot(x_quat, patterns[:, :, 2, :].contiguous()) + x_pos
        rotated_marker4 = qrot(x_quat, patterns[:, :, 3, :].contiguous()) + x_pos
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

        lstm_out, _ = self.lstm(x)

        x_quat = self.weak_dropout(F.relu(self.hidden2quat1(lstm_out)))
        x_quat = self.weak_dropout(F.relu(self.hidden2quat2(x_quat)))
        x_quat = self.hidden2quat3(x_quat)

        x_pos = self.weak_dropout(F.relu(self.hidden2pos1(lstm_out)))
        x_pos = self.weak_dropout(F.relu(self.hidden2pos2(x_pos)))
        x_pos = self.hidden2pos3(x_pos)

        quat_norm = torch.sqrt(torch.sum(torch.pow(x_quat, 2, ), dim=2))
        x_quat = x_quat / torch.unsqueeze(quat_norm, dim=2)

        #print('Inside FOrward():')
        #print(x_quat.shape)
        #print(marker1.shape)
        rotated_marker1 = qrot(x_quat, marker1) + x_pos
        rotated_marker2 = qrot(x_quat, marker2) + x_pos
        rotated_marker3 = qrot(x_quat, marker3) + x_pos
        rotated_marker4 = qrot(x_quat, marker4) + x_pos
        rotated_pattern = torch.cat([rotated_marker1,
                                     rotated_marker2,
                                     rotated_marker3,
                                     rotated_marker4], dim=2)

        return x_quat, x_pos, rotated_pattern


#model = customLSTM(hidden_dim, bias=True)
model = LSTMTracker(hidden_dim)
if use_colab and torch.cuda.is_available():
    print('USING CUDA DEVICE')
    model.cuda()
    device = torch.device('cuda')
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

#def loss_function_pose(pred_quat, pred_delta_pos, pos_truth, marker_truth):


loss_function_pos = nn.MSELoss()
# TODO: respect antipodal pair as well!
loss_function_quat = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
lr_scheduler_params = {'mode': 'min', 'factor': 0.5, 'patience': 3, 'min_lr': 1e-06, 'cooldown': 4}
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=lr_scheduler_params['mode'],
                                                       factor=lr_scheduler_params['factor'],
                                                       patience=lr_scheduler_params['patience'],
                                                       cooldown= lr_scheduler_params['cooldown'],
                                                       verbose=True, min_lr=lr_scheduler_params['min_lr'])

hyper_params = HyperParams(N_train, N_test, T, BATCH_SIZE, optimizer, LEARNING_RATE, scheduler, lr_scheduler_params,
                           STRONG_DROPOUT_RATE, 'NONE', 'l2 on pos + 5* l1 on quat', '')
logger = TrainingLogger(MODEL_NAME, TASK, hyper_params)
name = logger.folder_name + '/model_best.npy'


def train(data):
    for gci in range(10):
        gc.collect()

    for epoch in range(1, NUM_EPOCHS + 1):
        gc.collect()
        model.train()

        delta_detection_batches = torch.split(data.X_train_shuffled, BATCH_SIZE, 1)
        quat_truth_batches = torch.split(data.quat_train, BATCH_SIZE, 1)
        pos_truth_batches = torch.split(data.pos_train, BATCH_SIZE, 1)
        delta_pos_truth_batches = torch.split(data.delta_pos_train, BATCH_SIZE, 1)
        detections_truth_batches = torch.split(data.X_train, BATCH_SIZE, 1)
        pattern_batches = torch.split(data.pattern_train, BATCH_SIZE, 1)
        avg_loss_pose = 0
        avg_loss_quat = 0
        avg_loss_pos = 0
        n_batches_per_epoch = len(delta_detection_batches)

        for k, [delta_dets, quat_truth, pos_truth, marker_truth, delta_pos_truth, pattern_batch] in enumerate(
                zip(delta_detection_batches, quat_truth_batches, pos_truth_batches, detections_truth_batches,
                    delta_pos_truth_batches, pattern_batches)):
            model.zero_grad()

            #print('Printing tensor shapes:')
            #print(delta_dets.shape)
            #print(pattern_batch.shape)

            pred_quat, pred_delta_pos, pred_delta_markers = model(delta_dets[:-1, :, :], pattern_batch[:-1, :, :, :])

            pred_markers = pos_truth[:-1, :, :].repeat(1, 1, 4) + pred_delta_markers
            loss_pose = loss_function_pos(pred_markers, marker_truth[1:, :, :])
            loss_quat = loss_function_quat(pred_quat, quat_truth[1:, :, :])
            loss_pos = loss_function_pos(pred_delta_pos, delta_pos_truth[1:, :, :])

            loss = loss_pos + loss_quat
            loss.backward()
            optimizer.step()
            avg_loss_pose += loss_pose
            avg_loss_quat += loss_quat
            avg_loss_pos += loss_pos

        avg_loss_pose /= n_batches_per_epoch
        avg_loss_quat /= n_batches_per_epoch
        avg_loss_pos /= n_batches_per_epoch

        model.eval()
        with torch.no_grad():
            pred_quat, pred_delta_pos, pred_delta_markers = model(data.X_test_shuffled[:-1, :, :], data.pattern_test[:-1, :, :, :])
            pred_markers = data.pos_test[:-1, :, :].repeat(1, 1, 4) + pred_delta_markers
            loss_pose = loss_function_pos(pred_markers, data.X_test[1:, :, :])
            loss_quat = loss_function_quat(pred_quat, data.quat_test[1:, :, :])
            loss_pos = loss_function_pos(pred_delta_pos, data.delta_pos_test[1:, :, :])
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


def eval(name, data):
    #data = TrainingData()
    #data.load_data(generated_data_dir, N_train, N_test, 'all')
    #data.convert_to_torch()
    model = torch.load(name, map_location=lambda storage, loc: storage)
    model.eval()

    with torch.no_grad():
        quat_preds, pred_delta_pos, _ = model(data.X_test_shuffled[:-1, :, :],
                                                              data.pattern_test[:-1, :, :, :])
        # TODO: 1: oder :-1??
        pos_preds = data.pos_test[:-1, :, :] + pred_delta_pos

        for n in range(10000):
            if np.amax(np.reshape(pos_preds[:, n, :].numpy(), -1)) < 2:
                continue
            visualize_tracking(pos_preds[:, n, :].detach().numpy(),
                               quat_preds[:, n, :].detach().numpy(),
                               data.pos_test[1:, n, :].detach().numpy(),
                               data.quat_test[1:, n, :].detach().numpy(),
                               data.X_test_shuffled[:-1, n, :].numpy(),
                               data.pattern_test[0, n, :].numpy())


#######################################################################################################################

#######################################################################################################################

#######################################################################################################################

if use_colab:
    data = TrainingData()
    #if not generate_data:
    #    train_data, test_data, N_train, N_test = gen_data(N_train, N_test)
    #else:
    #    (train_data, test_data) = gen_data(N_train, N_test)
    #data.set_train_data(train_data)
    #data.set_test_data(test_data)
    #data.save_data(generated_data_dir, 'all')
    data.load_data(generated_data_dir, N_train, N_test, 'all')
    data.convert_to_torch()

else:
    data = TrainingData()
    #if not generate_data:
    #    (train_data, test_data), N_train, N_test = gen_data(N_train, N_test)
    #else:
    #    (train_data, test_data) = gen_data(N_train, N_test)
    #data.set_train_data(train_data)
    #data.set_test_data(test_data)
    #data.save_data(generated_data_dir, 'all')
    data.load_data(generated_data_dir, N_train, N_test, 'all')
    data.convert_to_torch()

gc.collect()
#train(data)
gc.collect()
if not use_colab:
    #eval(name, data)
    eval('LSTM_BIRDS_relative/model_best.npy', data)