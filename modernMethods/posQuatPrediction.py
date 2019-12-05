use_colab = True

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

    from BehaviourModel import NoiseModelFN

drop_some_dets = True
add_false_positives = False
add_noise = False
NOISE_STD = 0.000001
use_const_pat = False
generate_data = False
multi_modal = False

TASK = 'PosQuatPred; '
MODEL_NAME = 'SOTNet'

if multi_modal:
    MODEL_NAME += '_MoG'

if use_colab:
    if use_const_pat:
        generated_data_dir = colab_path_prefix + 'generated_data' + '_const_pat'
    else:
        generated_data_dir = colab_path_prefix + 'generated_data'

else:
    if use_const_pat:
        generated_data_dir = 'generated_data' + '_const_pat'
    else:
        generated_data_dir = 'generated_data'

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

BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.0001
STRONG_DROPOUT_RATE = 0.10
WEAK_DROPOUT_RATE = 0.05
lambda_classification = 1 / 10
lambda_pos = 100
lambda_quat = 1

CHECKPOINT_INTERVAL = 10
save_model_every_interval = False
save_best_model = True

N_train = 600 * BATCH_SIZE
N_test = int(N_train / 10)

T = 100

fc1_det_dim = 200
fc2_det_dim = 250
fc3_det_dim = 300
fc4_det_dim = 200
fc5_det_dim = 100

fc1_pat_dim = 50
fc2_pat_dim = 100
fc3_pat_dim = 200
fc4_pat_dim = 50

hidden_dim = 500

fc1_quat_size = 50
fc2_quat_size = 20

fc1_pos_size = 50
fc2_pos_size = 20

n_mixture_components = 3


# intermediate network which predicts marker identity
# Then feed markers in specific order into LSTM
# But this has to be done inside the LSTM since we need information of the hidden state
# customLSTm could be slow, maybe could do the same with stacked LSTM
# This could also act as the correction step!

# old TODO: incorporate quat and pos in MarkerNet architecture!

# old TODO incorporate dropped dets into markerNet ground truth!!

# old TODO: deepsort? is there such a thing?

# TODO: input absolute, predict relative???

# TODO: use noise models and quats from KF to generate more realistic data

# TODO: pytoch, gen new training data while training, progressively make harder training data, one easy and one hard test set

# TODO predict multiple steps into the future!, i.e. how to fill gaps with no information at all?


# TODO: incorporate long distance predictions and temporal consistency, is that compatible with the multi-modality
#       is it good to only predict delta_x since errors don't accumulate, think about how to fix
#       maybe remove delta again and make snippets shorter

# TODO: loss_pos + loss_quat vs. loss_pose auf den Grund gehen!

# TODO: predict gaussian, then mixture of gaussians


####################################################################################
######### REPORT #########
#
####################################################################################

def normalize_vector(v):
    return v / np.sum(v)


def gen_folder_name(task, name):
    if len(task) > 40:
        short_task = re.sub('[^A-Z]', '', task)
    else:
        short_task = task
    if len(name) > 40:
        short_name = re.sub('[^A-Z]', '', name)
    else:
        short_name = name
    now = datetime.now()
    dt_str = now.strftime("%d.%m.%Y@%H:%M:%S")
    return 'models/' + short_name + '_' + short_task + '_' + dt_str


class TrainingData():
    def __init__(self):

        is_numpy = True

        self.X_train = None
        self.X_train_shuffled = None
        self.quat_train = None
        self.pos_train = None
        self.pattern_train = None
        self.delta_pos_train = None
        self.marker_ids_train = None

        self.X_test = None
        self.X_test_shuffled = None
        self.quat_test = None
        self.pos_test = None
        self.pattern_test = None
        self.delta_pos_test = None
        self.marker_ids_test = None

    def shuffle(self):
        N = self.X_train.shape[1]
        if self.is_numpy:
            randperm = np.random.permutation(np.arange(0, N))
        else:
            randperm = torch.randperm(N)
        self.X_train = self.X_train[:, randperm, :]
        self.X_train_shuffled = self.X_train_shuffled[:, randperm, :]
        self.quat_train = self.quat_train[:, randperm, :]
        self.pos_train = self.pos_train[:, randperm, :]
        print(self.pattern_train.shape)
        self.pattern_train = self.pattern_train[:, randperm, :, :]
        if self.marker_ids_train is not None:
            self.marker_ids_train = self.marker_ids_train[:, randperm, :]
        if self.delta_pos_train is not None:
            self.delta_pos_train = self.delta_pos_train[:, randperm, :]
        if hasattr(self, 'delta_X_train_shuffled') and self.delta_X_train_shuffled is not None:
            self.delta_X_train_shuffled = self.delta_X_train_shuffled[:, randperm, :]
        else:
            print('Haalp')

    def set_data(self, train_data_dict, test_data_dict):
        self.X_train = train_data_dict['X']
        self.X_train_shuffled = train_data_dict['X_shuffled']
        self.quat_train = train_data_dict['quat']
        self.pos_train = train_data_dict['pos']
        self.pattern_train = train_data_dict['pattern']
        if 'marker_ids' in train_data_dict.keys():
            self.marker_ids_train = train_data_dict['marker_ids']

        self.X_test = test_data_dict['X']
        self.X_test_shuffled = test_data_dict['X_shuffled']
        self.quat_test = test_data_dict['quat']
        self.pos_test = test_data_dict['pos']
        self.pattern_test = test_data_dict['pattern']
        if 'marker_ids' in test_data_dict.keys():
            self.marker_ids_test = test_data_dict['marker_ids']

        self.make_relative()

    def make_relative(self):
        self.delta_X_train_shuffled = make_detections_relative(self.X_train_shuffled, self.pos_train)
        self.delta_X_test_shuffled = make_detections_relative(self.X_test_shuffled, self.pos_test)
        self.delta_pos_train = make_pos_relative(self.pos_train)
        self.delta_pos_test = make_pos_relative(self.pos_test)
        # self.pattern_train = self.pattern_train[1:, :, :]
        # self.pattern_test = self.pattern_test[1:, :, :]
        # self.X_train = self.X_train[1:, :, :]
        # self.X_test = self.X_test[1:, :, :]
        # self.quat_train = self.quat_train[1:, :, :]
        # self.quat_test = self.quat_test[1:, :, :]
        # self.pos_train = self.pos_train[1:, :, :]
        # self.pos_test = self.pos_test[1:, :, :]
        # self.marker_ids_train = self.marker_ids_train[1:, :, :]
        # self.marker_ids_test = self.marker_ids_test[1:, :, :]

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
        if os.path.isfile(dname + '/marker_ids_train' + postfix):
            self.marker_ids_train = np.load(dname + '/marker_ids_train' + postfix)

        if generate_data:
            postfix = str(N_test) + '.npy'
        else:
            postfix = '.npy'
        self.X_test = np.load(dname + '/X_test' + postfix)
        self.X_test_shuffled = np.load(dname + '/X_test_shuffled' + postfix)
        self.quat_test = np.load(dname + '/quat_test' + postfix)
        self.pos_test = np.load(dname + '/pos_test' + postfix)
        self.pattern_test = np.load(dname + '/pattern_test' + postfix)
        if os.path.isfile(dname + '/marker_ids_test' + postfix):
            self.marker_ids_test = np.load(dname + '/marker_ids_test' + postfix)

        print('Loaded data successfully!')
        # self.make_relative()
        # print('computed delta values!')

    def save_data(self, dir_name, name):
        dname = dir_name + '_' + name
        print(dname)
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
        if self.marker_ids_train is not None:
            np.save(dname + '/marker_ids_train' + postfix, self.marker_ids_train)

        N = np.shape(self.X_test)[1]
        np.save(dname + '/X_test' + postfix, self.X_test)
        np.save(dname + '/X_test_shuffled' + postfix, self.X_test_shuffled)
        np.save(dname + '/quat_test' + postfix, self.quat_test)
        np.save(dname + '/pos_test' + postfix, self.pos_test)
        np.save(dname + '/pattern_test' + postfix, self.pattern_test)
        if self.marker_ids_test is not None:
            np.save(dname + '/marker_ids_test' + postfix, self.marker_ids_test)

        print('Saved data successfully!')

    def convert_to_torch(self):
        self.is_numpy = False
        if use_colab:
            self.X_train = torch.from_numpy(self.X_train).float().cuda()
            self.X_train_shuffled = torch.from_numpy(self.X_train_shuffled).float().cuda()
            if hasattr(self, 'delta_X_train_shuffled'):
                self.delta_X_train_shuffled = torch.from_numpy(self.delta_X_train_shuffled).float().cuda()
            self.quat_train = torch.from_numpy(self.quat_train).float().cuda()
            self.pos_train = torch.from_numpy(self.pos_train).float().cuda()
            self.pattern_train = torch.from_numpy(self.pattern_train).float().cuda()
            if hasattr(self, 'delta_pos_train') and self.delta_pos_train is not None:
                self.delta_pos_train = torch.from_numpy(self.delta_pos_train).float().cuda()
            if self.marker_ids_train is not None:
                self.marker_ids_train = torch.from_numpy(self.marker_ids_train).type(torch.LongTensor).cuda()

            self.X_test = torch.from_numpy(self.X_test).float().cuda()
            self.X_test_shuffled = torch.from_numpy(self.X_test_shuffled).float().cuda()
            if hasattr(self, 'delta_X_test_shuffled'):
                self.delta_X_test_shuffled = torch.from_numpy(self.delta_X_test_shuffled).float().cuda()
            self.quat_test = torch.from_numpy(self.quat_test).float().cuda()
            self.pos_test = torch.from_numpy(self.pos_test).float().cuda()
            self.pattern_test = torch.from_numpy(self.pattern_test).float().cuda()
            if hasattr(self, 'delta_pos_test') and self.delta_pos_test is not None:
                self.delta_pos_test = torch.from_numpy(self.delta_pos_test).float().cuda()
            if self.marker_ids_test is not None:
                self.marker_ids_test = torch.from_numpy(self.marker_ids_test).type(torch.LongTensor).cuda()

        else:
            self.X_train = torch.from_numpy(self.X_train).float()
            self.X_train_shuffled = torch.from_numpy(self.X_train_shuffled).float()
            if hasattr(self, 'delta_X_train_shuffled'):
                self.delta_X_train_shuffled = torch.from_numpy(self.delta_X_train_shuffled).float()
            self.quat_train = torch.from_numpy(self.quat_train).float()
            self.pos_train = torch.from_numpy(self.pos_train).float()
            self.pattern_train = torch.from_numpy(self.pattern_train).float()
            if hasattr(self, 'delta_pos_train') and self.delta_pos_train is not None:
                self.delta_pos_train = torch.from_numpy(self.delta_pos_train).float()
            if self.marker_ids_train is not None:
                self.marker_ids_train = torch.from_numpy(self.marker_ids_train).type(torch.LongTensor)

            self.X_test = torch.from_numpy(self.X_test).float()
            self.X_test_shuffled = torch.from_numpy(self.X_test_shuffled).float()
            if hasattr(self, 'delta_X_test_shuffled'):
                self.delta_X_test_shuffled = torch.from_numpy(self.delta_X_test_shuffled).float()
            self.quat_test = torch.from_numpy(self.quat_test).float()
            self.pos_test = torch.from_numpy(self.pos_test).float()
            self.pattern_test = torch.from_numpy(self.pattern_test).float()
            if hasattr(self, 'delta_pos_test') and self.delta_pos_test is not None:
                self.delta_pos_test = torch.from_numpy(self.delta_pos_test).float()
            if self.marker_ids_test is not None:
                self.marker_ids_test = torch.from_numpy(self.marker_ids_test).type(torch.LongTensor)

        print('Converted data to torch format.')

    def convert_to_numpy(self):
        self.is_numpy = True
        self.X_train = self.X_train.numpy()
        self.X_train_shuffled = self.X_train_shuffled.numpy()
        self.delta_X_train_shuffled = self.delta_X_train_shuffled.numpy()
        self.quat_train = self.quat_train.numpy()
        self.pos_train = self.pos_train.numpy()
        self.pattern_train = self.pattern_train.numpy()
        self.delta_pos_train = self.delta_pos_train.numpy()
        if self.marker_ids_train is not None:
            self.marker_ids_train = self.marker_ids_train.numpy()

        self.X_test = self.X_test.numpy()
        self.X_test_shuffled = self.X_test_shuffled.numpy()
        self.delta_X_test_shuffled = self.delta_X_test_shuffled.numpy()
        self.quat_test = self.quat_test.numpy()
        self.pos_test = self.pos_test.numpy()
        self.pattern_test = self.pattern_test.numpy()
        self.delta_pos_test = self.delta_pos_test.numpy()
        if self.marker_ids_test is not None:
            self.marker_ids_test = self.marker_ids_test.numpy()

        print('Converted data to numpy format.')

    # normalize X_train and X_test and pos_train, pos_test such that x,y,z cooredinated lie within [-1,1].
    # Also normalize delta versions such that x,y,z coordinated liw within [-1,1]
    def normalize(self):
        detection_scale = np.maximum(np.max(np.abs(self.pos_train), axis=(0, 1)),
                                     np.max(np.abs(self.pos_test), axis=(0, 1)))
        detection_scale = np.expand_dims(np.expand_dims(detection_scale, 0), 0)
        self.X_train = self.X_train / np.tile(detection_scale, [1, 1, 4])
        self.X_test = self.X_test / np.tile(detection_scale, [1, 1, 4])
        self.X_train_shuffled = self.X_train_shuffled / np.tile(
            np.concatenate([detection_scale, np.ones([1, 1, 1])], axis=2), [1, 1, 4])
        self.X_test_shuffled = self.X_test_shuffled / np.tile(
            np.concatenate([detection_scale, np.ones([1, 1, 1])], axis=2), [1, 1, 4])
        self.pattern_train = self.pattern_train / np.expand_dims(detection_scale, 0)
        self.pattern_test = self.pattern_test / np.expand_dims(detection_scale, 0)
        self.pos_train = self.pos_train / detection_scale
        self.pos_test = self.pos_test / detection_scale

        delta_scale = np.maximum(np.max(np.abs(self.delta_pos_train), axis=(0, 1)),
                                 np.max(np.abs(self.delta_pos_test), axis=(0, 1)))
        delta_scale = np.expand_dims(np.expand_dims(delta_scale, 0), 0)
        self.delta_pos_train = self.delta_pos_train / delta_scale
        self.delta_pos_test = self.delta_pos_test / delta_scale
        self.delta_X_train_shuffled = self.delta_X_train_shuffled / np.tile(
            np.concatenate([delta_scale, np.ones([1, 1, 1])], axis=2), [1, 1, 4])
        self.delta_X_test_shuffled = self.delta_X_test_shuffled / np.tile(
            np.concatenate([delta_scale, np.ones([1, 1, 1])], axis=2), [1, 1, 4])


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
        print('Logging model to:')
        print(path)

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
    theta_range = np.random.uniform(0.5, 1)
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


def rotate_quats(quats, theta):
    q = Quaternion(axis=[0, 0, 1], angle=theta)
    rotated_quats = []
    for quat in quats:
        qq = Quaternion(quat)
        rotated_quats.append((qq * q).elements)
    return rotated_quats


def rotate_snippet(snip, theta):
    # theta = np.random.uniform(1, 5)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    rotated_xy = np.matmul(R, snip[:, :2].T).T
    return np.concatenate([rotated_xy, np.expand_dims(snip[:, 2], axis=1)], axis=1)


# TODO: vecotrize with qrot() and by shuffling markers while generating them
def gen_data(N_train, N_test):
    p1 = [0.1, 0.2]
    p2 = [0.2, 0.3]
    p3 = [0.3, 0.4]
    p4 = [0.4, 0.5]
    noise_model_states = ['all', 'some', 'none']
    noise_model_transition_prob = {'all': [0.89, 0.1, 0.01], 'some': [0.02, 0.97, 0.01], 'none': [0.48, 0.48, 0.04]}
    noise_model_initial_state = 'all'

    nM = NoiseModelFN(noise_model_states, noise_model_transition_prob, noise_model_initial_state, p1, p2, p3, p4)

    if not generate_data:
        ratio = N_train / (N_train + N_test)
        pos = np.load('data/cleaned_kalman_pos_all.npy')
        quats = np.load('data/cleaned_kalman_quat_all.npy')
        true_N = pos.shape[1]
        norms = np.linalg.norm(pos, axis=2)
        is_moving = np.any(norms > 1.1, axis=0)
        num_moving = np.count_nonzero(is_moving)
        print('Num moving:')
        print(num_moving)
        num_static = true_N - num_moving
        print('Num static: ')
        print(num_static)
        quat_diff = np.abs(quats[5::5, :, :] - quats[0:-5:5, :, :])
        is_twisting_ = np.mean(quat_diff, axis=(0, 2))
        is_twisting = is_twisting_ > 0.025
        num_twisting = np.count_nonzero(is_twisting)
        print('Num twisting:')
        print(num_twisting)
        # for k in range(len(is_twisting)):
        #    if is_twisting[k]:
        #        plt.plot(quats[:, k, :])
        #        plt.show()

        interesting = np.logical_or(is_twisting, is_moving)
        print('Num interesting:')
        print(np.count_nonzero(interesting))

        pos = pos[:, interesting, :]
        quats = quats[:, interesting, :]

        order = np.arange(0, np.shape(pos)[1])
        perm = np.random.permutation(order)
        pos = pos[:, perm, :]
        quats = quats[:, perm, :]
        true_N = np.shape(pos)[1]

        N_train = ceil(true_N * ratio)
        pos_train = pos[:, :N_train, :]
        quats_train = quats[:, :N_train, :]
        pos_test = pos[:, N_train:, :]
        quats_test = quats[:, N_train:, :]
        N_test = pos_test.shape[1]
        print('N Train')
        print(N_train)
        print('N test')
        print(N_test)

    def gen_datum(N, pos_data=None, quats_data=None):
        if use_colab:
            patterns = np.load(colab_path_prefix + 'patterns.npy')
        else:
            patterns = np.load('data/patterns.npy')
            # patterns = patterns[1, :, :]
            # patterns = np.expand_dims(patterns, axis=0)
        patterns = patterns / 10

        if generate_data:
            pos = gen_pos(N)

            quats = np.zeros([T, N, 4], dtype=np.float32)

            for n in range(N):
                quats[:, n, :] = gen_quats(T)
        else:
            augmentation_factor = 5

            pos = np.zeros(pos_data.shape)
            quats = quats_data
            # q_norm = np.linalg.norm(quats_real, axis=2)
            # print(q_norm)
            # print(np.any(q_norm != 1))
            # quats = np.zeros([100, pos.shape[1], 4])
            # for n in range(pos.shape[1]):
            #    quats[:, n, :] = gen_quats(100)

            # print(quats.shape)
            # for k in range(1, 1000, 7):
            #    plt.subplot(1,2,1)
            #    plt.plot(quats[:, k, 0])
            #    plt.plot(quats[:, k, 1])
            #    plt.plot(quats[:, k, 2])
            #    plt.plot(quats[:, k, 3])
            #    plt.subplot(1,2,2)
            #    plt.plot(quats_real[:, k, 0])
            #    plt.plot(quats_real[:, k, 1])
            #    plt.plot(quats_real[:, k, 2])
            #    plt.plot(quats_real[:, k, 3])
            #    plt.show()

            augmented_pos = []
            augmented_quats = []
            for n in range(pos.shape[1]):
                snip = pos[:, n, :]
                qsnip = quats[:, n, :]
                for k in range(augmentation_factor):
                    scale = np.random.uniform(0.8, 1.1, [1, 3])
                    theta = np.random.uniform(0, 6)
                    snippy = snip * scale
                    snippy = rotate_snippet(snippy, theta)
                    qsnippy = rotate_quats(qsnip, theta)
                    augmented_pos.append(snippy)
                    augmented_quats.append(qsnippy)

            pos = np.stack(augmented_pos, axis=1)
            quats = np.stack(augmented_quats, axis=1)
            num_positions = N
            N = N * len(patterns) * augmentation_factor

        X = np.zeros([T, N, 12])
        print('Gnerating data with size:')
        print(X.shape)
        if add_false_positives:
            X_shuffled = np.zeros([T, N, 20])
        else:
            X_shuffled = np.zeros([T, N, 16])

        all_patterns = np.zeros([T, N, 4, 3])

        marker_identities = np.zeros([T, N, 4])

        for n in range(N):
            gc.collect()
            if drop_some_dets:
                marker_visibility = np.ones([100, 4])
                # marker_visibility = nM.rollout(T) > 0
            for t in range(T):

                p_idx = int(n / (num_positions * augmentation_factor))
                p = patterns[p_idx, :, :]
                p_copy = np.copy(p)
                q = Quaternion(quats[t, n % (num_positions * augmentation_factor), :])
                rnd_perm = np.random.permutation(np.arange(0, 4))
                p_copy = p_copy[rnd_perm, :]
                marker_identities[t, n, :] = rnd_perm[rnd_perm]
                rotated_pattern = (q.rotation_matrix @ p_copy.T).T
                isMissing = np.zeros([4])
                if add_false_positives:
                    raise ValueError('ADD FALSE POSITIVES IS NOT YET SUPPORTED')
                    # rotated_pattern = np.concatenate([rotated_pattern, np.ones([1, 3]) * 0], axis=0)
                    # if np.random.uniform(0, 1) < 0.1:
                    #    if drop_some_dets:
                    #        if np.random.uniform(0, 1) < 0.5:
                    #            if np.random.uniform(0, 1) < 0.5:
                    #                rotated_pattern[3, :] = np.ones([1, 3]) * 0
                    #                rotated_pattern[2, :] = np.random.uniform(-2, 2, [1, 3])
                    #            else:
                    #                rotated_pattern[3, :] = np.random.uniform(-2, 2, [1, 3])
                    #        else:
                    #            rotated_pattern[4, :] = np.random.uniform(-2, 2, [1, 3])
                    #    else:
                    #        rotated_pattern[4, :] = np.random.uniform(-2, 2, [1, 3])
                    # else:
                    #    if drop_some_dets and np.random.uniform(0, 1) < 0.5:
                    #        rotated_pattern[3, :] = np.array([0, 0, 0])
                    #        if drop_some_dets and np.random.uniform(0, 1) < 0.5:
                    #            rotated_pattern[2, :] = np.array([0, 0, 0])
                else:
                    if drop_some_dets:
                        rotated_pattern[rnd_perm[np.logical_not(marker_visibility[t, :])], :] = 0
                        # print(rotated_pattern)
                        marker_identities[t, n, rnd_perm[np.where(marker_visibility[t, :] < 1)]] = 4
                        isMissing[rnd_perm[np.where(marker_visibility[t, :] < 1)]] = 1
                        # print(isMissing)
                    # if drop_some_dets and np.random.uniform(0, 1) < 0.5:
                    #    rotated_pattern[3, :] = np.array([0, 0, 0])
                    #    marker_identities[t, n, rnd_perm[3]] = 4
                    #    isMissing[rnd_perm[3]] = 1
                    #    if drop_some_dets and np.random.uniform(0, 1) < 0.5:
                    #        rotated_pattern[2, :] = np.array([0, 0, 0])
                    #        marker_identities[t, n, rnd_perm[2]] = 4
                    #        isMissing[rnd_perm[2]] = 1
                rotated_pattern_aug = np.zeros([4, 4])
                rotated_pattern_aug[:, :3] = rotated_pattern
                rotated_pattern_aug[:, 3] = isMissing
                dets = np.reshape(rotated_pattern_aug, -1)
                if add_noise:
                    noise = np.random.normal(0, NOISE_STD, np.shape(dets))
                    dets = dets + noise
                X_shuffled[t, n, :] = dets

                rotated_pattern = (q.rotation_matrix @ p.T).T
                X[t, n, :] = np.reshape(rotated_pattern, -1)
                all_patterns[t, n, :, :] = p
        # TODO replace 4 with correct value
        print(X.shape)
        print(X_shuffled.shape)
        X = X + np.tile(pos, [1, len(patterns), 4])
        zero_pos = np.zeros([pos.shape[0], pos.shape[1], 1])
        pos_aug = np.concatenate([pos, zero_pos], axis=2)
        X_shuffled = X_shuffled + np.tile(pos_aug, [1, len(patterns), 4])

        return {'X': X, 'X_shuffled': X_shuffled, 'quat': np.tile(quats, [1, len(patterns), 1]),
                'pos': np.tile(pos, [1, len(patterns), 1]), 'pattern': all_patterns, 'marker_ids': marker_identities}

    if generate_data:
        return (gen_datum(N_train), gen_datum(N_test))
    else:
        train_data = gen_datum(N_train, pos_train, quats_train)
        gc.collect()
        N_train = train_data['X'].shape[1]
        print('Generated the following training data:')
        for key in train_data.keys():
            print(key)
            print(train_data[key].shape)
        test_data = gen_datum(N_test, pos_test, quats_test)
        N_test = test_data['X'].shape[1]
        return (train_data, test_data), N_train, N_test


def gen_data_(N_train, N_test):
    print(N_train)
    print(N_test)
    if not generate_data:
        ratio = N_train / (N_train + N_test)
        pos = np.load('data/cleaned_kalman_pos_all.npy')
        # pos_fast = []
        true_N = pos.shape[1]
        # for n in range(true_N):
        #    p = pos[:, n, :]
        #    if np.max(p) > 2:
        #        pos_fast.append(p)
        # pos = np.stack(pos_fast, 1)
        # true_N = pos.shape[1]
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

        patterns = np.load('data/patterns.npy')
        n_pats = len(patterns)
        pos_stacked = np.tile(pos, [1, n_pats, 4])
        if add_false_positives:
            pos_stacked_fp = np.tile(pos, [1, n_pats, 5])
        else:
            pos_stacked_fp = np.tile(pos, [1, n_pats, 4])

        X = np.zeros([T, N * n_pats, 12])
        if add_false_positives:
            X_shuffled = np.zeros([T, n_pats * N, 15])
        else:
            X_shuffled = np.zeros([T, n_pats * N, 12])
        all_patterns = np.zeros([T, n_pats * N, 4, 3])
        for k in range(n_pats):
            pat = np.expand_dims(np.expand_dims(patterns[k, :, :], axis=0), axis=0)
            pattern = np.tile(pat, [T, N, 1, 1])
            pattern = pattern / 10
            all_patterns[:, k * N:(k + 1) * N, :, :] = pattern

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
                    X_shuffled[t, n * k, :] = dets

                    rotated_pattern = (q.rotation_matrix @ p.T).T
                    X[t, n * k, :] = np.reshape(rotated_pattern, -1)
        X = X  # + pos_stacked
        X_shuffled = X_shuffled  # + pos_stacked_fp

        return {'X': X, 'X_shuffled': X_shuffled, 'quat': np.tile(quat, [1, n_pats, 1]),
                'pos': np.tile(pos, [1, n_pats, 1]), 'pattern': all_patterns}

    if generate_data:
        return (gen_datum(N_train), gen_datum(N_test))
    else:
        data_train = gen_datum(N_train, pos_train)
        data_test = gen_datum(N_test, pos_test)
        return (data_train, data_test), data_train['X'].shape[1], data_test['X'].shape[1]


def make_detections_relative(dets, pos):
    zero_pos_shape = [0, 0, 0]
    zero_pos_shape[0] = pos.shape[0]
    zero_pos_shape[1] = pos.shape[1]
    zero_pos_shape[2] = 1
    zero_pos = np.zeros(zero_pos_shape)
    tiled_pos = np.tile(np.concatenate([pos, zero_pos], axis=2), [1, 1, 4])
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
        # return h_t, (h_t, c_t)

    @staticmethod
    def _init_hidden(input_):
        # h = torch.zeros_like(input_.view(1, input_.size(1), -1))
        # c = torch.zeros_like(input_.view(1, input_.size(1), -1))
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
            # x = torch.cat([x, x_pat], dim=2)

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


class MarkerNetOld(nn.Module):
    def __init__(self):
        super(MarkerNetOld, self).__init__()
        if add_false_positives:
            self.fc1_det = nn.Linear(15, fc1_det_dim)
        else:
            self.fc1_det = nn.Linear(12, fc1_det_dim)

        self.fc2_det = nn.Linear(fc1_det_dim, fc2_det_dim)
        self.fc3_det = nn.Linear(fc2_det_dim, fc3_det_dim)
        self.fc4_det = nn.Linear(fc3_det_dim, fc4_det_dim)

        self.fc1_pat = nn.Linear(12, fc1_pat_dim)
        self.fc2_pat = nn.Linear(fc1_pat_dim, fc2_pat_dim)
        self.fc3_pat = nn.Linear(fc2_pat_dim, fc3_pat_dim)

        self.fc1 = nn.Linear(fc4_det_dim + fc3_pat_dim, 300)
        self.fc2_marker1 = nn.Linear(300, 100)
        self.fc2_marker2 = nn.Linear(300, 100)
        self.fc2_marker3 = nn.Linear(300, 100)
        self.fc2_marker4 = nn.Linear(300, 100)

        # if add_false_positives:
        #    self.fc3 = nn.Linear(100, 5)
        #
        # else:
        self.fc3_marker1 = nn.Linear(100, 4)
        self.fc3_marker2 = nn.Linear(100, 4)
        self.fc3_marker3 = nn.Linear(100, 4)
        self.fc3_marker4 = nn.Linear(100, 4)

        self.weak_dropout = nn.Dropout(p=0.1)

    # TODO concat
    # TODO maybe only tile pattern now
    #   make to 4 dim in las dim
    #   apply softmax
    #   search for alternative
    #   think about loss
    #   train in isolation, then jointly with other model
    def forward(self, detections, pattern):
        x = self.weak_dropout(F.relu(self.fc1_det(detections)))
        x = self.weak_dropout(F.relu(self.fc2_det(x)))
        x = self.weak_dropout(F.relu(self.fc3_det(x)))
        x = self.weak_dropout(F.relu(self.fc4_det(x)))

        x_pat = self.weak_dropout(F.relu(self.fc1_pat(pattern.view(T - 1, -1, 12))))
        x_pat = self.weak_dropout(F.relu(self.fc2_pat(x_pat)))
        x_pat = self.weak_dropout(F.relu(self.fc3_pat(x_pat)))

        x = torch.cat([x, x_pat], dim=2)
        x = self.weak_dropout(F.relu(self.fc1(x)))

        x_marker1 = F.relu(self.fc2_marker1(x))
        x_marker1 = F.relu(self.fc3_marker1(x_marker1))
        # x_marker1 = F.softmax(x_marker1, dim=2)

        x_marker2 = F.relu(self.fc2_marker2(x))
        x_marker2 = F.relu(self.fc3_marker2(x_marker2))
        # x_marker2 = F.softmax(x_marker2, dim=2)

        x_marker3 = F.relu(self.fc2_marker3(x))
        x_marker3 = F.relu(self.fc3_marker3(x_marker3))
        # x_marker3 = F.softmax(x_marker3, dim=2)

        x_marker4 = F.relu(self.fc2_marker4(x))
        x_marker4 = F.relu(self.fc3_marker4(x_marker4))
        # x_marker4 = F.softmax(x_marker4, dim=2)

        return x_marker1, x_marker2, x_marker3, x_marker4


class BirdPoseTracker(nn.Module):
    def __init__(self, hidden_dim):
        super(BirdPoseTracker, self).__init__()
        self.hidden_dim = hidden_dim
        if add_false_positives:
            self.fc1_det = nn.Linear(20, fc1_det_dim)
        else:
            self.fc1_det = nn.Linear(16, fc1_det_dim)
        self.fc2_det = nn.Linear(fc1_det_dim, fc2_det_dim)
        self.fc3_det = nn.Linear(fc2_det_dim, fc3_det_dim)
        # self.fc4_det = nn.Linear(fc3_det_dim, fc4_det_dim)
        # self.fc5_det = nn.Linear(fc4_det_dim, fc5_det_dim)

        if not use_const_pat:
            self.fc1_pat = nn.Linear(12, fc1_pat_dim)
            self.fc2_pat = nn.Linear(fc1_pat_dim, fc2_pat_dim)
            # self.fc3_pat = nn.Linear(fc2_pat_dim, fc3_pat_dim)
            # self.fc4_pat = nn.Linear(fc3_pat_dim, fc4_pat_dim)

        if use_const_pat:
            self.lstm = nn.LSTM(fc3_det_dim, hidden_dim)
        else:
            self.lstm = nn.LSTM(fc3_det_dim + fc2_pat_dim, hidden_dim)

        self.hidden2out1 = nn.Linear(hidden_dim, 500)
        self.hidden2out2 = nn.Linear(500, 200)

        self.hidden2quat1 = nn.Linear(200, 100)
        self.hidden2quat2 = nn.Linear(100, 4)

        self.hidden2pos1 = nn.Linear(200, 50)
        self.hidden2pos2 = nn.Linear(50, 3)

        self.hidden2m11 = nn.Linear(200, 100)
        self.hidden2m12 = nn.Linear(100, 5)
        self.hidden2m21 = nn.Linear(200, 100)
        self.hidden2m22 = nn.Linear(100, 5)
        self.hidden2m31 = nn.Linear(200, 100)
        self.hidden2m32 = nn.Linear(100, 5)
        self.hidden2m41 = nn.Linear(200, 100)
        self.hidden2m42 = nn.Linear(100, 5)

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
        # x = self.strong_dropout(F.relu(self.fc4_det(x)))
        # x = self.strong_dropout(F.relu(self.fc5_det(x)))

        if not use_const_pat:
            x_pat = self.weak_dropout(F.relu(self.fc1_pat(patterns.view(T - 2, -1, 12))))
            x_pat = self.strong_dropout(F.relu(self.fc2_pat(x_pat)))
            # x_pat = self.strong_dropout(F.relu(self.fc3_pat(x_pat)))
            # x_pat = self.strong_dropout(F.relu(self.fc4_pat(x_pat)))
            x = torch.cat([x, x_pat], dim=2)

        x, _ = self.lstm(x)
        x = self.strong_dropout(F.relu(self.hidden2out1(x)))
        x = self.weak_dropout(F.relu(self.hidden2out2(x)))

        m1 = self.weak_dropout(F.relu(self.hidden2m11(x)))
        m1 = self.hidden2m12(m1)
        m2 = self.weak_dropout(F.relu(self.hidden2m21(x)))
        m2 = self.hidden2m22(m2)
        m3 = self.weak_dropout(F.relu(self.hidden2m31(x)))
        m3 = self.hidden2m32(m3)
        m4 = self.weak_dropout(F.relu(self.hidden2m41(x)))
        m4 = self.hidden2m42(m4)

        x_quat = self.weak_dropout(F.relu(self.hidden2quat1(x)))
        x_quat = self.hidden2quat2(x_quat)

        x_pos = self.weak_dropout(F.relu(self.hidden2pos1(x)))
        x_pos = self.hidden2pos2(x_pos)

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

        return x_quat, x_pos, rotated_pattern, m1, m2, m3, m4


class LSTMTracker(nn.Module):
    def __init__(self, hidden_dim):
        super(LSTMTracker, self).__init__()
        self.hidden_dim = hidden_dim
        if add_false_positives:
            self.fc1_det = nn.Linear(15, fc1_det_dim)
        else:
            self.fc1_det = nn.Linear(12, fc1_det_dim)
        # self.fc2_det = nn.Linear(fc1_det_dim, fc2_det_dim)
        # self.fc3_det = nn.Linear(fc2_det_dim, fc3_det_dim)
        # self.fc4_det = nn.Linear(fc3_det_dim, fc4_det_dim)
        # self.fc5_det = nn.Linear(fc4_det_dim, fc5_det_dim)

        if not use_const_pat:
            self.fc1_pat = nn.Linear(12, fc1_pat_dim)
            # self.fc2_pat = nn.Linear(fc1_pat_dim, fc2_pat_dim)
            # self.fc3_pat = nn.Linear(fc2_pat_dim, fc3_pat_dim)
            # self.fc4_pat = nn.Linear(fc3_pat_dim, fc4_pat_dim)

        if use_const_pat:
            self.lstm = nn.LSTM(fc5_det_dim, hidden_dim)
        else:
            self.lstm = nn.LSTM(fc1_det_dim + fc1_pat_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2quat1 = nn.Linear(hidden_dim, fc1_quat_size)
        # self.hidden2quat2 = nn.Linear(fc1_quat_size, fc2_quat_size)
        self.hidden2quat3 = nn.Linear(fc1_quat_size, 4)

        self.hidden2pos1 = nn.Linear(hidden_dim, fc1_pos_size)
        # self.hidden2pos2 = nn.Linear(fc1_pos_size, fc2_pos_size)
        self.hidden2pos3 = nn.Linear(fc1_pos_size, 3)

        self.strong_dropout = nn.Dropout(p=STRONG_DROPOUT_RATE)
        self.weak_dropout = nn.Dropout(p=WEAK_DROPOUT_RATE)

    def forward(self, detections, patterns):
        marker1 = patterns[:, :, 0, :].contiguous()
        marker2 = patterns[:, :, 1, :].contiguous()
        marker3 = patterns[:, :, 2, :].contiguous()
        marker4 = patterns[:, :, 3, :].contiguous()

        x = self.weak_dropout(F.relu(self.fc1_det(detections)))
        # x = self.strong_dropout(F.relu(self.fc2_det(x)))
        # x = self.strong_dropout(F.relu(self.fc3_det(x)))
        # x = self.strong_dropout(F.relu(self.fc4_det(x)))
        # x = self.strong_dropout(F.relu(self.fc5_det(x)))

        if not use_const_pat:
            x_pat = self.weak_dropout(F.relu(self.fc1_pat(patterns.view(T - 3, -1, 12))))
            # x_pat = self.weak_dropout(F.relu(self.fc1_pat(patterns.view(T - 2, -1, 12))))
            # x_pat = self.strong_dropout(F.relu(self.fc2_pat(x_pat)))
            # x_pat = self.strong_dropout(F.relu(self.fc3_pat(x_pat)))
            # x_pat = self.strong_dropout(F.relu(self.fc4_pat(x_pat)))
            x = torch.cat([x, x_pat], dim=2)

        lstm_out, _ = self.lstm(x)

        x_quat = self.weak_dropout(F.relu(self.hidden2quat1(lstm_out)))
        # x_quat = self.weak_dropout(F.relu(self.hidden2quat2(x_quat)))
        x_quat = self.hidden2quat3(x_quat)

        x_pos = self.weak_dropout(F.relu(self.hidden2pos1(lstm_out)))
        # x_pos = self.weak_dropout(F.relu(self.hidden2pos2(x_pos)))
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


class SOTTracker(nn.Module):
    def __init__(self):
        super(SOTTracker, self).__init__()

        if add_false_positives:
            self.fc1_det = nn.Linear(20, 150)
        else:
            self.fc1_det = nn.Linear(12, 150)
        self.bn1 = nn.BatchNorm1d(150)
        self.fc2_det = nn.Linear(150, 200)
        self.bn2 = nn.BatchNorm1d(200)
        self.fc3_det = nn.Linear(200, 250)
        self.bn3 = nn.BatchNorm1d(250)
        self.fc4_det = nn.Linear(250, 250)
        self.bn4 = nn.BatchNorm1d(250)
        # self.fc5_det = nn.Linear(fc4_det_dim, fc5_det_dim)

        if not use_const_pat:
            self.fc1_pat = nn.Linear(12, 50)
            self.fc2_pat = nn.Linear(50, 100)
            self.fc3_pat = nn.Linear(100, 100)
            self.fc4_pat = nn.Linear(100, 100)

            self.bn1_pat = nn.BatchNorm1d(50)
            self.bn2_pat = nn.BatchNorm1d(100)
            self.bn3_pat = nn.BatchNorm1d(100)
            self.bn4_pat = nn.BatchNorm1d(100)

            # self.fc3_pat = nn.Linear(fc2_pat_dim, fc3_pat_dim)
            # self.fc4_pat = nn.Linear(fc3_pat_dim, fc4_pat_dim)

        # if use_const_pat:
        #    self.lstm_prediction = nn.LSTM(fc3_det_dim, hidden_dim)
        # else:
        #    self.lstm_prediction = nn.LSTM(fc3_det_dim + fc2_pat_dim, hidden_dim)

        if use_const_pat:
            self.lstm = nn.LSTM(250, 500)
        else:
            self.lstm = nn.LSTM(250 + 100, 500)
        self.lstm2 = nn.LSTM(500, 350)

        self.hidden2out1_prediction = nn.Linear(350, 200)
        self.hidden2out2_prediction = nn.Linear(200, 128)
        self.hidden2out_bn1 = nn.BatchNorm1d(200)
        self.hidden2out_bn2 = nn.BatchNorm1d(128)

        self.hidden2quat1_prediction = nn.Linear(128, 128)
        self.hidden2quat2_prediction = nn.Linear(128, 128)
        self.hidden2quat3_prediction = nn.Linear(128, 6)
        self.hidden2quat_bn1 = nn.BatchNorm1d(128)
        self.hidden2quat_bn2 = nn.BatchNorm1d(128)

        self.hidden2pos1_prediction = nn.Linear(128, 128)
        self.hidden2pos2_prediction = nn.Linear(128, 3)
        self.hidden2pos_bn1 = nn.BatchNorm1d(128)

        # self.hidden2out1_correction = nn.Linear(hidden_dim, 500)
        # self.hidden2out2_correction = nn.Linear(500, 200)

        # self.hidden2quat1_correction = nn.Linear(200, 100)
        # self.hidden2quat2_correction = nn.Linear(100, 4)

        # self.hidden2pos1_correction = nn.Linear(200, 50)
        # self.hidden2pos2_correction = nn.Linear(50, 3)

        # self.hidden2m11_prediction = nn.Linear(200, 100)
        # self.hidden2m12_prediction = nn.Linear(100, 5)
        # self.hidden2m21_prediction = nn.Linear(200, 100)
        # self.hidden2m22_prediction = nn.Linear(100, 5)
        # self.hidden2m31_prediction = nn.Linear(200, 100)
        # self.hidden2m32_prediction = nn.Linear(100, 5)
        # self.hidden2m41_prediction = nn.Linear(200, 100)
        # self.hidden2m42_prediction = nn.Linear(100, 5)

        # self.strong_dropout = nn.Dropout(p=STRONG_DROPOUT_RATE)
        # self.weak_dropout = nn.Dropout(p=WEAK_DROPOUT_RATE)

    def forward(self, detections, patterns):
        x = F.leaky_relu(self.bn1(self.fc1_det(detections).permute(0, 2, 1)).permute(0, 2, 1))
        x = F.leaky_relu(self.bn2(self.fc2_det(x).permute(0, 2, 1)).permute(0, 2, 1))
        x = F.leaky_relu(self.bn3(self.fc3_det(x).permute(0, 2, 1)).permute(0, 2, 1))
        x = F.leaky_relu(self.bn4(self.fc4_det(x).permute(0, 2, 1)).permute(0, 2, 1))

        if not use_const_pat:
            x_pat = F.leaky_relu(self.bn1_pat(self.fc1_pat(patterns.view(T, -1, 12)).permute(0, 2, 1)).permute(0, 2, 1))
            x_pat = F.leaky_relu(self.bn2_pat(self.fc2_pat(x_pat).permute(0, 2, 1)).permute(0, 2, 1))
            x_pat = F.leaky_relu(self.bn3_pat(self.fc3_pat(x_pat).permute(0, 2, 1)).permute(0, 2, 1))
            x_pat = F.leaky_relu(self.bn4_pat(self.fc4_pat(x_pat).permute(0, 2, 1)).permute(0, 2, 1))
            # x_pat = self.strong_dropout(F.relu(self.fc4_pat(x_pat)))
            x = torch.cat([x, x_pat], dim=2)

        # x_prediction, _ = self.lstm_prediction(x)
        # x_correction, _ = self.lstm_correction(torch.cat([x_prediction[:, :, :], x[:, :, :]], dim=2))
        x, _ = self.lstm(x)
        x, _ = self.lstm2(x)

        x_prediction = F.leaky_relu(
            self.hidden2out_bn1(self.hidden2out1_prediction(x).permute(0, 2, 1)).permute(0, 2, 1))
        x_prediction = F.leaky_relu(
            self.hidden2out_bn2(self.hidden2out2_prediction(x_prediction).permute(0, 2, 1)).permute(0, 2, 1))

        # x_correction = self.strong_dropout(F.relu(self.hidden2out1_correction(x)))
        # x_correction = self.weak_dropout(F.relu(self.hidden2out2_correction(x_correction)))

        # m1 = self.weak_dropout(F.relu(self.hidden2m11(x)))
        # m1 = self.hidden2m12(m1)
        # m2 = self.weak_dropout(F.relu(self.hidden2m21(x)))
        # m2 = self.hidden2m22(m2)
        # m3 = self.weak_dropout(F.relu(self.hidden2m31(x)))
        # m3 = self.hidden2m32(m3)
        # m4 = self.weak_dropout(F.relu(self.hidden2m41(x)))
        # m4 = self.hidden2m42(m4)

        x_quat_prediction = F.leaky_relu(
            self.hidden2quat_bn1(self.hidden2quat1_prediction(x_prediction).permute(0, 2, 1)).permute(0, 2, 1))
        x_quat_prediction = F.leaky_relu(
            self.hidden2quat_bn2(self.hidden2quat2_prediction(x_quat_prediction).permute(0, 2, 1)).permute(0, 2, 1))
        x_quat_prediction = self.hidden2quat3_prediction(x_quat_prediction)

        # x_quat_correction = self.weak_dropout(F.relu(self.hidden2quat1_correction(x_correction)))
        # x_quat_correction = self.hidden2quat2_correction(x_quat_correction)

        x_pos_prediction = F.leaky_relu(
            self.hidden2pos_bn1(self.hidden2pos1_prediction(x_prediction).permute(0, 2, 1)).permute(0, 2, 1))
        x_pos_prediction = self.hidden2pos2_prediction(x_pos_prediction)

        # x_pos_correction = self.weak_dropout(F.relu(self.hidden2pos1_correction(x_correction)))
        # x_pos_correction = self.hidden2pos2_correction(x_pos_correction)

        # quat_norm_correction = torch.sqrt(torch.sum(torch.pow(x_quat_correction, 2, ), dim=2))
        # x_quat_correction = x_quat_correction / torch.unsqueeze(quat_norm_correction, dim=2)

        return x_quat_prediction, x_pos_prediction  # x_quat_correction#, x_pos_correction#rotated_pattern, m1, m2, m3, m4


class PointPatternTracker(nn.Module):
    def __init__(self):
        super(PointPatternTracker, self).__init__()

        if add_false_positives:
            self.fc1 = nn.Linear(20, 150)
        else:
            self.fc1 = nn.Linear(3, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.pool = nn.MaxPool1d(4)
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)

        if use_const_pat:
            self.lstm = nn.LSTM(128, 500)
        else:
            self.lstm = nn.LSTM(128*2, 500)
        self.lstm2 = nn.LSTM(500, 350)

        self.hidden2out1_prediction = nn.Linear(350, 200)
        self.hidden2out2_prediction = nn.Linear(200, 128)
        self.hidden2out_bn1 = nn.BatchNorm1d(200)
        self.hidden2out_bn2 = nn.BatchNorm1d(128)

        self.hidden2quat1_prediction = nn.Linear(128, 128)
        self.hidden2quat2_prediction = nn.Linear(128, 128)
        self.hidden2quat3_prediction = nn.Linear(128, 6)
        self.hidden2quat_bn1 = nn.BatchNorm1d(128)
        self.hidden2quat_bn2 = nn.BatchNorm1d(128)

        self.hidden2pos1_prediction = nn.Linear(128, 128)
        self.hidden2pos2_prediction = nn.Linear(128, 3)
        self.hidden2pos_bn1 = nn.BatchNorm1d(128)

        # self.strong_dropout = nn.Dropout(p=STRONG_DROPOUT_RATE)
        # self.weak_dropout = nn.Dropout(p=WEAK_DROPOUT_RATE)

    def forward(self, detections, patterns):
        x = F.leaky_relu(self.bn1(self.fc1(detections).view(T, -1, 64).permute(0, 2, 1)).permute(0, 2, 1)).view(T, -1, 4, 64)
        x = F.leaky_relu(self.bn2(self.fc2(x).view(T, -1, 128).permute(0, 2, 1)).permute(0, 2, 1)).view(T, -1, 4, 128)
        x = F.leaky_relu(self.bn3(self.fc3(x).view(T, -1, 512).permute(0, 2, 1)).permute(0, 2, 1)).view(T, -1, 4, 512)
        x = self.pool(x.view([T, -1, 4*512]))
        x = F.leaky_relu(self.bn4(self.fc4(x).permute(0, 2, 1)).permute(0,2,1))
        x = F.leaky_relu(self.bn5(self.fc5(x).permute(0, 2, 1)).permute(0,2,1))


        #TODO: actually a fully siamese network will not be optimal, since noise is never present in the pattern!!!
        if not use_const_pat:
            x_pat = F.leaky_relu(self.bn1(self.fc1(patterns).view(T, -1, 64).permute(0, 2, 1)).permute(0, 2, 1)).view(T,
                                                                                                                    -1,
                                                                                                                    4,
                                                                                                                    64)
            x_pat = F.leaky_relu(self.bn2(self.fc2(x_pat).view(T, -1, 128).permute(0, 2, 1)).permute(0, 2, 1)).view(T,
                                                                                                            -1,
                                                                                                            4, 128)
            x_pat = F.leaky_relu(self.bn3(self.fc3(x_pat).view(T, -1, 512).permute(0, 2, 1)).permute(0, 2, 1)).view(T,
                                                                                                            -1,
                                                                                                            4, 512)
            x_pat = self.pool(x_pat.view([T, -1, 4*512]))
            x_pat = F.leaky_relu(self.bn4(self.fc4(x_pat).permute(0, 2, 1)).permute(0, 2, 1))
            x_pat = F.leaky_relu(self.bn5(self.fc5(x_pat).permute(0, 2, 1)).permute(0, 2, 1))
            x = torch.cat([x, x_pat], dim=2)

        # x_prediction, _ = self.lstm_prediction(x)
        # x_correction, _ = self.lstm_correction(torch.cat([x_prediction[:, :, :], x[:, :, :]], dim=2))
        x, _ = self.lstm(x)
        x, _ = self.lstm2(x)

        x_prediction = F.leaky_relu(
            self.hidden2out_bn1(self.hidden2out1_prediction(x).permute(0, 2, 1)).permute(0, 2, 1))
        x_prediction = F.leaky_relu(
            self.hidden2out_bn2(self.hidden2out2_prediction(x_prediction).permute(0, 2, 1)).permute(0, 2, 1))

        # x_correction = self.strong_dropout(F.relu(self.hidden2out1_correction(x)))
        # x_correction = self.weak_dropout(F.relu(self.hidden2out2_correction(x_correction)))

        # m1 = self.weak_dropout(F.relu(self.hidden2m11(x)))
        # m1 = self.hidden2m12(m1)
        # m2 = self.weak_dropout(F.relu(self.hidden2m21(x)))
        # m2 = self.hidden2m22(m2)
        # m3 = self.weak_dropout(F.relu(self.hidden2m31(x)))
        # m3 = self.hidden2m32(m3)
        # m4 = self.weak_dropout(F.relu(self.hidden2m41(x)))
        # m4 = self.hidden2m42(m4)

        x_quat_prediction = F.leaky_relu(
            self.hidden2quat_bn1(self.hidden2quat1_prediction(x_prediction).permute(0, 2, 1)).permute(0, 2, 1))
        x_quat_prediction = F.leaky_relu(
            self.hidden2quat_bn2(self.hidden2quat2_prediction(x_quat_prediction).permute(0, 2, 1)).permute(0, 2, 1))
        x_quat_prediction = self.hidden2quat3_prediction(x_quat_prediction)

        # x_quat_correction = self.weak_dropout(F.relu(self.hidden2quat1_correction(x_correction)))
        # x_quat_correction = self.hidden2quat2_correction(x_quat_correction)

        x_pos_prediction = F.leaky_relu(
            self.hidden2pos_bn1(self.hidden2pos1_prediction(x_prediction).permute(0, 2, 1)).permute(0, 2, 1))
        x_pos_prediction = self.hidden2pos2_prediction(x_pos_prediction)

        # x_pos_correction = self.weak_dropout(F.relu(self.hidden2pos1_correction(x_correction)))
        # x_pos_correction = self.hidden2pos2_correction(x_pos_correction)

        # quat_norm_correction = torch.sqrt(torch.sum(torch.pow(x_quat_correction, 2, ), dim=2))
        # x_quat_correction = x_quat_correction / torch.unsqueeze(quat_norm_correction, dim=2)

        return x_quat_prediction, x_pos_prediction  # x_quat_correction#, x_pos_correction#rotated_pattern, m1, m2, m3, m4



# model = customLSTM(hidden_dim, bias=True)
# model = LSTMTracker(hidden_dim)
# model = MarkerNet()
# model = BirdPoseTracker(hidden_dim)
model = PointPatternTracker()

if use_colab and torch.cuda.is_available():
    print('USING CUDA DEVICE')
    model.cuda()
    device = torch.device('cuda')
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

loss_function_pos = nn.MSELoss()
loss_function_quat = nn.L1Loss()
loss_cross_entropy = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
lr_scheduler_params = {'mode': 'min', 'factor': 0.5, 'patience': 3, 'min_lr': 1e-06, 'cooldown': 4}
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=lr_scheduler_params['mode'],
                                                       factor=lr_scheduler_params['factor'],
                                                       patience=lr_scheduler_params['patience'],
                                                       cooldown=lr_scheduler_params['cooldown'],
                                                       verbose=True, min_lr=lr_scheduler_params['min_lr'])

hyper_params = HyperParams(N_train, N_test, T, BATCH_SIZE, optimizer, LEARNING_RATE, scheduler, lr_scheduler_params,
                           STRONG_DROPOUT_RATE, 'NONE', 'l2 on pos + 5* l1 on quat', '')
logger = TrainingLogger(MODEL_NAME, TASK, hyper_params)
name = logger.folder_name + '/model_best.npy'

def normalize_torch(tensor):
    return tensor / torch.unsqueeze(torch.sqrt(torch.sum(torch.pow(tensor, 2), dim=2)), dim=2)

def pose_loss6D(pos, rot_param, patterns, true_markers):
    marker1 = patterns[:, :, 0, :].contiguous()
    marker2 = patterns[:, :, 1, :].contiguous()
    marker3 = patterns[:, :, 2, :].contiguous()
    marker4 = patterns[:, :, 3, :].contiguous()

    column1 = normalize_torch(rot_param[:, :, :3])
    column2 = rot_param[:, :, 3:] - torch.unsqueeze(torch.sum((column1 * rot_param[:, :, 3:]), dim=2), dim=2)*column1
    column3 = torch.cross(column1, column2, dim=2)
    rot_mat = torch.stack([column1, column2, column3], dim=2)
    rotated_marker1 = torch.squeeze(torch.matmul(rot_mat, torch.unsqueeze(marker1, dim=3))) + pos
    rotated_marker2 = torch.squeeze(torch.matmul(rot_mat, torch.unsqueeze(marker2, dim=3))) + pos
    rotated_marker3 = torch.squeeze(torch.matmul(rot_mat, torch.unsqueeze(marker3, dim=3))) + pos
    rotated_marker4 = torch.squeeze(torch.matmul(rot_mat, torch.unsqueeze(marker4, dim=3))) + pos

    rotated_pattern = torch.cat([rotated_marker1,
                                 rotated_marker2,
                                 rotated_marker3,
                                 rotated_marker4], dim=2)

    return loss_function_pos(rotated_pattern, true_markers)


def pose_loss(pos, quats, patterns, true_markers):
    marker1 = patterns[:, :, 0, :].contiguous()
    marker2 = patterns[:, :, 1, :].contiguous()
    marker3 = patterns[:, :, 2, :].contiguous()
    marker4 = patterns[:, :, 3, :].contiguous()

    quat_norm = torch.sqrt(torch.sum(torch.pow(quats, 2), dim=2))
    quats = quats / torch.unsqueeze(quat_norm, dim=2)
    rotated_marker1 = qrot(quats, marker1) + pos
    rotated_marker2 = qrot(quats, marker2) + pos
    rotated_marker3 = qrot(quats, marker3) + pos
    rotated_marker4 = qrot(quats, marker4) + pos
    rotated_pattern = torch.cat([rotated_marker1,
                                 rotated_marker2,
                                 rotated_marker3,
                                 rotated_marker4], dim=2)

    return loss_function_pos(rotated_pattern, true_markers), loss_function_pos(quat_norm,
                                                                               torch.ones(quat_norm.shape).cuda())


def train_assigner(data):
    data.shuffle()
    for gci in range(10):
        gc.collect()

    for epoch in range(1, NUM_EPOCHS + 1):
        gc.collect()
        model.train()

        delta_detection_batches = torch.split(data.X_train_shuffled[:, :, :], BATCH_SIZE, 1)
        quat_truth_batches = torch.split(data.quat_train[:, :, :], BATCH_SIZE, 1)
        pos_truth_batches = torch.split(data.pos_train[:, :, :], BATCH_SIZE, 1)
        delta_pos_truth_batches = torch.split(data.delta_pos_train[:, :, :], BATCH_SIZE, 1)
        detections_truth_batches = torch.split(data.X_train[:, :, :], BATCH_SIZE, 1)
        pattern_batches = torch.split(data.pattern_train[:, :, :, :], BATCH_SIZE, 1)
        marker_assignment_batches = torch.split(data.marker_ids_train[:, :, :], BATCH_SIZE, 1)
        avg_loss_class = 0
        avg_loss_pose = 0
        avg_loss_quat = 0
        avg_loss_pos = 0
        n_batches_per_epoch = len(delta_detection_batches)
        for k, [delta_dets, quat_truth, pos_truth, marker_truth, delta_pos_truth, pattern_batch,
                marker_ass] in enumerate(
                zip(delta_detection_batches[:-1], quat_truth_batches[:-1], pos_truth_batches[:-1],
                    detections_truth_batches[:-1],
                    delta_pos_truth_batches[:-1], pattern_batches[:-1], marker_assignment_batches[:-1])):
            model.zero_grad()
            gc.collect()

            pred_quat, pred_delta_pos, pred_delta_markers, marker1, marker2, marker3, marker4 = model(
                delta_dets[:-1, :, :], pattern_batch[:-1, :, :, :])
            # loss_class =  loss_cross_entropy(marker1.contiguous().view(-1, 5), marker_ass[0:-1, :, 0].contiguous().view(-1))
            # loss_class += loss_cross_entropy(marker2.contiguous().view(-1, 5), marker_ass[0:-1, :, 1].contiguous().view(-1))
            # loss_class += loss_cross_entropy(marker3.contiguous().view(-1, 5), marker_ass[0:-1, :, 2].contiguous().view(-1))
            # loss_class += loss_cross_entropy(marker4.contiguous().view(-1, 5), marker_ass[0:-1, :, 3].contiguous().view(-1))

            pred_markers = pos_truth[:-1, :, :].repeat(1, 1, 4) + pred_delta_markers
            loss_pose = loss_function_pos(pred_markers, marker_truth[1:, :, :])
            loss_quat = torch.min(loss_function_quat(pred_quat, quat_truth[1:, :, :]),
                                  loss_function_quat(-pred_quat, quat_truth[1:, :, :]))
            loss_pos = loss_function_pos(pred_delta_pos, delta_pos_truth[1:, :, :])

            loss = lambda_pos * loss_pos + lambda_quat * loss_quat  # + lambda_classification * loss_class

            loss.backward()
            optimizer.step()

            avg_loss_pose += loss_pose.item()
            avg_loss_quat += loss_quat.item()
            avg_loss_pos += loss_pos.item()
            # avg_loss_class += loss_class.item()

        avg_loss_pose /= n_batches_per_epoch
        avg_loss_quat /= n_batches_per_epoch
        avg_loss_pos /= n_batches_per_epoch
        # avg_loss_class /= n_batches_per_epoch

        model.eval()
        with torch.no_grad():
            pred_quat, pred_delta_pos, pred_delta_markers, marker1, marker2, marker3, marker4 = model(
                data.X_test_shuffled[:-1, :, :],
                data.pattern_test[:-1, :, :, :])
            # loss_class = loss_cross_entropy(marker1.contiguous().view(-1, 5),
            #                                data.marker_ids_test[0:-1, :, 0].contiguous().view(-1))
            # loss_class += loss_cross_entropy(marker2.contiguous().view(-1, 5),
            #                                 data.marker_ids_test[0:-1:, :, 1].contiguous().view(-1))
            # loss_class += loss_cross_entropy(marker3.contiguous().view(-1, 5),
            #                                 data.marker_ids_test[0:-1, :, 2].contiguous().view(-1))
            # loss_class += loss_cross_entropy(marker4.contiguous().view(-1, 5),
            #                                 data.marker_ids_test[0:-1, :, 3].contiguous().view(-1))

            pred_markers = data.pos_test[:-1, :, :].repeat(1, 1, 4) + pred_delta_markers
            loss_pose = loss_function_pos(pred_markers, data.X_test[1:, :, :])
            loss_quat = torch.min(loss_function_quat(pred_quat, data.quat_test[1:, :, :]),
                                  loss_function_quat(-pred_quat, data.quat_test[1:, :, :]))
            loss_pos = loss_function_pos(pred_delta_pos, data.delta_pos_test[1:, :, :])
            # val_loss = lambda_classification * loss_class.item() + \
            val_loss = lambda_pos * loss_pos.item() + lambda_quat * loss_quat.item()

            for param_group in optimizer.param_groups:
                learning_rate = param_group['lr']
                print("Epoch: {epoch:2d}, Learning Rate: {learning_rate:1.8f} \n TrainClass: {train_class:1.6f}, "
                      "TrainQuat: {train_quat:1.4f}  TrainPos: {train_pos:1.6f} \t "
                      "TestClass: {test_class:1.6f}, TestQuat: {test_quat:1.4f}, TestPos: {test_pos:1.6f}".format(
                    epoch=epoch, learning_rate=learning_rate,
                    train_class=avg_loss_pose, train_quat=avg_loss_quat, train_pos=avg_loss_pos,
                    test_class=loss_pose, test_quat=loss_quat, test_pos=loss_pos))
            scheduler.step(val_loss)
            logger.log_epoch(avg_loss_pose, avg_loss_quat, avg_loss_pos,
                             loss_pose, loss_quat, loss_pos,
                             model,
                             learning_rate)
        if use_colab:
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

    logger.save_log()


def train_sot_tracker(data):
    data.shuffle()
    for gci in range(10):
        gc.collect()

    for epoch in range(1, NUM_EPOCHS + 1):
        gc.collect()
        model.train()

        # delta_detection_batches = torch.split(data.delta_X_train_shuffled[:,:,:], BATCH_SIZE, 1)
        detection_batches = torch.split(data.X_train_shuffled, BATCH_SIZE, 1)
        quat_truth_batches = torch.split(data.quat_train[:, :, :], BATCH_SIZE, 1)
        pos_truth_batches = torch.split(data.pos_train[:, :, :], BATCH_SIZE, 1)
        # delta_pos_truth_batches = torch.split(data.delta_pos_train[:,:,:], BATCH_SIZE, 1)
        detections_truth_batches = torch.split(data.X_train[:, :, :], BATCH_SIZE, 1)
        pattern_batches = torch.split(data.pattern_train[:, :, :, :], BATCH_SIZE, 1)
        # marker_assignment_batches = torch.split(data.marker_ids_train[:,:,:], BATCH_SIZE, 1)
        # avg_loss_class = 0
        avg_loss_pose = 0
        avg_loss_quat = 0
        avg_loss_pos_pred = 0
        avg_loss_quat_corr = 0
        avg_loss_pos_corr = 0
        n_batches_per_epoch = len(detection_batches)
        # for k, [delta_dets, quat_truth, pos_truth, marker_truth, delta_pos_truth, pattern_batch, marker_ass] in enumerate(
        #        zip(delta_detection_batches[:-1], quat_truth_batches[:-1], pos_truth_batches[:-1], detections_truth_batches[:-1],
        #            delta_pos_truth_batches[:-1], pattern_batches[:-1], marker_assignment_batches[:-1])):
        for k, [dets, quat_truth, pos_truth, marker_truth, pattern_batch] in enumerate(
                zip(detections_truth_batches[:-1], quat_truth_batches[:-1], pos_truth_batches[:-1],
                    detections_truth_batches[:-1], pattern_batches[:-1])):
            model.zero_grad()
            gc.collect()

            pred_quats, pred_pos = model(dets.view(T, BATCH_SIZE, 4, 3), pattern_batch)
            loss_pose= pose_loss6D(pred_pos[:-1, :, :], pred_quats[:-1, :, :], pattern_batch[1:, :, :, :],
                                            marker_truth[1:, :, :])
            loss = loss_pose
            # loss_class =  loss_cross_entropy(marker1.contiguous().view(-1, 5), marker_ass[0:-1, :, 0].contiguous().view(-1))
            # loss_class += loss_cross_entropy(marker2.contiguous().view(-1, 5), marker_ass[0:-1, :, 1].contiguous().view(-1))
            # loss_class += loss_cross_entropy(marker3.contiguous().view(-1, 5), marker_ass[0:-1, :, 2].contiguous().view(-1))
            # loss_class += loss_cross_entropy(marker4.contiguous().view(-1, 5), marker_ass[0:-1, :, 3].contiguous().view(-1))

            # pred_markers = pos_truth[:-1, :, :].repeat(1, 1, 4) + pred_delta_markers
            # loss_pose_p = loss_function_pos(pred_markers, marker_truth[1:, :, :])
            # loss_quat = torch.min(loss_function_quat(pred_quats[:-1, :, :], quat_truth[1:, :, :]),
            #                      loss_function_quat(-pred_quats[:-1, :, :], quat_truth[1:, :, :]))
            # loss_pos = loss_function_pos(pred_pos[:-1, :, :], pos_truth[1:, :, :])
            # loss = loss_aux * 0.001 + loss_quat + loss_pos

            # loss_quat_corr = torch.min(loss_function_quat(corr_quat, quat_truth[:, :, :]),
            #                      loss_function_quat(-corr_quat, quat_truth[:, :, :]))
            # loss_pos_corr = loss_function_pos(corr_pos, pos_truth[:, :, :])

            # loss = loss_quat_corr + loss_quat_pred #+ 100*loss_pos_pred
            # loss = lambda_pos * loss_pos_pred + lambda_quat * loss_quat_pred + loss_pos_corr.item() + loss_quat_corr.item() #+ lambda_classification * loss_class

            loss.backward()
            optimizer.step()

            avg_loss_pose += loss_pose.item()
            # avg_loss_quat += loss_quat.item()
            # avg_loss_pos_pred += loss_pos_pred.item()
            # avg_loss_quat_corr += loss_quat_corr.item()
            # avg_loss_pos_corr += loss_pos_corr.item()
            # avg_loss_class += loss_class.item()

        avg_loss_pose /= n_batches_per_epoch
        # avg_loss_quat /= n_batches_per_epoch
        # avg_loss_pos_pred /= n_batches_per_epoch
        # avg_loss_quat_corr /= n_batches_per_epoch
        # avg_loss_pos_corr /= n_batches_per_epoch
        # avg_loss_class /= n_batches_per_epoch

        model.eval()
        with torch.no_grad():
            # pred_quat, pred_delta_pos, pred_delta_markers, marker1, marker2, marker3, marker4 = model(data.X_test_shuffled[:-1, :, :],
            #                                                      data.pattern_test[:-1, :, :, :])
            n = data.X_test.shape[1]
            X_split = torch.split(data.X_test, int(n/10), dim=1)
            pattern_split = torch.split(data.pattern_test, int(n/10), dim=1)
            avg_loss_pose_test = 0
            for (X, pattern) in zip(X_split, pattern_split):
              pred_quats, pred_pos = model(X.view(T, -1, 4, 3), pattern)
              avg_loss_pose_test += pose_loss6D(pred_pos[:-1, :, :], pred_quats[:-1, :, :], data.pattern_test[1:, :, :, :],
                                            data.X_test[1:, :, :]).item()
            avg_loss_pose_test /= 10
            val_loss = avg_loss_pose_test
            for param_group in optimizer.param_groups:
                learning_rate = param_group['lr']
                # print("Epoch: {epoch:2d}, Learning Rate: {learning_rate:1.8f} \n TrainPosCorr: {train_pos_corr:1.6f}, "
                #      "TrainQuatCorr: {train_quat_corr:1.4f}  TrainPosPred: {train_pos_pred:1.6f}  TrainQuatPred: {train_quat_pred:1.6f} \t , "
                #      "TestPosCorr: {test_pos_corr:1.6f} TestQuatCorr: {test_quat_corr:1.4f}  TestPosPred: {test_pos_pred:1.6f}  TestQuatPred: {test_quat_pred:1.6f}".format(
                #    epoch=epoch, learning_rate=learning_rate,
                #    train_pos_corr=avg_loss_pos_corr, train_quat_corr=avg_loss_quat_corr, train_pos_pred=avg_loss_pos_pred, train_quat_pred=avg_loss_quat_pred,
                #    test_pos_corr=loss_pos_corr, test_quat_corr=loss_quat_corr,
                #    test_pos_pred=loss_pos_pred, test_quat_pred=loss_quat_pred))
                print("Epoch: {epoch:2d}, Learning Rate: {learning_rate:1.8f} \n TrainPose: {train_pose:1.6f}\t "
                      "TestPose: {test_pose:1.6f}".format(epoch=epoch, learning_rate=learning_rate,
                                                          train_pose=avg_loss_pose,
                                                          test_pose=avg_loss_pose_test))
            scheduler.step(val_loss)

            logger.log_epoch(avg_loss_pose, -1, -1,
                             avg_loss_pose_test, -1, -1,
                             model,
                             learning_rate)
        if use_colab:
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

    logger.save_log()


def train(data):
    data.shuffle()

    # data.convert_to_numpy()
    # data.X_train_shuffled = make_detections_relative(data.X_train, data.pos_train)
    # data.X_train = data.X_train[1:, :, :]
    # data.pos_train = data.pos_train[1:, :, :]
    # data.delta_pos_train = data.delta_pos_train[1:, :, :]
    # data.pattern_train = data.pattern_train[1:, :, :]
    # data.quat_train = data.quat_train[1:, :, :]
    #
    # data.X_test_shuffled = make_detections_relative(data.X_test, data.pos_test)
    # data.X_test = data.X_test[1:, :, :]
    # data.pos_test = data.pos_test[1:, :, :]
    # data.delta_pos_test = data.delta_pos_test[1:, :, :]
    # data.pattern_test = data.pattern_test[1:, :, :]
    # data.quat_test = data.quat_test[1:, :, :]
    # data.convert_to_torch()

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
                zip(delta_detection_batches[:-1], quat_truth_batches[:-1], pos_truth_batches[:-1],
                    detections_truth_batches[:-1],
                    delta_pos_truth_batches[:-1], pattern_batches[:-1])):
            model.zero_grad()
            gc.collect()

            pred_quat, pred_delta_pos, pred_delta_markers = model(delta_dets[:-1, :, :], pattern_batch[:-1, :, :, :])

            pred_markers = pos_truth[:-1, :, :].repeat(1, 1, 4) + pred_delta_markers
            loss_pose = loss_function_pos(pred_markers, marker_truth[1:, :, :])
            loss_quat = torch.min(loss_function_quat(pred_quat, quat_truth[1:, :, :]),
                                  loss_function_quat(-pred_quat, quat_truth[1:, :, :]))
            loss_pos = loss_function_pos(pred_delta_pos, delta_pos_truth[1:, :, :])

            loss = 100 * loss_pos + loss_quat
            # loss = loss_pose
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
            pred_quat, pred_delta_pos, pred_delta_markers = model(data.X_test_shuffled[:-1, :, :],
                                                                  data.pattern_test[:-1, :, :, :])
            pred_markers = data.pos_test[:-1, :, :].repeat(1, 1, 4) + pred_delta_markers
            loss_pose = loss_function_pos(pred_markers, data.X_test[1:, :, :])
            loss_quat = torch.min(loss_function_quat(pred_quat, data.quat_test[1:, :, :]),
                                  loss_function_quat(-pred_quat, data.quat_test[1:, :, :]))
            loss_pos = loss_function_pos(pred_delta_pos, data.delta_pos_test[1:, :, :])
            val_loss = 100 * loss_pos + loss_quat
            for param_group in optimizer.param_groups:
                learning_rate = param_group['lr']
                print("Epoch: {epoch:2d}, Learning Rate: {learning_rate:1.8f} \n TrainPose: {train_pose:1.6f}, "
                      "TrainQuat: {train_quat:1.4f}  TrainPos: {train_pos:1.6f} \t "
                      "TestPose: {test_pose:1.6f}, TestQuat: {test_quat:1.4f}, TestPos: {test_pos:1.6f}".format(
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


def eval(data, name=None):
    if name is not None:
        model = torch.load(name, map_location=lambda storage, loc: storage)
    model.eval()
    with torch.no_grad():
        quat_preds, pred_pos = model(data.X_test[:, :, :], data.pattern_test[:, :, :, :])

        for n in range(3, 100, 21):
            visualize_tracking(pred_pos[1:, n, :].detach().numpy(),
                               quat_preds[1:, n, :].detach().numpy(),
                               data.pos_test[2:, n, :].detach().numpy(),
                               data.quat_test[2:, n, :].detach().numpy(),
                               data.X_test[1:-1, n, :].numpy(),  # + np.tile(data.pos_test[:-2, n, :].numpy(), [1, 4]),
                               data.pattern_test[0, n, :].numpy())


def eval_(name, data):
    model = torch.load(name, map_location=lambda storage, loc: storage)
    model.eval()
    printed_slow = False
    with torch.no_grad():
        quat_preds, pred_delta_pos, _, m1, m2, m3, m4 = model(data.X_test_shuffled[:-1, :, :],
                                                              data.pattern_test[:-1, :, :, :])
        print(m1[:10, 1, :])
        print(data.marker_ids_test[:10, 1, 0])
        # TODO: 1: oder :-1??
        pos_preds = data.pos_test[:-1, :, :] + pred_delta_pos

        for n in range(10000):
            if np.amax(np.reshape(pos_preds[:, n, :].numpy(), -1)) < 2:
                continue
            print('fast')
            print(pred_delta_pos[:, n, :])

            visualize_tracking(pos_preds[1:, n, :].detach().numpy(),
                               quat_preds[1:, n, :].detach().numpy(),
                               data.pos_test[2:, n, :].detach().numpy(),
                               data.quat_test[2:, n, :].detach().numpy(),
                               data.X_test_shuffled[1:-1, n, :].numpy() + np.tile(data.pos_test[:-2, n, :].numpy(),
                                                                                  [1, 4]),
                               data.pattern_test[0, n, :].numpy())


def show_data(data):
    print(data.X_test_shuffled.shape)
    for n in range(0, 100, 5):
        visualize_tracking(data.pos_test[:, n, :].numpy(),
                           data.quat_test[:, n, :].numpy(),
                           data.pos_test[:, n, :].numpy(),
                           data.quat_test[:, n, :].numpy(),
                           data.X_test[:, n, :3].numpy(),
                           data.pattern_test[0, n, :].numpy())


#######################################################################################################################

#######################################################################################################################

#######################################################################################################################

if use_colab:
    data = TrainingData()
    # if not generate_data:
    #    train_data, test_data, N_train, N_test = gen_data(N_train, N_test)
    # else:
    #    (train_data, test_data) = gen_data(N_train, N_test)
    # data.set_data(train_data, test_data)
    # data.save_data(generated_data_dir, 'all')
    data.load_data(generated_data_dir, N_train, N_test, 'all')
    data.convert_to_torch()

else:
    data = TrainingData()
    # if not generate_data:
    #    (train_data, test_data), N_train, N_test = gen_data(N_train, N_test)
    # else:
    #   (train_data, test_data) = gen_data(N_train, N_test)
    # data.set_data(train_data, test_data)
    # data.save_data(generated_data_dir, 'all')
    # data = TrainingData()
    data.load_data(generated_data_dir, N_train, N_test, 'all')
    # data.normalize()
    data.convert_to_torch()
# show_data(data)

gc.collect()
train_sot_tracker(data)
gc.collect()
if not use_colab:
     eval(data, name)
    #eval(data, 'models/SOTNet_6D/model_best.npy')

# TODO: checke daten, wieso ist grau und orange komplett anders?
# TODO: werden verschiedene patterns benutzt?

# TODO try with pos = 0 (const)
# TODO: fix normalize()!!!

# TODO make relative??