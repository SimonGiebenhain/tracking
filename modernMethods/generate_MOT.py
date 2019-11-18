import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd
from pyquaternion import Quaternion as Quaternion
import scipy.io

from BehaviourModel import NoiseModelFN, NoiseModelFP



def plot_trajectories(pos):
    colors = np.load('data/colors.npy')
    for k in range(10):
        traj = pos[k, :, 3]
        plt.plot(traj, c=colors[k, :])


def generate_pos():
    rolling_media_window_size = 40
    rolling_mean_window_size = 10


    path = 'data/'
    with open(path + 'all_positionsX.pkl', 'rb') as fin:
        posX = pkl.load(fin)
    with open(path + 'all_positionsY.pkl', 'rb') as fin:
        posY = pkl.load(fin)
    with open(path + 'all_positionsZ.pkl', 'rb') as fin:
        posZ = pkl.load(fin)

    pos = np.stack([posX, posY, posZ], axis=2)
    #np.save('data/pos_all.npy', pos)
    n_objects = pos.shape[0]
    T = pos.shape[1]

    pos_smoothened = np.zeros([n_objects, T - rolling_media_window_size + 1, 3])
    for k in range(n_objects):
        pos_smoothened[k, :, :] = pd.DataFrame(pos[k, :, :]).rolling(rolling_media_window_size).median().to_numpy()[rolling_media_window_size-1:,
                                  :]

    pos_smoothened2 = np.zeros([n_objects, T - rolling_media_window_size + 1 - rolling_mean_window_size + 1, 3])
    for k in range(n_objects):
        pos_smoothened2[k, :, :] = pd.DataFrame(pos_smoothened[k, :, :]).rolling(
            rolling_mean_window_size).mean().to_numpy()[rolling_mean_window_size-1:, :]

    pos = pos_smoothened2
    bad_frames = np.any(np.isnan(pos[:, :, 0]), axis=0)
    intervals = []
    cur_interval_left = 0
    cur_interval_length = 0
    for t in range(0, len(bad_frames)):
        if not bad_frames[t]:
            cur_interval_length += 1
            if cur_interval_left == -1:
                cur_interval_left = t
        elif cur_interval_length > 200:
            intervals.append((cur_interval_left, t-1))
            cur_interval_left = -1
            cur_interval_length = 0
        else:
            cur_interval_left = -1
            cur_interval_length = 0
    if cur_interval_length > 200:
        intervals.append( (cur_interval_left, len(bad_frames)-1))
    print(intervals)

    #seq_found = False
    #t0 = 1500
    #t = 1500
    #while not seq_found:
    #    if t >= len(bad_frames):
    #        break
    #    if not bad_frames[t]:
    #        t += 1
    #        if t - t0 == length:
    #            break
    #    else:
    #        t0 = t+1
    #        t +=1
    #if not t - t0 == length:
    #    print('Failed to find anything')
    #    return None
    #print(t0)
    #print(t)
    posIntervals = []
    for interval in intervals:
        posIntervals.append(pos[:, interval[0]:interval[1]+1, :])
    return posIntervals, intervals


def generate_quat(intervals):
    rolling_media_window_size = 20
    rolling_mean_window_size = 10

    path = 'data/'
    with open(path + 'all_quats1.pkl', 'rb') as fin:
        q1 = pkl.load(fin)
    with open(path + 'all_quats2.pkl', 'rb') as fin:
        q2 = pkl.load(fin)
    with open(path + 'all_quats3.pkl', 'rb') as fin:
        q3 = pkl.load(fin)
    with open(path + 'all_quats4.pkl', 'rb') as fin:
        q4 = pkl.load(fin)

    quats = np.stack([q1, q2, q3, q4], axis=2)
    n_objects = quats.shape[0]
    T = quats.shape[1]

    #plt.subplot(121)
    #plot_trajectories(quats)

    quats_smoothened = np.zeros([n_objects, T - rolling_media_window_size + 1, 4])
    for k in range(n_objects):
        quats_smoothened[k, :, :] = pd.DataFrame(quats[k, :, :]).rolling(rolling_media_window_size).median().to_numpy()[
                                  rolling_media_window_size - 1:,
                                  :]

    quats_smoothened2 = np.zeros([n_objects, T - rolling_media_window_size + 1 - rolling_mean_window_size + 1, 4])
    for k in range(n_objects):
        quats_smoothened2[k, :, :] = pd.DataFrame(quats_smoothened[k, :, :]).rolling(
            rolling_mean_window_size).mean().to_numpy()[rolling_mean_window_size - 1:, :]


    #plt.subplot(122)
    #plot_trajectories(quats_smoothened2)
    #plt.show()


    # normalize quats, as they should be
    quats_norm = np.linalg.norm(quats, axis=2)
    quats = quats / np.expand_dims(quats_norm, axis=2)
    #np.save('data/quats_all.npy',quats)

    #plot_trajectories(quats)
    #plt.show()

    quatIntervals = []
    for interval in intervals:
        quatIntervals.append(quats[:, interval[0]:interval[1]+1, :])
    return quatIntervals


def simulate_markers(pos, quats, visibility, patterns, FPs, scale):
    T = pos.shape[1]
    N_birds = pos.shape[0]
    detections = np.zeros([N_birds, T, 6, 3]) * np.NaN
    for t in range(T):
        for n in range(N_birds):
            p = pos[n, t, :]
            q = quats[n, t, :]
            Q = Quaternion(q)
            R = Q.rotation_matrix
            pat = patterns[n, :, :]

            rotated_pat = (R @ pat.T).T + p

            detections[n, t, :4, :] = rotated_pat + np.random.normal(0, scale, [4, 3])
            non_vis = np.logical_not(visibility[n, t, :])
            non_vis = np.concatenate([non_vis, np.array([np.True_, np.True_])])
            detections[n, t, non_vis , :] = np.NaN
            for i in range(len(FPs[n][t])):
                fp = FPs[n][t][i,:]
                detections[n, t, 4+i, :] = fp + p

    #scipy.io.savemat('data/matlab/generated_data.mat', dict(D=detections, pos=pos, quat=quats))
    return detections


FN_transitions = [{'all': [0.95, 0.04, 0.01], 'some': [0.16, 0.83, 0.01], 'none': [0.66, 0.30, 0.04]},
                  {'all': [0.93, 0.06, 0.01], 'some': [0.14, 0.85, 0.01], 'none': [0.61, 0.35, 0.04]},
                  {'all': [0.91, 0.08, 0.01], 'some': [0.12, 0.87, 0.01], 'none': [0.56, 0.40, 0.04]},
                  {'all': [0.89, 0.10, 0.01], 'some': [0.10, 0.89, 0.01], 'none': [0.51, 0.45, 0.04]},
                  {'all': [0.88, 0.10, 0.02], 'some': [0.08, 0.91, 0.01], 'none': [0.48, 0.48, 0.04]},
                  {'all': [0.87, 0.11, 0.02], 'some': [0.08, 0.91, 0.01], 'none': [0.48, 0.48, 0.04]},
                  {'all': [0.87, 0.11, 0.02], 'some': [0.06, 0.92, 0.02], 'none': [0.48, 0.48, 0.04]},
                  {'all': [0.87, 0.11, 0.02], 'some': [0.06, 0.92, 0.02], 'none': [0.48, 0.48, 0.04]},
                  {'all': [0.85, 0.13, 0.02], 'some': [0.06, 0.90, 0.04], 'none': [0.48, 0.48, 0.04]},
                  {'all': [0.85, 0.13, 0.02], 'some': [0.04, 0.92, 0.04], 'none': [0.48, 0.48, 0.04]}]

FN_p1s = [[0.01, 0.05], [0.05, 0.10], [0.10, 0.15], [0.1, 0.2], [0.10, 0.20], [0.10, 0.20], [0.10, 0.20], [0.10, 0.20], [0.15, 0.25], [0.15, 0.25]]
FN_p2s = [[0.05, 0.25], [0.10, 0.25], [0.15, 0.35], [0.2, 0.4], [0.25, 0.50], [0.30, 0.55], [0.35, 0.60], [0.35, 0.60], [0.40, 0.60], [0.45, 0.65]]
FN_p3s = [[0.25, 0.45], [0.35, 0.45], [0.45, 0.60], [0.4, 0.6], [0.50, 0.70], [0.55, 0.75], [0.60, 0.80], [0.60, 0.85], [0.65, 0.80], [0.65, 0.85]]
FN_p4s = [[0.45, 0.75], [0.55, 0.75], [0.60, 0.80], [0.6, 0.8], [0.70, 0.85], [0.75, 0.90], [0.80, 0.90], [0.80, 0.95], [0.80, 0.95], [0.80, 0.95]]


FP_transitions = [[[0.9, 0.1, 0.0], [0.7, 0.2, 0.1], [0.7, 0.2, 0.1]],
                  [[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.7, 0.2, 0.1]],
                  [[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.6, 0.2, 0.2]],
                  [[0.7, 0.2, 0.1], [0.6, 0.2, 0.2], [0.5, 0.2, 0.3]],
                  [[0.7, 0.2, 0.1], [0.5, 0.3, 0.2], [0.4, 0.2, 0.4]],
                  [[0.6, 0.2, 0.2], [0.4, 0.4, 0.2], [0.4, 0.2, 0.4]],
                  [[0.6, 0.2, 0.2], [0.4, 0.4, 0.2], [0.3, 0.2, 0.5]],
                  [[0.6, 0.2, 0.2], [0.3, 0.5, 0.2], [0.3, 0.2, 0.5]],
                  [[0.5, 0.3, 0.2], [0.3, 0.5, 0.2], [0.3, 0.2, 0.5]],
                  [[0.4, 0.4, 0.2], [0.3, 0.5, 0.2], [0.3, 0.2, 0.5]]]

FP_probs = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
FP_scales = [8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13]
FP_radiuses = [125, 120, 115, 110, 105, 100, 95, 90, 85, 80]

noise_scales = [0, 0, 0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2]

T = 20000
patterns = np.load('data/patterns.npy')

for i in range(len(noise_scales)):
    noise_scale = noise_scales[i]

    noise_model_states = ['all', 'some', 'none']
    noise_model_transition_prob = FN_transitions[i]
    noise_model_initial_state = 'all'
    p1 = FN_p1s[i]
    p2 = FN_p2s[i]
    p3 = FN_p3s[i]
    p4 = FN_p4s[i]
    noise_FN_model = NoiseModelFN(noise_model_states, noise_model_transition_prob, noise_model_initial_state, p1, p2, p3, p4)


    noise_model_FP_states = [0, 1, 2]
    noise_model_FP_transition_probs = FP_transitions[i]
    noise_model_FP_initial_probs = np.array([1, 0, 0])
    fp_scale = FP_scales[i]
    fp_prob = FP_probs[i]
    radius = FP_radiuses[i]

    noise_FP_model = NoiseModelFP(noise_model_FP_states, noise_model_FP_transition_probs, noise_model_FP_initial_probs,
                                     fp_scale, fp_prob, radius)

    marker_visibility = np.zeros([10, T, 4])
    for i in range(10):
        marker_visibility[i, :, :] = noise_FN_model.rollout(T)

    false_positives = []
    for i in range(10):
        noise_FP_model = NoiseModelFP(noise_model_FP_states, noise_model_FP_transition_probs, noise_model_FP_initial_probs,
                                      fp_scale, fp_prob, radius)
        false_positives.append(noise_FP_model.rollout(T))

    posIntervals, intervals = generate_pos()
    quatIntervals = generate_quat(intervals)

    marker_vis_intervals = []
    fp_intervals =[]
    for interval in intervals:
        marker_vis_intervals.append(marker_visibility[:, interval[0]:interval[1]+1, :])
        fp_interval = []
        for i in range(10):
            fp_interval.append(false_positives[i][interval[0]:interval[1]+1])
        fp_intervals.append(fp_interval)
    detIntervals = []

    for i in range(len(intervals)):
        detIntervals.append(simulate_markers(posIntervals[i], quatIntervals[i], marker_vis_intervals[i], patterns, fp_intervals[i], noise_scale))

    data_dict = {}

    for i,dets in enumerate(detIntervals):
        data_dict['D' + str(i)] = dets
        data_dict['pos' + str(i)] = posIntervals[i]
        data_dict['quat' + str(i)] = quatIntervals[i]

    scipy.io.savemat('data/matlab/data_difficulty_' + str(i) + '.mat', data_dict)
