import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd
from pyquaternion import Quaternion as Quaternion
import scipy.io

from BehaviourModel import NoiseModelFN, NoiseModelFP

FN_transitions = [{'all': [ 0.2, 0.5, 0.3], 'some': [0.1, 0.4, 0.5], 'none': [0.1, 0.3, 0.6]},
                  {'all': [ 0.2, 0.5, 0.3], 'some': [0.1, 0.4, 0.5], 'none': [0.1, 0.3, 0.6]},
                  {'all': [ 0.2, 0.5, 0.3], 'some': [0.1, 0.4, 0.5], 'none': [0.1, 0.3, 0.6]},
                  {'all': [ 0.2, 0.5, 0.3], 'some': [0.1, 0.4, 0.5], 'none': [0.1, 0.3, 0.6]},
                  {'all': [ 0.2, 0.5, 0.3], 'some': [0.1, 0.4, 0.5], 'none': [0.1, 0.3, 0.6]},
                  {'all': [ 0.2, 0.5, 0.3], 'some': [0.1, 0.4, 0.5], 'none': [0.1, 0.3, 0.6]},
                  {'all': [ 0.2, 0.5, 0.3], 'some': [0.1, 0.4, 0.5], 'none': [0.1, 0.3, 0.6]},
                  {'all': [ 0.2, 0.5, 0.3], 'some': [0.1, 0.4, 0.5], 'none': [0.1, 0.3, 0.6]},
                  {'all': [ 0.2, 0.5, 0.3], 'some': [0.1, 0.4, 0.5], 'none': [0.1, 0.3, 0.6]},
                  {'all': [ 0.2, 0.5, 0.3], 'some': [0.1, 0.4, 0.5], 'none': [0.1, 0.3, 0.6]}]

FN_p1s = [[0.1, 0.2], [0.1, 0.2], [0.1, 0.2], [0.1, 0.2], [0.1, 0.2], [0.1, 0.2], [0.1, 0.2], [0.1, 0.2], [0.1, 0.2], [0.1, 0.2]]
FN_p2s = [[0.35, 0.6], [0.35, 0.6], [0.35, 0.6], [0.35, 0.6], [0.35, 0.6], [0.35, 0.6], [0.35, 0.6], [0.35, 0.6], [0.35, 0.6], [0.35, 0.6]]
FN_p3s = [[0.6, 0.85], [0.6, 0.85], [0.6, 0.85], [0.6, 0.85], [0.6, 0.85], [0.6, 0.85], [0.6, 0.85], [0.6, 0.85], [0.6, 0.85], [0.6, 0.85]]
FN_p4s = [[0.8, 0.95], [0.8, 0.95], [0.8, 0.95], [0.8, 0.95], [0.8, 0.95], [0.8, 0.95], [0.8, 0.95], [0.8, 0.95], [0.8, 0.95], [0.8, 0.95]]


FP_transitions = [[[0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.4, 0.1, 0.5]],
                  [[0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.4, 0.1, 0.5]],
                  [[0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.4, 0.1, 0.5]],
                  [[0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.4, 0.1, 0.5]],
                  [[0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.4, 0.1, 0.5]],
                  [[0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.4, 0.1, 0.5]],
                  [[0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.4, 0.1, 0.5]],
                  [[0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.4, 0.1, 0.5]],
                  [[0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.4, 0.1, 0.5]],
                  [[0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.4, 0.1, 0.5]]]

FP_probs = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
FP_scales = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
FP_radiuses = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

#TODO: implement gaussian noise on markers
Noise_scales = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

noise_std = 0

#TODO: loop over all settings, create different .mat-file for every setting

noise_model_states = ['all', 'some', 'none']
noise_model_transition_prob = {'all': [ 0.2, 0.5, 0.3], 'some': [0.1, 0.4, 0.5], 'none': [0.1, 0.3, 0.6]}
noise_model_initial_state = 'all'
p1 = [0.1, 0.2]
p2 = [0.35, 0.6]
p3 = [0.6, 0.85]
p4 = [0.8, 0.95]
noise_FN_model = NoiseModelFN(noise_model_states, noise_model_transition_prob, noise_model_initial_state, p1, p2, p3, p4)


noise_model_FP_states = [0, 1, 2]
noise_model_FP_transition_probs = np.array([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.4, 0.1, 0.5]])
noise_model_FP_initial_probs = np.array([1, 0, 0])
fp_scale = 10
fp_prob = 0.6
radius = 100

noise_FP_model = NoiseModelFP(noise_model_FP_states, noise_model_FP_transition_probs, noise_model_FP_initial_probs,
                              fp_scale, fp_prob, radius)



def plot_trajectories(pos):
    colors = np.load('data/colors.npy')
    for k in range(10):
        traj = pos[k, :, 3]
        plt.plot(traj, c=colors[k, :])


def generate_pos(length):
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
    np.save('data/pos_all.npy', pos)
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
    print(pos.shape)
    bad_frames = np.any(np.isnan(pos[:, :, 0]), axis=0)

    seq_found = False
    t0 = 1500
    t = 1500
    while not seq_found:
        if t >= len(bad_frames):
            break
        if not bad_frames[t]:
            t += 1
            if t - t0 == length:
                break
        else:
            t0 = t+1
            t +=1
    if not t - t0 == length:
        print('Failed to find anything')
        return None
    print(t0)
    print(t)
    return pos[:, t0:t, :], t0, t


def generate_quat(t0, t):
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

    plt.subplot(121)
    plot_trajectories(quats)

    quats_smoothened = np.zeros([n_objects, T - rolling_media_window_size + 1, 4])
    for k in range(n_objects):
        quats_smoothened[k, :, :] = pd.DataFrame(quats[k, :, :]).rolling(rolling_media_window_size).median().to_numpy()[
                                  rolling_media_window_size - 1:,
                                  :]

    quats_smoothened2 = np.zeros([n_objects, T - rolling_media_window_size + 1 - rolling_mean_window_size + 1, 4])
    for k in range(n_objects):
        quats_smoothened2[k, :, :] = pd.DataFrame(quats_smoothened[k, :, :]).rolling(
            rolling_mean_window_size).mean().to_numpy()[rolling_mean_window_size - 1:, :]


    plt.subplot(122)
    plot_trajectories(quats_smoothened2)
    plt.show()


    # normalize quats, as they should be
    quats_norm = np.linalg.norm(quats, axis=2)
    quats = quats / np.expand_dims(quats_norm, axis=2)
    np.save('data/quats_all.npy',quats)

    plot_trajectories(quats)
    plt.show()
    return quats[:, t0:t, :]


def simulate_markers(pos, quats, visibility, patterns, FPs):
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

            detections[n, t, :4, :] = rotated_pat
            non_vis = np.logical_not(visibility[n, t, :])
            non_vis = np.concatenate([non_vis, np.array([np.True_, np.True_])])
            detections[n, t, non_vis , :] = np.NaN
            for i in range(len(FPs[n][t])):
                fp = FPs[n][t][i,:]
                detections[n, t, 4+i, :] = fp + p

    scipy.io.savemat('data/matlab/generated_data.mat', dict(D=detections, pos=pos, quat=quats))
    return detections

T = 1000
marker_visibility = np.zeros([10, T, 4])
for i in range(10):
    marker_visibility[i, :, :] = noise_FN_model.rollout(T)

false_positives = []
for i in range(10):
    noise_FP_model = NoiseModelFP(noise_model_FP_states, noise_model_FP_transition_probs, noise_model_FP_initial_probs,
                                  fp_scale, fp_prob, radius)
    false_positives.append(noise_FP_model.rollout(T))

pos, t0, t = generate_pos(T)
print(pos.shape)
quats = generate_quat(t0, t)
print(quats.shape)

#pos = np.load('data/pos_all.npy')
#quats = np.load('data/quats_all.npy')
patterns = np.load('data/patterns.npy')

dets = simulate_markers(pos, quats, marker_visibility, patterns, false_positives)
print(dets[0, :, :, 0])
print(dets.shape)

print(dets[0, 1:10, 5 , :])