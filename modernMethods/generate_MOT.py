import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd
from pyquaternion import Quaternion as Quaternion
import scipy.io


noise_std = 0
#TODO: noise_FN_model =
#TODO: noise_FP_model =


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


def simulate_markers(pos, quats, patterns):
    T = pos.shape[1]
    N_birds = pos.shape[0]
    detections = np.zeros([N_birds, T, 4, 3])
    for t in range(T):
        for n in range(N_birds):
            p = pos[n, t, :]
            q = quats[n, t, :]
            Q = Quaternion(q)
            R = Q.rotation_matrix
            pat = patterns[n, :, :]

            rotated_pat = (R @ pat.T).T + p

            detections[n, t, :, :] = rotated_pat

    scipy.io.savemat('data/matlab/generated_data.mat', dict(D=detections, pos=pos, quat=quats))
    return detections


pos, t0, t = generate_pos(1000)
print(pos.shape)
quats = generate_quat(t0, t)
print(quats.shape)

#pos = np.load('data/pos_all.npy')
#quats = np.load('data/quats_all.npy')
patterns = np.load('data/patterns.npy')

dets = simulate_markers(pos, quats, patterns)
print(dets.shape)