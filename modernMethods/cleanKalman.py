import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from viz import visualize
import pickle as pkl


# TODO: apply kalamnFIlter to complete sequence
# TODO: make use of second dataset

path = 'data/'
with open(path + 'all_positionsX.pkl', 'rb') as fin:
    posX = pkl.load(fin)
with open(path + 'all_positionsY.pkl', 'rb') as fin:
    posY = pkl.load(fin)
with open(path + 'all_positionsZ.pkl', 'rb') as fin:
    posZ = pkl.load(fin)

pos = np.stack([posX, posY, posZ], axis=2)
np.save('data/pos_all.npy', pos)
#pos = np.load('data/pos.npy')
colors = np.load('data/colors.npy')
n_objects = pos.shape[0]
T = pos.shape[1]

rolling_media_window_size = 30
rolling_mean_window_size = 10


def plot_trajectories(pos):
    for k in range(n_objects):
        traj = pos[k, :, 2]
        plt.plot(traj, c=colors[k, :])


def get_snippets(pos, length, interval):
    snippets = []
    for k in range(n_objects):
        traj = pos[k, :, :]
        T = traj.shape[0]
        r_bnd = T - length
        for t in range(0, r_bnd, interval):
            tr = traj[t:t+length, :]
            if not np.any(np.isnan(tr)):
                tr_norm = np.linalg.norm(tr, axis=1)
                diff = tr_norm[1:] - tr_norm[:-1]
                if np.any(diff > 50):
                    print('Big Jump encountered!')
                else:
                    snippets.append(tr)
            else:
                print('NaN encountered!')
    return np.stack(snippets, axis=0)


def center_snippets(snips):
    return snips - np.expand_dims(np.nanmean(snips, axis=1), axis=1)


def scale_snippets(snips):
    norms = np.linalg.norm(snips, axis=2)
    norms_std = np.std(np.reshape(norms, -1))
    snips = snips / (10*np.sqrt(norms_std))
    return snips


def visualize_snippet(snip):
    visualize(np.expand_dims(snip, axis=1), isNumpy=True)


def rotate_snippets(snips, number_of_rotations):

    def rotate_snippet(snip):
        theta = np.random.uniform(1, 5)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        rotated_xy = np.matmul(R, snip[:, :2].T).T
        return np.concatenate([rotated_xy, np.expand_dims(snip[:, 2], axis=1)], axis=1)

    rotated_snips = []
    for i in range(len(snips)):
        for r in range(number_of_rotations):
            rotated_snip = rotate_snippet(snips[i, :, :])
            rotated_snips.append(rotated_snip)
    rotated_snips = np.stack(rotated_snips, axis=0)
    return np.concatenate([rotated_snips, snips], axis=0)

def save_as_training_data(snips):
    snips = np.transpose(snips, [1, 0, 2])
    np.save('data/cleaned_kalman_pos_all.npy', snips)


plt.subplot(121)
plot_trajectories(pos)
pos_smoothened = np.zeros([n_objects, T - rolling_media_window_size + 1, 3])
for k in range(n_objects):
    pos_smoothened[k, :, :] = pd.DataFrame(pos[k, :, :]).rolling(rolling_media_window_size).median().to_numpy()[29:, :]

pos_smoothened2 = np.zeros([n_objects,T - rolling_media_window_size + 1 - rolling_mean_window_size + 1, 3])
for k in range(n_objects):
    pos_smoothened2[k, :, :] = pd.DataFrame(pos_smoothened[k, :, :]).rolling(rolling_mean_window_size).mean().to_numpy()[9:, :]
plt.subplot(122)
plot_trajectories(pos_smoothened2)
plt.show()


snippets = get_snippets(pos_smoothened2, 100, 20)
print(len(snippets))
snippets = center_snippets(snippets)
snippets = scale_snippets(snippets)

number_of_rotations = 3
snippets = rotate_snippets(snippets, number_of_rotations)

print(snippets.shape)

save_as_training_data(snippets)

# TODO: could also mirror

#for k in range(500, 1020, 20):
#    visualize_snippet(snippets[k, :, :])