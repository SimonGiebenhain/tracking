import numpy as np
import pandas as pd
from vizTracking import visualize_tracking
from pyquaternion import Quaternion as Quaternion
import random

SEQUENCE_LENGTH = 100
rolling_meadian_window_size = 5

df = pd.read_csv('data/pigeon.csv')
pigeon = df.to_numpy()
pigeon = pigeon[:, -7:]
pos = pigeon[:, -3:]
pos_df = pd.DataFrame(pos).rolling(rolling_meadian_window_size).median()
pos = pos_df.to_numpy()
quat = pigeon[:, :4]
quat_df = pd.DataFrame(quat).rolling(rolling_meadian_window_size).median()
quat = quat_df.to_numpy()
quat = np.stack([quat[:, 1], quat[:, 2], quat[:, 3], quat[:, 0]], axis=1)

pattern = np.array([[0, 0, 0],
                    [1, 1, 1],
                    [1, 0, 0],
                    [0, 1, -1]])
def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).+

    source: https://github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    if not q.shape[0] == v.shape[0]:
        v = np.tile(v, [q.shape[0], 1])

    original_shape = v.shape
    q = np.reshape(q, [-1, 4])
    v = np.reshape(v, [-1, 3])

    qvec = q[:, 1:]
    uv = np.cross(qvec, v, axis=1)
    uuv = np.cross(qvec, uv, axis=1)
    return np.reshape((v + 2 * (q[:, :1] * uv + uuv)), original_shape)


def get_plausibility_mask(pos):
    mask = np.ones(len(pos))
    mask[np.any(np.isnan(pos), axis=1)] = 0
    mask = mask[:-1]
    diff = np.linalg.norm(pos[1:, :] - pos[:-1, :], axis=1)
    mask[diff > 5] = 0
    inverted_mask = -mask + 1
    splitpoints = np.nonzero(inverted_mask)[0]
    return mask, splitpoints


def gen_clean_snippets(pos, quat, splitpoints, length):
    clean_sections_pos = np.split(pos, splitpoints, axis=0)
    clean_sections_quat = np.split(quat, splitpoints, axis=0)
    long_sections_pos = [snip[1:, :]  for snip in clean_sections_pos if len(snip) >= length]
    long_sections_quat = [snip[1:, :]  for snip in clean_sections_quat if len(snip) >= length]
    clean_snippets_pos = []
    clean_snippets_quat = []
    for sec_pos, sec_quat in zip(long_sections_pos, long_sections_quat):
        l = len(sec_pos)
        n_snips = int(l/length)
        if n_snips == 0:
            continue
        sec_pos = sec_pos[:n_snips*length, :]
        clean_snippets_pos += np.split(sec_pos, n_snips, axis=0)
        sec_quat = sec_quat[:n_snips*length, :]
        clean_snippets_quat += np.split(sec_quat, n_snips, axis=0)
    for i, snip in enumerate(clean_snippets_pos):
        clean_snippets_pos[i] = snip - np.mean(snip, axis=0)

    return clean_snippets_pos, clean_snippets_quat


def gen_detections(pos, quat, pattern):
    rotated_marker1 = qrot(quat, pattern[0, :])
    rotated_marker2 = qrot(quat, pattern[1, :])
    rotated_marker3 = qrot(quat, pattern[2, :])
    rotated_marker4 = qrot(quat, pattern[3, :])

    detections = np.zeros([pos.shape[0], 4, 3])
    detections[:, 0, :] = rotated_marker1 + pos
    detections[:, 1, :] = rotated_marker2 + pos
    detections[:, 2, :] = rotated_marker3 + pos
    detections[:, 3, :] = rotated_marker4 + pos

    return np.reshape(detections, [-1, 12])


def gen_detections_(pos, quat, pattern, length):
    T = len(pos)
    r_bnd = T - length - 1
    t0 = np.random.randint(0, r_bnd)
    pos_snip = pos[t0:t0 + length, :]
    pos_snip = pos_snip - np.nanmean(pos_snip, axis=0)
    quat_snip = quat[t0:t0 + length, :]
    detections = np.zeros([length, 4, 3])
    for t in range(length):
        q = Quaternion(quat_snip[t, :])
        r = q.rotation_matrix
        rotated_pattern = (r @ pattern.T).T
        detections[t, :, :] = rotated_pattern + np.expand_dims(pos_snip[t, :], axis=0)
    return detections, pos_snip, quat_snip


[mask, split_points] = get_plausibility_mask(pos[:, :])
#print(mask)
#print(split_points)
snippets_pos, snippets_quat = gen_clean_snippets(pos, quat, split_points, SEQUENCE_LENGTH)
print(len(snippets_pos))

for i in range(10, 11):
    p = snippets_pos[i] #+ snippets_pos[i + np.random.randint(1, 10)]
    q = snippets_quat[i] #+ snippets_quat[i + np.random.randint(1, 10)]
    #q_norm = np.linalg.norm(q, axis=1)
    #q = q / np.expand_dims(q_norm, axis=1)

    dets = gen_detections(p, q, pattern)

    #detections, pos_snip, quat_snip = gen_detections(pos, quat, pattern, 100)
    visualize_tracking(p, q, p, q, dets, pattern)


# TODO: center and scale snippets
# TODO: think about how to augment data, aka. stitch together snippets




function_repository = [ lambda x: np.sin(x),
                        lambda x: np.cos(x),
                        lambda x: np.arctan(x),
                        #lambda x: np.power(np.abs(x), 0.1),
                        #lambda x: np.power(np.abs(x), 0.25),
                        lambda x: np.power(np.abs(x), 0.5),
                        #lambda x: np.power(x, 1)/3,
                        lambda x: np.power(x, 2)/(np.max(np.abs(np.power(x,2)))),
                        lambda x: np.power(x, 3)/(np.max(np.abs(np.power(x,3)))),
                        lambda x: np.exp(-1/np.square(x)),
                        lambda x: -np.sin(x),
                        lambda x: -np.cos(x),
                        lambda x: -np.arctan(x),
                        #lambda x: -np.power(np.abs(x), 0.1),
                        #lambda x: -np.power(np.abs(x), 0.25),
                        lambda x: -np.power(np.abs(x), 0.5),
                        #lambda x: -np.power(x, 1) / 3,
                        lambda x: -np.power(x, 2) / (np.max(np.abs(np.power(x, 2)))),
                        lambda x: -np.power(x, 3) / (np.max(np.abs(np.power(x, 3)))),
                        lambda x: -np.exp(-1 / np.square(x))]

import matplotlib.pyplot as plt
for _ in range(10):
    l1 = np.random.uniform(-15, 10)
    r1 = l1 + np.random.uniform(1, 10)
    l2 = np.random.uniform(-15, 5)
    r2 = l2 + np.random.uniform(1, 10)
    l3 = np.random.uniform(-15, 5)
    r3 = l3 + np.random.uniform(1, 10)
    x1 = np.linspace(l1, r1, SEQUENCE_LENGTH)
    x2 = np.linspace(l2, r2, SEQUENCE_LENGTH)
    x3 = np.linspace(l3, r3, SEQUENCE_LENGTH)


    f1 = random.choice(function_repository)
    f2 = random.choice(function_repository)
    f3 = random.choice(function_repository)
    plt.plot(f1(x1) + f2(x2) + f3(x3))

plt.show()

print(np.exp(-1/np.linspace(1,1,50)))

