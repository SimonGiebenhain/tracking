import numpy as np
import pandas as pd
from vizTracking import visualize_tracking
from pyquaternion import Quaternion as Quaternion


df = pd.read_csv('data/pigeon.csv')
pigeon = df.to_numpy()
pigeon = pigeon[:, -7:]
pos = pigeon[:, -3:]
pos_df = pd.DataFrame(pos).rolling(10).median()
pos = pos_df.to_numpy()
quat = pigeon[:, :4]
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
    long_sections_pos = [snip - np.mean(snip, axis=0)  for snip in clean_sections_pos if len(snip) >= length]
    long_sections_quat = [snip  for snip in clean_sections_quat if len(snip) >= length]
    clean_snippets_pos = []
    clean_snippets_quat = []
    for sec_pos, sec_quat in zip(long_sections_pos, long_sections_quat):
        l = len(sec_pos)
        n_snips = int(l/length)
        sec_pos = sec_pos[:n_snips*length, :]
        clean_snippets_pos += np.split(sec_pos, n_snips, axis=0)
        sec_quat = sec_quat[:n_snips*length, :]
        clean_snippets_quat += np.split(sec_quat, n_snips, axis=0)
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
snippets_pos, snippets_quat = gen_clean_snippets(pos, quat, split_points, 50)
print(len(snippets_pos))

for i in range(10):
    dets = gen_detections(snippets_pos[i], snippets_quat[i], pattern)

    #detections, pos_snip, quat_snip = gen_detections(pos, quat, pattern, 100)
    visualize_tracking(snippets_pos[i], snippets_quat[i], snippets_pos[i], snippets_quat[i], dets, pattern)


# TODO: center and scale snippets
# TODO: think about how to augment data, aka. stitch together snippets



