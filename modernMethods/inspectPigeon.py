import numpy as np
import pandas as pd
from vizTracking import visualize_tracking
from pyquaternion import Quaternion as Quaternion


df = pd.read_csv('data/pigeon.csv')
pigeon = df.to_numpy()
pigeon = pigeon[:, -7:]
print(len(pigeon))
pos = pigeon[:, -3:]
quat = pigeon[:, :4]
quat = np.stack([quat[:, 1], quat[:, 2], quat[:, 3], quat[:, 0]], axis=1)

pattern = np.array([[0, 0, 0],
                    [1, 1, 1],
                    [1, 0, 0],
                    [0, 1, -1]])

def get_plausibility_mask(pos):
    mask = np.ones(len(pos))
    mask[np.any(np.isnan(pos), axis=1)] = 0
    mask = mask[:-1]
    diff = np.linalg.norm(pos[1:, :] - pos[:-1, :], axis=1)
    print(diff)
    print(np.nanmean(diff))
    big_jumps = diff > 5
    print(np.shape(big_jumps))
    print(np.count_nonzero(big_jumps))
    mask[diff > 10] = 0
    #TODO use inverse of np.nonzero() to calculate split points
    # TODO: thwor away small sequences
    return mask


def gen_detections(pos, quat, pattern, length):
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

print(get_plausibility_mask(pos[:, :]))

detections, pos_snip, quat_snip = gen_detections(pos, quat, pattern, 100)
#visualize_tracking(pos_snip, quat_snip, pos_snip, quat_snip, np.reshape(detections, [100, 12]), pattern)


# TODO: generate training data from pigeon data
# TODO: generate small snippets, remove snippets with unrealistic jumps and nans
# TODO: center and scale snippets
# TODO: think about how to augment data



