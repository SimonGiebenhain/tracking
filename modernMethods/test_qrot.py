import numpy as np
import torch

from pyquaternion import Quaternion as Quaternion


def qrot(q, v):
    #TODO check order of quaternion
    #TODO should I make 12 dim output?
    #TODO can I change this function to also work with constant v and changing quaternions?
    # if not just tile/stack v accordingly
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).

    source: https://github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]


    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

def quat2matbad(q):
    """
    TODO

    Original code from: https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
    """

    original_shape = list(q.shape)
    q = q.view(-1, 4)

    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMats = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1)
    rotMats = rotMats.view(original_shape[:-1]+ [9])
    rotMats = rotMats.view(original_shape[:-1]+ [3, 3])
    return rotMats


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    #norm_quat = torch.cat([quat[:,:1].detach()*0 + 1, quat], dim=1)
    #norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3] #norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat

pattern = np.array([[0,0,0], [0,0,0.5], [-0.7,-1,0], [1.1, -1, 0.8]])

quat = np.array([0.25, -0.5, 0, 0.25])
quat = quat/np.sqrt(np.sum(np.square(quat)))

q = Quaternion(quat)

rot_mat = q.rotation_matrix
print(rot_mat)

rotated_pattern = np.dot(rot_mat, pattern.T).T
print(rotated_pattern)

R = quat2matbad(torch.from_numpy(np.expand_dims(quat, axis=0))).numpy()
print(R)
rot_p = np.squeeze(np.dot(R, pattern.T).T)

#sq = np.stack([quat, quat, quat, quat], axis=0)
#print(sq)

#rot_p = qrot(torch.from_numpy(sq), torch.from_numpy(pattern))

print(np.shape(rotated_pattern))
print(np.shape(rot_p))

print(rotated_pattern - rot_p)


stacked_quat = np.stack([quat, quat, quat], axis=0)
stacked_quat = np.stack([stacked_quat, stacked_quat], axis=0)
print('3-D quat, as in [time x batch_size x 4]')
print(np.shape(stacked_quat))
R = quat2matbad(torch.from_numpy(stacked_quat))
print(R)

rotated = torch.matmul(R, torch.from_numpy(pattern.T)).permute([0,1,3,2])
#rotated = np.tensordot(R, pattern, (2,1))
print(rotated)

