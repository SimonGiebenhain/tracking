import numpy as np
import pickle

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

from scipy.spatial.transform import Rotation as R


class BirdData:
    def __init__(self, kalmanPos, kalmanQuat, viconPos, viconQuat, detections, poseLine, viconPosLine, scatter, kalman_preds, vicon_preds):
        self.kalmanPos = kalmanPos
        self.kalmanQuat = kalmanQuat
        self.viconPos = viconPos
        self.viconQuat = viconQuat
        self.detections = detections
        self.posLine = poseLine
        self.viconPosLine = viconPosLine
        self.scatter = scatter
        self.kalman_preds = kalman_preds
        self.vicon_preds = vicon_preds



def update_lines(t, dataLines, birdId, trajectory_length):
    #TODO test whether trajLength=0 works

    #meanPos = dataLines[0][num,:]
    #for line, data in zip(lines, dataLines):
    #    line.set_data((data[:num+1, 0:2]-meanPos[0:2]).T)
    #    line.set_3d_properties(data[:num+1, 2]-meanPos[2])
    #return lines
    lines = []
    center = dataLines[birdId].kalmanPos[t, :]
    for bird in dataLines:
        if trajectory_length > 0:
            if t > trajectory_length:
                t0 = t - trajectory_length - 1
            else:
                t0 = 0
        elif trajectory_length == -1:
            t0 = 0
        else:
            raise ValueError("""The value for trajectory_length is illegal. Use a positive integer or -1 for unlimited trajectories. \n 
                                The following value for trajectory_length was supplied: {}""".format(trajectory_length))
        bird.posLine.set_data((bird.kalmanPos[t0:t + 1, 0:2] - center[:2]).T)
        bird.posLine.set_3d_properties(bird.kalmanPos[t0:t + 1, 2] - center[2])
        bird.viconPosLine.set_data((bird.viconPos[t0:t + 1, 0:2] - center[:2]).T)
        bird.viconPosLine.set_3d_properties(bird.viconPos[t0:t + 1, 2] - center[2])
        lines.append(bird.posLine)
        lines.append(bird.viconPosLine)

        # calculate the expected marker locations
        # first with the kalman predictions
        #TODO pattern = bird.pattern
        pattern = 15 * np.array([[0, 0, 0],
                            [1, 1, 1],
                            [-1, -1, -1],
                            [1, 0, -1]])
        rot = R.from_quat(bird.kalmanQuat[t, :], normalized=False)
        rotated_pattern = rot.apply(pattern)
        expected_markers_kalman = rotated_pattern + bird.kalmanPos[t, :]

        # now with VICON predictions
        rot = R.from_quat(bird.viconQuat[t, :], normalized=True)
        rotated_pattern = rot.apply(pattern)
        expected_markers_vicon = rotated_pattern + bird.viconPos[t,:]

        # display expected marker locations
        bird.kalman_preds._offsets3d = (expected_markers_kalman[:, 0] - center[0],
                                       expected_markers_kalman[:, 1] - center[1],
                                       expected_markers_kalman[:, 2] - center[2])
        bird.vicon_preds._offsets3d = (expected_markers_vicon[:, 0] - center[0],
                                      expected_markers_vicon[:, 1] - center[1],
                                      expected_markers_vicon[:, 2] - center[2])
        lines.append(bird.kalman_preds)
        lines.append(bird.vicon_preds)



    bird = dataLines[0]

    bird.scatter._offsets3d = (bird.detections[t, :, 0] - center[0],
                               bird.detections[t, :, 1] - center[1],
                               bird.detections[t, :, 2] - center[2])
    lines.append(bird.scatter)
    return lines

def visualize(data, isNumpy=False):
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    N = np.shape(data)[1]
    length = np.shape(data)[0]
    data_list = []
    if isNumpy:
        for n in range(N):
            data_list.append(np.squeeze(data[:, n, :]))
    else:
        for n in range(N):
            data_list.append(np.squeeze(data[:, n, :].numpy()))
    lines = [ax.plot(dat[0:1, 0], dat[0:1, 1], dat[0:1, 2])[0] for dat in data_list]

    ax.set_xlim3d([-1, 1])
    ax.set_xlabel('X')

    ax.set_ylim3d([-1, 1])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-1, 1])
    ax.set_zlabel('Z')

    ax.set_title('3D Test')

    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, update_lines, length, fargs=(data_list, lines),
                                       interval=50, blit=False)

    plt.show()

def loadColors():
    path = 'data/'
    with open(path + 'colors.pkl', 'rb') as fin:
        colors = pickle.load(fin)
    np.save('data/colors.npy', colors)

def importAndStoreMATLABData():

    # import kalman filter positions predictions
    path = '../behaviour/'
    with open(path + 'positionsX.pkl', 'rb') as fin:
        posX = pickle.load(fin)
    with open(path + 'positionsY.pkl', 'rb') as fin:
        posY = pickle.load(fin)
    with open(path + 'positionsZ.pkl', 'rb') as fin:
        posZ = pickle.load(fin)

    pos = np.stack([posX[:, :-1], posY[:, :-1], posZ[:, :-1]], axis=2)

    np.save('data/pos.npy', pos)

    # import kalman filter orientation predictions
    with open(path + 'quats1.pkl', 'rb') as fin:
        quats1 = pickle.load(fin)
    with open(path + 'quats2.pkl', 'rb') as fin:
        quats2 = pickle.load(fin)
    with open(path + 'quats3.pkl', 'rb') as fin:
        quats3 = pickle.load(fin)
    with open(path + 'quats4.pkl', 'rb') as fin:
        quats4 = pickle.load(fin)

    quats = np.stack([quats1[:, :-1], quats2[:, :-1], quats3[:, :-1], quats4[:, :-1]], axis=2)

    np.save('data/quats.npy', quats)


    # import VICON predictions
    with open(path + 'vicon.pkl', 'rb') as fin:
        vicon = pickle.load(fin)
    # extract positions and quaternions from unified table
    # make sure to bring quaternions into normal order
    viconPos = np.zeros([10, np.shape(vicon)[0], 3]) * np.NaN
    viconQuats = np.zeros([10, np.shape(vicon)[0], 4]) * np.NaN
    for t in range(np.shape(vicon)[0]):
        for k in range(10):
            viconPos[k, t, :] = vicon[t, k*7+4:(k+1)*7]
            viconQuats[k, t, 0] = vicon[t, 3]
            viconQuats[k, t, 1:] = vicon[t, k*7:k*7+3]
    np.save('data/viconPos.npy', viconPos)
    np.save('data/viconQuats.npy', viconQuats)


    # import unlabeled detections
    with open(path + 'detectionsX.pkl', 'rb') as fin:
        detsX = pickle.load(fin)
    with open(path + 'detectionsY.pkl', 'rb') as fin:
        detsY = pickle.load(fin)
    with open(path + 'detectionsZ.pkl', 'rb') as fin:
        detsZ = pickle.load(fin)

    dets = np.stack([detsX[:, :-1], detsY[:, :-1], detsZ[:, :-1]], axis=2)

    np.save('data/detections.npy', dets)


def old(birdId, firstFrame, lastFrame, trajectory_length=-1):
    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    # load data
    pos = np.load('data/pos.npy')
    pos = pos[:, firstFrame:lastFrame+1, :]
    quats = np.load('data/quats.npy')
    quats = quats[:, firstFrame:lastFrame+1, :]
    viconPos = np.load('data/viconPos.npy')
    viconPos = viconPos[:, firstFrame:lastFrame+1, :]
    viconQuats = np.load('data/viconQuats.npy')
    viconQuats = viconQuats[:, firstFrame:lastFrame+1, :]
    detections = np.load('data/detections.npy')
    detections = detections[firstFrame:lastFrame+1, :, :]

    colors = np.load('data/colors.npy')

    print(np.shape(detections))
    dataAndLines = []
    center = pos[birdId, 0, :]
    for k in range(10):
        kalmanColor = colors[k,:]
        viconColor =(kalmanColor+3)/(np.max(kalmanColor)+3)
        bird = BirdData(pos[k,:,:], quats[k,:,:], viconPos[k,:,:], viconQuats[k,:,:], detections,
                        ax.plot(pos[k, 0:1, 0]-center, pos[k, 0:1, 1]-center, pos[k, 0:1, 2]-center, color=kalmanColor)[0],
                        ax.plot(viconPos[k, 0:1, 0]-center, viconPos[k, 0:1, 1]-center, viconPos[k, 0:1, 2]-center, color=viconColor)[0],
                        ax.scatter([], [], [], marker='x'),
                        ax.scatter([], [], [], marker='o', facecolor=(0, 0, 0, 0), edgecolor=kalmanColor),
                        ax.scatter([], [], [], marker='s', facecolor=(0, 0, 0, 0), edgecolor=viconColor)
                        )
        dataAndLines.append(bird)


    # Creating fifty line objects.
    # NOTE: Can't pass empty arrays into 3d version of plot()
    #lines = [ax.plot(dat[0:1, 0], dat[0:1, 1], dat[0:1, 2])[0] for dat in data]

    # Setting the axes properties
    ax.set_xlim3d([-500, 500])
    ax.set_xlabel('X')

    ax.set_ylim3d([-500, 500])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-500, 500])
    ax.set_zlabel('Z')

    ax.set_title('3D Test')

    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, update_lines, np.shape(quats)[1]-1, fargs=(dataAndLines, birdId, trajectory_length),
                                       interval=2, blit=False)

    plt.show()

birdId = 2
firstFrame = 1300
lastFrame = 2500
trajectory_length = 100

#TODO specify whether to use VICON or correctedVICON
#TODO expected marker locations, with param to turn viz of expected markers off
#TODO implement pause feature
#TODO init azimuth
#TODO save clips
#TODO smoothen camera motion
#TODO write instructions:
# choose bird (from 0 to 9)
# zoom
# rotate
# which vicon to use
# starting frame
# animation speed


# at the end bird 0 has no detections anymore, i.e. it cannot be tracked

old(birdId, firstFrame, lastFrame, trajectory_length)
