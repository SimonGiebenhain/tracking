import numpy as np
import pickle

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

from pyquaternion import Quaternion as Q

import os, os.path

# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)

patterns = np.load('data/patterns.npy')

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

def run_animation(dataAndLines, length, trajectory_length, hide_expected_marker_locations, save_animation):
    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)


    anim_running = True

    def on_press(event):
        if event.key == " ":
            nonlocal anim_running
            if anim_running:
                anim.event_source.stop()
                anim_running = False
            else:
                anim.event_source.start()
                anim_running = True
        elif event.key in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            global birdId
            birdId = int(event.key)


    def update_lines(t, dataLines, trajectory_length, hide_expected_marker_locations):
        global patterns
        global birdId

        lines = []
        smoothness_constant = 15
        if smoothness_constant <= 0:
            center = dataLines[birdId].kalmanPos[t,:]
        else:
            if t < smoothness_constant:
                center = np.median(dataLines[birdId].kalmanPos[:smoothness_constant, :], axis=0)
            elif t + smoothness_constant + 1 > np.shape(dataLines[birdId].kalmanPos)[0]:
                center = np.median(dataLines[birdId].kalmanPos[-smoothness_constant:, :], axis=0)
            else:
                center = np.median(dataLines[birdId].kalmanPos[t-smoothness_constant:t+smoothness_constant, :], axis=0)

        for id, bird in enumerate(dataLines):
            if trajectory_length > 0:
                if t > trajectory_length:
                    t0 = t - trajectory_length - 1
                else:
                    t0 = 0
            elif trajectory_length == -1:
                t0 = 0
            else:
                raise ValueError("""The value for trajectory_length is illegal. Use a positive integer or -1 for unlimited trajectories. \n 
                                    The following value for trajectory_length was supplied: {}""".format(
                    trajectory_length))
            bird.posLine.set_data((bird.kalmanPos[t0:t + 1, 0:2] - center[:2]).T)
            bird.posLine.set_3d_properties(bird.kalmanPos[t0:t + 1, 2] - center[2])
            bird.viconPosLine.set_data((bird.viconPos[t0:t + 1, 0:2] - center[:2]).T)
            bird.viconPosLine.set_3d_properties(bird.viconPos[t0:t + 1, 2] - center[2])
            lines.append(bird.posLine)
            lines.append(bird.viconPosLine)

            if not hide_expected_marker_locations:
                # calculate the expected marker locations
                # first with the kalman predictions
                # TODO pattern = bird.pattern
                pattern = patterns[id, :, :]
                # qu = bird.kalmanQuat[t,:].tolist()
                # qu = [qu[1], qu[2], qu[3], qu[0]]
                # q = Q(qu)
                q = Q(bird.kalmanQuat[t, :])
                rot_mat = q.rotation_matrix
                # rot = R.from_quat(bird.kalmanQuat[t, :], normalized=False)
                # rotated_pattern = rot.apply(pattern)
                rotated_pattern = np.dot(rot_mat, pattern.T).T
                expected_markers_kalman = rotated_pattern + bird.kalmanPos[t, :]

                # now with VICON predictions
                #qu = bird.viconQuat[t, :].tolist()
                q_vicon = Q(bird.viconQuat[t, :])
                if id == 3:
                    print(bird.viconQuat[t, :])
                # qu = [qu[1], qu[2], qu[3], qu[0]]
                # TODO fix bug in rotation?!
                # q = Q(qu)
                rot_mat_vicon = q_vicon.rotation_matrix
                rotated_pattern_vicon = np.dot(rot_mat_vicon, pattern.T).T
                # rot = R.from_quat(bird.viconQuat[t, :], normalized=True)
                # rotated_pattern = rot.apply(pattern)
                expected_markers_vicon = rotated_pattern_vicon + bird.viconPos[t, :]

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

    fig.canvas.mpl_connect('key_press_event', on_press)

    anim = animation.FuncAnimation(fig, update_lines, length,
                                       fargs=(dataAndLines, trajectory_length, hide_expected_marker_locations),
                                       interval=20, blit=False, repeat=True)

    if show_legend:
        ax.legend()
    if save_animation:
        anim.save('lines.mp4', writer=writer)
    plt.show()



def loadColors():
    path = 'data/'
    with open(path + 'colors.pkl', 'rb') as fin:
        colors = pickle.load(fin)
    np.save('data/colors.npy', colors)

def load_pattern():
    patterns = []
    path = '../behaviour/'
    for f in sorted(os.listdir(path)):
        ext = os.path.splitext(f)[0]
        print(ext)
        if not '.vsk' in ext:
            continue
        with open(os.path.join(path, f), 'rb') as fin:
            patterns.append(pickle.load(fin))

    np.save('data/patterns.npy', np.stack(patterns, axis=0))

def load_corrected_vicon():
    import pandas as pd
    corrected_vicon_df = pd.read_csv('../../correctedVICON.csv')
    corrected_vicon = corrected_vicon_df.to_numpy()
    corrected_vicon = corrected_vicon[1:,1:-1]
    viconPos = np.zeros([10, np.shape(corrected_vicon)[0], 3]) * np.NaN
    viconQuats = np.zeros([10, np.shape(corrected_vicon)[0], 4]) * np.NaN
    for t in range(np.shape(corrected_vicon)[0]):
        for k in range(10):
            viconPos[k, t, :] = corrected_vicon[t, k * 7 + 4:(k + 1) * 7]
            viconQuats[k, t, 0] = corrected_vicon[t, k * 7 + 3]
            viconQuats[k, t, 1:] = corrected_vicon[t, k * 7:k * 7 + 3]
    print(np.shape(viconPos))
    print(np.shape(viconQuats))
    np.save('data/corrected_vicon_pos.npy', viconPos)
    np.save('data/corrected_vicon_quats.npy', viconQuats)

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
            viconQuats[k, t, 0] = vicon[t, k*7+3]
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


def old(firstFrame, lastFrame, trajectory_length=-1, hide_expected_marker_locations=False,
        use_corrected_vicon=True, show_legend=False, save_animation=False):

    global birdId
    # load data
    pos = np.load('data/pos.npy')
    pos = pos[:, firstFrame:lastFrame+1, :]
    quats = np.load('data/quats.npy')
    quats = quats[:, firstFrame:lastFrame+1, :]
    if use_corrected_vicon:
        viconPos = np.load('data/corrected_vicon_pos.npy')
        viconPos = viconPos[:, firstFrame:lastFrame + 1, :]
        viconQuats = np.load('data/corrected_vicon_quats.npy')
        viconQuats = viconQuats[:, firstFrame:lastFrame + 1, :]
    else:
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

        viconColor =(kalmanColor+1.5)/(np.max(kalmanColor)+1.5)
        viconColor = viconColor * 0.75
        if k == 0:
            bird = BirdData(pos[k,:,:], quats[k,:,:], viconPos[k,:,:], viconQuats[k,:,:], detections,
                            ax.plot(pos[k, 0:1, 0]-center, pos[k, 0:1, 1]-center, pos[k, 0:1, 2]-center, color=kalmanColor,
                                    linewidth=1, label='bird {}'.format(k))[0],
                            ax.plot(viconPos[k, 0:1, 0]-center, viconPos[k, 0:1, 1]-center, viconPos[k, 0:1, 2]-center, color=viconColor,
                                    linewidth=1)[0],
                            ax.scatter([], [], [], alpha=1, s=40, marker='o', label='unlabeled detections'),
                            ax.scatter([], [], [], alpha=1,  s=20, marker='x', color=kalmanColor, linewidth=1),
                            ax.scatter([], [], [], alpha=1, s=40, marker='+', color=viconColor, linewidth=1.5)
                            )
        else:
            bird = BirdData(pos[k, :, :], quats[k, :, :], viconPos[k, :, :], viconQuats[k, :, :], detections,
                            ax.plot(pos[k, 0:1, 0] - center, pos[k, 0:1, 1] - center, pos[k, 0:1, 2] - center,
                                    color=kalmanColor,
                                    linewidth=1, label='bird {}'.format(k))[0],
                            ax.plot(viconPos[k, 0:1, 0] - center, viconPos[k, 0:1, 1] - center,
                                    viconPos[k, 0:1, 2] - center, color=viconColor,
                                    linewidth=1)[0],
                            ax.scatter([], [], [], alpha=1, s=40, marker='o'),
                            ax.scatter([], [], [], alpha=1, s=20, marker='x', color=kalmanColor, linewidth=1),
                            ax.scatter([], [], [], alpha=1, s=40, marker='+', color=viconColor, linewidth=1.5)
                            )
        dataAndLines.append(bird)


    # Creating fifty line objects.
    # NOTE: Can't pass empty arrays into 3d version of plot()
    #lines = [ax.plot(dat[0:1, 0], dat[0:1, 1], dat[0:1, 2])[0] for dat in data]

    # Setting the axes properties
    field_of_view = 120
    ax.set_xlim3d([-field_of_view, field_of_view])
    ax.set_xlabel('X')

    ax.set_ylim3d([-field_of_view, field_of_view])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-field_of_view, field_of_view])
    ax.set_zlabel('Z')

    ax.set_title('3D Test')
    ax.view_init(elev=70.)

    # Creating the Animation object
    run_animation(dataAndLines, np.shape(quats)[1] - 1, trajectory_length, hide_expected_marker_locations, save_animation)

birdId = 5
firstFrame = 1500
lastFrame = 2000
trajectory_length = 100
hide_expected_marker_locations = False
use_corrected_vicon = True
show_legend = False
save_animation = False

#TODO encapsulate arguments
#TODO plot some graphs to compare vicon, corrected_vicon and kalman

#TODO write instructions:
# choose bird (from 0 to 9)
# zoom
# rotate
# which vicon to use
# starting frame
# animation speed
# pause
# switching birds, switch only visible after unpause


# at the end bird 0 has no detections anymore, i.e. it cannot be tracked
old(firstFrame, lastFrame, trajectory_length, hide_expected_marker_locations, use_corrected_vicon, show_legend, save_animation)