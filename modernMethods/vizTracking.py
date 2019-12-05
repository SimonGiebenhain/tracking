######################################
########### INSTRUCTIONS #############
# This file visualizes the tracking results from VICON and a Kalman Filter in a 3d animation.
# It is possible to zoom(press right mouse button and move mouse up and down) and rotate(press left mouse button and move mouse around)
# But the user cannot move inside the animation. Therefore the animation follows a specific bird.
#
# color code: My algorithm has brighter colors. VICON has a similar color but a little more dull.
#
# In the following I explain how to use this file:
# 1) Choose the bird to follow by setting birdId to a value from 0 o 9
# 2) You can pause the animation by pressing the space bar and unpause by pressing it again.
#     Attetion: When you switch to another bird (as explained in (1)) while the animation is paused, you will only see the switch after you unpause. I don't want to waste too much time on this.
# 3) Specify the first and last frame of the animation. somewhere between 0 and 2960, lastFrame = -1 will use the actual last frame as last frame
# 4) shown_trajectory_length tells how long the past trajectory is shown. i.e. 100 will show the trajectory of every bird for the last 100 frames, -1 will show the complete trajectory
# 5) Since VICON and my algorithm are both predicting the location and rotation of a bird, we can apply this knowledge to the pattern of a bird.
#     This way we can see where the algorithm expects the 4 markers of the bird to be. When these expected marker locations correspond with the detections (big blue circles), the algorithm seems correct.
#     The expected markers have the form of an 'x' for my algorithm and '+' for VICON.
#     You can hide these by setting: hide_expected_marker_locations = True
# 6) When setting use_corrected_vicon = True, the animation will show the VICON predictions I corrected with my algorithm. Corrected means that I undid the ID-switches. But there are sof course still holes in the VICON data.
#    Set this to False if you want to show the unaltered VICON data
# 7) show_ledgend = True will show a ledgend, but the ledgend is annoying

import numpy as np
import pickle

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

from pyquaternion import Quaternion as Quaternion

import os, os.path

birdId = 9
firstFrame = 1500
lastFrame = -1
shown_trajectory_length = 100
hide_expected_marker_locations = False
use_corrected_vicon = True
show_legend = False
save_animation = False


# Attaching 3D axis to the figure
#fig = plt.figure()
#ax = p3.Axes3D(fig)
#patterns = np.load('data/patterns.npy')


patterns = np.load('data/patterns.npy')

def rot_mat_from_6d(rot):
    column1 = rot[:3] / np.linalg.norm(rot[:3])
    tmp = np.sum(column1* rot[3:]) * column1
    column2 = rot[3:] - tmp
    column3 = np.cross(column1, column2)
    return np.stack([column1, column2, column3], axis=0)


class MetaParams:
    def __init__(self, first_frame, last_frame, use_corrected_vicon):
        self.first_frame = first_frame
        self.last_frame = last_frame
        self.use_corrected_vicon = use_corrected_vicon


class AnimationParams:
    def __init__(self, shown_trajectory_length, hide_expected_marker_locations, show_legend, save_animation):
        self.shown_trajectory_length = shown_trajectory_length
        self.hide_expected_marker_locations = hide_expected_marker_locations
        self.show_legend = show_legend
        self.save_animation = save_animation


class TrackedObject:
    def __init__(self, predicted_pos, predicted_rot, true_pos, true_rot, pattern, detections,
                 predicted_line, true_line, scatter, predicted_markers, true_markers, rot_representation):
        self.predicted_pos = predicted_pos
        self.predicted_rot = predicted_rot
        self.true_pos = true_pos
        self.true_rot = true_rot
        self.pattern = pattern
        self.detections = detections
        self.predicted_line = predicted_line
        self.true_line = true_line
        self.scatter = scatter
        self.predicted_markers = predicted_markers
        self.true_markers = true_markers
        self.rot_representation = rot_representation


def run_tracking_animation(tracked_objects, length, animation_params, fig, ax):
    if animation_params.save_animation:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Simon Giebenhain'))

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

    def update_tracks(t, tracked_objects, shown_trajectory_length, hide_markers):
        #if t == 0:
        #    return []
        lines = []
        for object in tracked_objects:
            if object.predicted_pos is not None:
                object.predicted_line.set_data(object.predicted_pos[:t+1, 0:2].T)
                object.predicted_line.set_3d_properties(object.predicted_pos[:t+1, 2])
                lines.append(object.predicted_line)

                object.true_line.set_data(object.true_pos[:t+1, 0:2].T)
                object.true_line.set_3d_properties(object.true_pos[:t+1, 2])
                lines.append(object.true_line)

            if object.rot_representation == 'quat':
                q_predicted = Quaternion(object.predicted_rot[t, :])
                rot_mat_predicted = q_predicted.rotation_matrix
            else:
                rot_mat_predicted = rot_mat_from_6d(object.predicted_rot[t, :])

            rotated_pattern_predicted = (rot_mat_predicted @ object.pattern.T).T
            if object.predicted_pos is not None:
                predicted_markers = rotated_pattern_predicted + object.predicted_pos[t, :]
            else:
                predicted_markers = rotated_pattern_predicted

            # now with VICON predictions
            #if object.rot_representation == 'quat':
            q_true = Quaternion(object.true_rot[t, :])
            rot_mat_true = q_true.rotation_matrix
            #else:
            #    rot_mat_true = rot_mat_from_6d(object.true_rot[t, :])
            rotated_pattern_true = (rot_mat_true @ object.pattern.T).T
            if object.predicted_pos is not None:
                true_markers = rotated_pattern_true + object.true_pos[t, :]
            else:
                true_markers = rotated_pattern_true

            # display expected marker locations
            object.predicted_markers._offsets3d = (predicted_markers[:, 0],
                                                   predicted_markers[:, 1],
                                                   predicted_markers[:, 2])
            object.true_markers._offsets3d = (true_markers[:, 0],
                                              true_markers[:, 1],
                                              true_markers[:, 2])
            lines.append(object.predicted_markers)
            lines.append(object.true_markers)

        object = tracked_objects[0]
        object.scatter._offsets3d = (object.detections[t, 0::3],
                                     object.detections[t, 1::3],
                                     object.detections[t, 2::3])
        lines.append(object.scatter)

        return lines

    fig.canvas.mpl_connect('key_press_event', on_press)

    anim = animation.FuncAnimation(fig, update_tracks, length,
                                   fargs=(tracked_objects, animation_params.shown_trajectory_length,
                                          animation_params.hide_expected_marker_locations),
                                   interval=50, blit=False, repeat=True)

    if animation_params.show_legend:
        ax.legend()
    if animation_params.save_animation:
        print('SAVING')
        birdnames = load_pattern()
        filename = 'tracking.mp4'#.format(birdnames[birdId])
        anim.save(filename, writer=writer)
    else:
        print('SHOWING')
        plt.show()



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


def run_animation(dataAndLines, length, animation_params, ax, fig):

    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Simon Giebenhain'))

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
                pattern = patterns[id, :, :]

                # calculate the expected marker locations
                # first with the kalman predictions
                q = Quaternion(bird.kalmanQuat[t, :])
                rot_mat = q.rotation_matrix
                rotated_pattern = np.dot(rot_mat, pattern.T).T
                expected_markers_kalman = rotated_pattern + bird.kalmanPos[t, :]

                # now with VICON predictions
                q_vicon = Quaternion(bird.viconQuat[t, :])
                rot_mat_vicon = q_vicon.rotation_matrix
                rotated_pattern_vicon = np.dot(rot_mat_vicon, pattern.T).T
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
                                       fargs=(dataAndLines, animation_params.shown_trajectory_length,
                                              animation_params.hide_expected_marker_locations),
                                       interval=20, blit=False, repeat=True)

    if animation_params.show_legend:
        ax.legend()
    if animation_params.save_animation:
        birdnames = load_pattern()
        filename = 'clips/bird_{}.mp4'.format(birdnames[birdId])
        anim.save(filename, writer=writer)
    else:
        plt.show()


def loadColors():
    path = 'data/'
    with open(path + 'colors.pkl', 'rb') as fin:
        colors = pickle.load(fin)
    np.save('data/colors.npy', colors)


def load_pattern():
    patterns = []
    birdnames = []
    path = '../behaviour/'
    for f in sorted(os.listdir(path)):
        ext = os.path.splitext(f)[0]
        if not '.vsk' in ext:
            continue
        birdname = ext.split('.')[0]
        birdnames.append(birdname)
        with open(os.path.join(path, f), 'rb') as fin:
            patterns.append(pickle.load(fin))

    np.save('data/patterns.npy', np.stack(patterns, axis=0))
    return birdnames


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


def old(meta_params, animation_params):
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    #global ax
    global birdId
    # load data
    pos = np.load('data/pos.npy')
    pos = pos[:, meta_params.first_frame:meta_params.last_frame, :]
    quats = np.load('data/quats.npy')
    quats = quats[:, meta_params.first_frame:meta_params.last_frame, :]
    if meta_params.use_corrected_vicon:
        viconPos = np.load('data/corrected_vicon_pos.npy')
        viconPos = viconPos[:, meta_params.first_frame:meta_params.last_frame, :]
        viconQuats = np.load('data/corrected_vicon_quats.npy')
        viconQuats = viconQuats[:, meta_params.first_frame:meta_params.last_frame, :]
    else:
        viconPos = np.load('data/viconPos.npy')
        viconPos = viconPos[:, meta_params.first_frame:meta_params.last_frame, :]
        viconQuats = np.load('data/viconQuats.npy')
        viconQuats = viconQuats[:, meta_params.first_frame:meta_params.last_frame, :]
    detections = np.load('data/detections.npy')
    detections = detections[meta_params.first_frame:meta_params.last_frame, :, :]

    colors = np.load('data/colors.npy')

    birdnames = load_pattern()
    dataAndLines = []
    center = pos[birdId, 0, :]
    for k in range(10):
        kalmanColor = colors[k,:]

        viconColor =(kalmanColor+1.5)/(np.max(kalmanColor)+1.5)
        viconColor = viconColor * 0.75
        if k == 0:
            bird = BirdData(pos[k,:,:], quats[k,:,:], viconPos[k,:,:], viconQuats[k,:,:], detections,
                            ax.plot(pos[k, 0:1, 0]-center, pos[k, 0:1, 1]-center, pos[k, 0:1, 2]-center, color=kalmanColor,
                                    linewidth=1, label='bird {}'.format(birdnames[k]))[0],
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
                                    linewidth=1, label='bird {}'.format(birdnames[k]))[0],
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
    ax.view_init(elev=50.)

    # Creating the Animation object
    run_animation(dataAndLines, np.shape(quats)[1] - 1, animation_params, ax, fig)


def save_clips():
    firstFrame = 0
    lastFrame = -1
    shown_trajectory_length = 100
    hide_expected_marker_locations = False
    use_corrected_vicon = True
    show_legend = False
    save_animation = True
    meta_params = MetaParams(firstFrame, lastFrame, use_corrected_vicon)
    animation_params = AnimationParams(shown_trajectory_length, hide_expected_marker_locations, show_legend,
                                       save_animation)
    old(meta_params, animation_params)


def plot_bird_traj(name1, name2):
    plt.subplot(121)
    pos = np.load(name1)
    colors = np.load('data/colors.npy')
    n_objects = pos.shape[0]
    print(n_objects)
    for k in range(n_objects):
        traj = pos[k, :, 0]
        plt.plot(traj, c=colors[k, :])
    plt.subplot(122)
    pos = np.load(name2)
    colors = np.load('data/colors.npy')
    n_objects = pos.shape[0]
    print(n_objects)
    for k in range(n_objects):
        traj = pos[k, :, 0]
        plt.plot(traj, c=colors[k, :])
    plt.show()

def visualize_tracking(predicted_pos, predicted_rot, true_pos, true_rot, detections, pattern, rot_representation):
    animation_params = AnimationParams(-1, True, False, False)


    fig = plt.figure()
    ax = p3.Axes3D(fig)
    if predicted_pos is None:
        track_viz = TrackedObject(None, predicted_rot, None, true_rot, pattern, detections,
                                  None,
                                  None,
                                  ax.scatter([], [], [], c='r'),
                                  ax.scatter([], [], [], c='g'),
                                  ax.scatter([], [], [], c='b'),  # TODO: add colors
                                  rot_representation
                                  )
    else:
        track_viz = TrackedObject(predicted_pos, predicted_rot, true_pos, true_rot, pattern, detections,
                                  ax.plot(predicted_pos[0:1, 0], predicted_pos[0:1, 1], predicted_pos[0:1, 2])[0],
                                  ax.plot(true_pos[0:1, 0], true_pos[0:1, 1], true_pos[0:1, 2])[0],
                                  ax.scatter([], [], [], c='gray'),
                                  ax.scatter([], [], []),
                                  ax.scatter([], [], []), #TODO: add colors
                                  rot_representation
                                  )

    # TODO: set inital view params, etc.
    field_of_view = 5
    ax.set_xlim3d([-field_of_view, field_of_view])
    ax.set_xlabel('X')

    ax.set_ylim3d([-field_of_view, field_of_view])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-field_of_view, field_of_view])
    ax.set_zlabel('Z')

    ax.set_title('3D Test')
    ax.view_init(elev=50.)

    if predicted_pos is not None:
        run_tracking_animation([track_viz], np.shape(predicted_pos)[0] - 1, animation_params, fig, ax)
    else:
        run_tracking_animation([track_viz], np.shape(predicted_rot)[0] - 1, animation_params, fig, ax)




meta_params = MetaParams(firstFrame, lastFrame, use_corrected_vicon)
animation_params = AnimationParams(shown_trajectory_length, hide_expected_marker_locations, show_legend, save_animation)



# at the end bird 0 has no detections anymore, i.e. it cannot be tracked
#save_clips()


#old(meta_params, animation_params)
#plot_bird_traj('data/viconPos.npy', 'data/pos.npy')
