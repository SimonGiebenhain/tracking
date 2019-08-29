import numpy as np
import pickle
from random import choice
from math import floor
from viz import visualize

path = '../behaviour/'
filenames = ['isFlying', 'isStarting', 'isLanding', 'isWalking', 'isSitting']
files = {}
for i in range(len(filenames)):
    with open(path + filenames[i] + '.pkl', 'rb') as fin:
        files[filenames[i]] = pickle.load(fin)


with open(path + 'positionsX.pkl', 'rb') as fin:
    posX = pickle.load(fin)
with open(path + 'positionsY.pkl', 'rb') as fin:
    posY = pickle.load(fin)
with open(path + 'positionsZ.pkl', 'rb') as fin:
    posZ = pickle.load(fin)

pos = np.stack([posX[:,:-1], posY[:,:-1], posZ[:,:-1]], axis=2)

#print(np.shape(pos[files['isFlying'] != 0,:]))


# Extracts all trajectories for a specified behaviour as a list of 2d numpy arrays
def get_behaviour_snippets(pos, behaviour_mask, n_objects, min_length):
    T = np.shape(behaviour_mask)[1]
    snippets = []
    for k in range(n_objects):
        inAction = False
        actionLength = 0

        for t in range(T):
            if not inAction:
                if behaviour_mask[k, t]:
                    inAction = True
                    actionLength = 1
            else:
                if behaviour_mask[k, t] and t < T - 1:
                    actionLength += 1
                elif actionLength > min_length:
                    #save action snippet
                    snip = np.squeeze(pos[k,t-actionLength:t, :])
                    if np.sum(np.isnan(snip)) > 0:
                        print(snip)
                        print('HHHHHHAAAAAAALLLLLLLLLLLLLLOOOOOO')
                    snippets.append(snip)
                    inAction = False
                    actionLength = 0
                else:
                    inAction = False
                    actionLength = 0

    return snippets


# Cuts snippets such that all snippets have length 'snippet_length'
def split_snippets(snippets, snippet_length):
    new_snippets = []
    for snippet in snippets:
        if np.shape(snippet)[0] % snippet_length != 0:
            n_splits = floor(np.shape(snippet)[0] / snippet_length)
            splits = np.split(snippet[:snippet_length*n_splits,:], n_splits, axis=0)
            splits.append(snippet[-snippet_length:])
        else:
            n_splits = int(np.shape(snippet)[0] / snippet_length)
            splits = np.split(snippet, n_splits,axis=0)
        new_snippets += splits
    return new_snippets


# get equal length snippets for all behaviours
flying_behaviour = split_snippets(get_behaviour_snippets(pos, files['isFlying'], 10, 10), 10)
starting_behaviour = split_snippets(get_behaviour_snippets(pos, files['isStarting'], 10, 10), 10)
landing_behaviour = split_snippets(get_behaviour_snippets(pos, files['isLanding'], 10, 10), 10)
walking_behaviour = split_snippets(get_behaviour_snippets(pos, files['isWalking'], 10, 10), 10)
sitting_behaviour = split_snippets(get_behaviour_snippets(pos, files['isSitting'], 10, 30), 10)


#print(len(flying_behaviour))
#print(len(starting_behaviour))
#print(len(landing_behaviour))
#print(len(sitting_behaviour))
#print(len(walking_behaviour))


# offset_noise basically has no effect
def center_snippet(snippet, offset):
    #offset_std =
    mean_pos = np.mean(snippet, axis=0)
    #rnd_offset = np.random.normal(loc=0, scale=offset_st, size=[1,3])
    snippet = snippet - mean_pos
    pos0 = snippet[0, :]
    return snippet - pos0 + offset #+ rnd_offset

# rotate snippet inorder to get more variety
# maybe don't change direction of z-axis in order to keep gravity in tact
# TODO: should I add little rotation  along arbitray axis?
def rotate_snippet(snippet, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    rotated_xy = np.matmul(R, snippet[:,:2].T).T
    return np.concatenate( [rotated_xy, np.expand_dims(snippet[:,2], axis=1)], axis=1)

def scale_snippet(snippet):
    scale_std = 0.5/3
    scale = np.random.normal(loc=1, scale=scale_std, size=[1,3])
    scale = np.maximum(0.5 * np.ones([1,3]), scale)
    scale = np.minimum(1.5 * np.ones([1,3]), scale)
    #TODO remove this line
    scale = np.ones([1,3])
    return snippet * scale

# counterclockwise angle from v to u
def get_counter_clockwise_anlge(v, u):
    return np.arcsin((v[0]*u[1] - v[1]*u[0])/(np.linalg.norm(v)*np.linalg.norm(u)))


# TODO figure out what is wrong!!!
for i in range(1000):
    u = np.random.normal(0,1,[3])
    u[2] = 0
    v = np.random.normal(0, 1, [3])
    v[2] = 0
    theta = get_counter_clockwise_anlge(v/np.linalg.norm(v), u/np.linalg.norm(u))
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    rot_v = np.matmul(R, v[:2])
    rot_v3 = np.zeros(3)
    rot_v3[:2] = rot_v
    rot_v3[2] = 0
    new_angle = get_counter_clockwise_anlge(rot_v, u)

    theta = get_counter_clockwise_anlge(v/np.linalg.norm(v), u/np.linalg.norm(u))
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    rot_v = np.matmul(R, v[:2])
    rot_v3 = np.zeros(3)
    rot_v3[:2] = rot_v
    rot_v3[2] = 0
    new_angle2 = get_counter_clockwise_anlge(rot_v, u)

    if min(new_angle, new_angle2) > 0.01:
        print(new_angle)
        print(u)
        print(v)

u = np.array([0.4, 0.06, 0])
v = np.array([0.136, 1.05, 0])

theta = -get_counter_clockwise_anlge(v, u)
c, s = np.cos(theta), np.sin(theta)
R = np.array(((c, -s), (s, c)))
rot_v = np.matmul(R, v[:2])
print(rot_v/np.linalg.norm(rot_v))
print(u/np.linalg.norm(u))


print('DDDOOOONNNNEEEEE')


# Walk through markov chain specified by transitions_probs for n_steps
# Generates trajectory in 2d numpy array of shape (n_steps * snippet_length) x (3)
def simulate_trajectory(snippets, transition_probs, n_steps):
    assert len(snippets) == np.shape(transition_probs)[0] and \
           np.shape(transition_probs)[0] == np.shape(transition_probs)[1]
    n_states = len(snippets)
    snippet_length = np.shape(snippets[0][0])[0]
    state = np.random.randint(1, n_states)
    trajectory = np.zeros([n_steps * snippet_length, 3])
    trajectory[0:snippet_length, :] = center_snippet(choice(snippets[state]), 0)
    for t in range(1, n_steps):
        snippet = choice(snippets[state])
        probs = transition_probs[state, :]
        state = np.argmax(np.random.multinomial(1, probs))
        offset = trajectory[t*snippet_length-1, :]
        if state != 0:
            # TODO enforce some kind of smoothness
            past_direction = trajectory[t*snippet_length-1, :] - trajectory[t*snippet_length-2, :]
            past_direction[2] = 0
            new_direction = snippet[1, :] - snippet[0, :]
            new_direction[2] = 0
            if np.linalg.norm(past_direction) < 0.1 or np.linalg.norm(new_direction) < 0.1:
                trajectory[t * snippet_length:(t + 1) * snippet_length, :] = scale_snippet(center_snippet(snippet, offset))
            else:
                rnd_rot_theta = get_counter_clockwise_anlge(new_direction, past_direction)
                #rnd_rot_theta = np.random.normal(rnd_rot_theta, scale=0.02)
                old_snippet = snippet
                snippet = rotate_snippet(center_snippet(snippet, np.zeros([1, 3])), rnd_rot_theta)
                past_direction = trajectory[t * snippet_length - 1, :] - trajectory[t * snippet_length - 2, :]
                new_direction = snippet[1, :] - snippet[0, :]
                past_direction[2] = 0
                new_direction[2] = 0
                if get_counter_clockwise_anlge(new_direction, past_direction) > 0.001:
                    snippet = rotate_snippet(center_snippet(old_snippet, np.zeros([1, 3])), rnd_rot_theta)
                past_direction = trajectory[t * snippet_length - 1, :] - trajectory[t * snippet_length - 2, :]
                new_direction = snippet[1, :] - snippet[0, :]
                if get_counter_clockwise_anlge(new_direction, past_direction) > 0.0000001:
                    print(get_counter_clockwise_anlge(new_direction, past_direction))
                trajectory[t*snippet_length:(t+1)*snippet_length, :] = scale_snippet(center_snippet(snippet, offset))
        else:
            trajectory[t*snippet_length:(t+1)*snippet_length, :] = scale_snippet(center_snippet(snippet, offset))

    return trajectory

biggest_p = 0.5
bigger_p = 0.25
big_p = 0.2
small_p = 0.05
tiny_p = 0

assert biggest_p + bigger_p +  big_p + small_p + tiny_p == 1

transition_matrix = np.array([ [biggest_p, bigger_p, big_p, small_p, tiny_p],
                               [bigger_p, biggest_p, big_p, small_p, tiny_p],
                               [tiny_p, small_p, biggest_p, bigger_p, big_p],
                               [small_p, big_p, tiny_p, biggest_p, bigger_p],
                               [bigger_p, big_p, small_p, tiny_p, biggest_p]])

assert np.array_equal(np.sum(transition_matrix, axis=1), np.ones([np.shape(transition_matrix)[0]]))

snippets = [sitting_behaviour, walking_behaviour, starting_behaviour, flying_behaviour, landing_behaviour]

traj = simulate_trajectory(snippets, transition_matrix, 100)
print(np.shape(traj))
print(traj)

traj = traj / np.max(np.abs(traj), axis=0)

visualize(np.expand_dims(traj,axis=1), isNumpy=True)




# TODO 3: define markov chain to generate new behaviour
# TODO 4: visualize results
# TODO 5: refine markov chain in order to get smooth results, maybe with curves to transit in flying mode





