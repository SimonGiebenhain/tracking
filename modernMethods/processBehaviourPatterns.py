import numpy as np
import pickle

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
                    snippets.append(np.squeeze(pos[k,t-actionLength:t, :]))
                    inAction = False
                    actionLength = 0
                else:
                    inAction = False
                    actionLength = 0

    return snippets


flying_behaviour = get_behaviour_snippets(pos, files['isFlying'], 10, 20)
starting_behaviour = get_behaviour_snippets(pos, files['isStarting'], 10, 10)
landing_behaviour = get_behaviour_snippets(pos, files['isLanding'], 10, 10)
walking_behaviour = get_behaviour_snippets(pos, files['isWalking'], 10, 10)
sitting_behaviour = get_behaviour_snippets(pos, files['isSitting'], 10, 30)

# TODO 1: center snippets
# TODO 2: function which augments single snippet, e.g. rotation
# TODO 3: define markov chain to generate new behaviour





