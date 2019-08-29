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

print(np.sum(posX))
pos = np.stack([posX, posY, posZ], axis=2)

