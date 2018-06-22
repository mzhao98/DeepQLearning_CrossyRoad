'''
Freeway Game State Preprocessing Module
'''
import numpy as np
import cv2

# Game state preprocessing
def preprocess(observation):
    # resize to two dimensions and makes grayscale
    observation = cv2.cvtColor(cv2.resize(observation, (108, 118)), cv2.COLOR_BGR2GRAY)
    # crop image (top is just the score and bottom is activision logo)
    observation = observation[9:109,8:108]
    # reshape to 100x100x1 matrix
    return np.reshape(observation,(100,100,1))

###############################################################################
# Uncomment to see how preprocessing works on first game frame
# action0 = 0  # do nothing
# observation0, reward0, terminal, info = env.step(action0)
# print("Before processing: " + str(np.array(observation0).shape))
# plt.imshow(np.array(observation0))
# plt.show()
# observation0 = pp.preprocess(observation0)
# print("After processing: " + str(np.array(observation0).shape))
# plt.imshow(np.array(np.squeeze(observation0)))
# plt.show()
###############################################################################
