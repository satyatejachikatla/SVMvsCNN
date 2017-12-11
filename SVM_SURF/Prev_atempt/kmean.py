from scipy.cluster.vq import *
from scipy.spatial import distance

import numpy as np

#import set it to a fixed in all programs
k=15

desc=np.load('desc.npy')

print('Strat KMean')
centroids,dev=kmeans(desc, k_or_guess=k, iter=2, thresh=1e-05)
del desc
print('Done Kmean')

np.save('centroids',centroids)