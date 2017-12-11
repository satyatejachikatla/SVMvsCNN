import cv2
import numpy as np
from scipy.spatial import distance
import glob

#import set it to a fixed in all programs
k=15

def belongs(a):
	global centroids
	dst = list(distance.euclidean(a,b) for b in centroids)
	return np.argmax(dst)

centroids=np.load('centroids.npy')
dogs=glob.glob('../train/dogs_s/*.jpg')
dogs.sort()

dog_feature_vectors=[]

surf=cv2.xfeatures2d.SURF_create()
#surf.setExtended(True)

print('Start Vectorizing dog')

for i in range(len(dogs)):
    pic=cv2.imread(dogs[i])
    pic=cv2.cvtColor(pic,cv2.COLOR_RGB2GRAY)
    kp, des = surf.detectAndCompute(pic,None)

    vec=[0]*k
    for point in des:
    	vec[belongs(point)]+=1
    dog_feature_vectors.append(vec)
    #Counter to check how much done
    if i%1000==0:
    	print('Done...',i)

np.save('dog_feature_vectors',dog_feature_vectors)