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
cats=glob.glob('../train/cats_s/*.jpg')
cats.sort()

cat_feature_vectors=[]

surf=cv2.xfeatures2d.SURF_create()
#surf.setExtended(True)

print('Start Vectorizing Cat')

for i in range(len(cats)):
    pic=cv2.imread(cats[i])
    pic=cv2.cvtColor(pic,cv2.COLOR_RGB2GRAY)
    kp, des = surf.detectAndCompute(pic,None)

    vec=[0]*k
    for point in des:
    	vec[belongs(point)]+=1
    cat_feature_vectors.append(vec)
    #Counter to check how much done
    if i%1000==0:
    	print('Done...',i)

np.save('cat_feature_vectors',cat_feature_vectors)