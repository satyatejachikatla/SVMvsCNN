import cv2
import numpy as np
import glob
from random import shuffle
from scipy.spatial import distance
from settings import *


def create_bag_of_words(images, detector_type, k_size = 10):
    # Create an empty vocabulary with BOWKMeans
    vocabulary = cv2.BOWKMeansTrainer(clusterCount=k_size)

    if detector_type == 'SURF':
        detector = cv2.xfeatures2d.SURF_create()
        detector.setExtended(EXT)

    elif detector_type == 'SIFT':
        detector = cv2.xfeatures2d.SIFT_create()

    elif detector_type == 'KAZE':
        detector = cv2.KAZE_create()

    else:
        raise ValueError('Not a suitable detector')

    print("Creating the unclustered geometric vocabulary")

    descriptors, keypoints = [], []

    for img in images:
        # Detect the keypoints on the image and
        # compute the descriptor for those keypoints
        kp, descriptor = detector.detectAndCompute(img, None)
        descriptors.append(descriptor)
        keypoints.append(kp)
        vocabulary.add(descriptor)

    print("DONE!!")
    print("Creating the clusters with K-means")
    # K-Means clustering
    BOW = vocabulary.cluster()
    print("DONE!!")
    BOW = BOW.astype(np.float32)

    return BOW, keypoints, descriptors


def compute_BOW_response(BOW, images, detector_type,
                         keypoints, descriptors, k_size):

    # Create the Brute-Force Matcher
    matcher = cv2.BFMatcher(normType=cv2.NORM_L2)

    if detector_type == 'SURF':
        detector = cv2.xfeatures2d.SURF_create()
        detector.setExtended(EXT)
        detector.setHessianThreshold(HTHOLD)


    elif detector_type == 'SIFT':
        detector = cv2.xfeatures2d.SIFT_create()

    elif detector_type == 'KAZE':
        detector = cv2.KAZE_create()

    else:
        raise ValueError('Not a suitable detector')

    BOW_extractor = cv2.BOWImgDescriptorExtractor(dextractor=detector,
                                                  dmatcher=matcher)

    # Set the vocabulary for the BOW extractor,
    # in order to compute the histograms for the images
    BOW_extractor.setVocabulary(BOW)
    BOW_descriptors = np.zeros([len(images), k_size], dtype=np.float32)
    print(BOW_descriptors.shape)

    print("Computing the descriptors for the images")
    # Compute the histograms
    i = 0
    for img in images:
        hist = BOW_extractor.compute(img, detector.detect(img))
        BOW_descriptors[i] = hist[0].flatten()
        i+=1
    print("DONE!!")

    return np.array(BOW_descriptors)


'''
Cars_s=glob.glob('train/Cars/*.jpg')[:1000]
Flowers_s=glob.glob('train/Flowers/*.jpg')[:1000]
'''

X = open('train.txt','r').readlines()

Cars_s=[]
Flowers_s=[]

for x in X:
    file , y = x.strip().split()
    if y=='0':
        Flowers_s.append(file)
    elif y=='1':
        Cars_s.append(file)
        

Cars_images=[]
Flowers_images=[]

for file in Cars_s:
	Cars_images.append(cv2.imread(file))
for file in Flowers_s:
	Flowers_images.append(cv2.imread(file))

BOW,kps,des=create_bag_of_words(Cars_images+Flowers_images,'SURF',k_size=K)
np.save('BOW',BOW)

Cars_des=[]
Flowers_des=[]

surf=cv2.xfeatures2d.SURF_create()
surf.setExtended(EXT)
surf.setHessianThreshold(HTHOLD)

for img in Cars_images:
	kp,des=surf.detectAndCompute(img, None)
	Cars_des.append(des)
Cars_BOW_descriptors = compute_BOW_response(BOW,Cars_images,'SURF',None,None,k_size=K)
np.save('Cars_feature_vectors',Cars_BOW_descriptors)
del kp,des,Cars_BOW_descriptors,Cars_images

for img in Flowers_images:
	kp,des=surf.detectAndCompute(img, None)
	Flowers_des.append(des)
Flowers_BOW_descriptors = compute_BOW_response(BOW,Flowers_images,'SURF',None,None,k_size=K)
np.save('Flowers_feature_vectors',Flowers_BOW_descriptors)
del kp,des,Flower_BOW_descriptors,Flowers_images
