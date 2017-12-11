from sklearn import svm
import numpy as np
from random import shuffle
from settings import *

# Flowers - 0 Cars - 1

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


Cars_x=list(np.load('Cars_feature_vectors.npy'))
Flowers_x=list(np.load('Flowers_feature_vectors.npy'))

for i in range(len(Cars_x)):
	Cars_x[i]=[Cars_x[i],0]
for i in range(len(Flowers_x)):
	Flowers_x[i]=[Flowers_x[i],1]

X=Cars_x+Flowers_x
shuffle(X)

y=[]
for i in range(len(X)):
	y.append(X[i][1])
	X[i]=X[i][0]
	
clf=svm.SVC()
clf.fit(X,y)

##
BOW = np.load('BOW.npy')
test_files = open('test.txt','r').readlines()
test_labels=[]
test_images=[]

for x in test_files:
    file , y = x.strip().split()
    test_images.append(cv2.imread(file))
    test_labels.append(int(y))
    
test_BOW_descriptors = compute_BOW_response(BOW,test_images,'SURF',None,None,k_size=K)
test_pred = clf.predict(test_BOW_descriptors)

c=0
for i in range(len(test_pred)):
        if test_pred[i] == test_labels[i]:
                c+=1
print('Acc',c/len(test_files))
##
