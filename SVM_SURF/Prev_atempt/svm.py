from sklearn import svm
import numpy as np
from random import shuffle

# CATS - 0 DOGS -1

cat_x=list(np.load('cat_feature_vectors.npy'))
dog_x=list(np.load('dog_feature_vectors.npy'))

cat_test=cat_x[:1000]
dog_test=dog_x[:1000]

cat_x=cat_x[1000:]
dog_x=dog_x[1000:]

for i in range(len(cat_x)):
	cat_x[i]=[cat_x[i],0]
for i in range(len(dog_x)):
	dog_x[i]=[dog_x[i],1]

X=cat_x+dog_x
shuffle(X)

y=[]
for i in range(len(X)):
	y.append(X[i][1])
	X[i]=X[i][0]
	

clf=svm.SVC()
clf.fit(X,y)

cat_pred=clf.predict(cat_test)
dog_pred=clf.predict(dog_test)

print('Cats acc',1-sum(cat_pred)/len(cat_pred))
print('Dogs acc',sum(dog_pred)/len(dog_pred))


