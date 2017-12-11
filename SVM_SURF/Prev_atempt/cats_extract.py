import numpy as np
import glob
import cv2

cats=glob.glob('../train/cats_s/*.jpg')
cats.sort()

print('Starting Reading & Extract')

print('Cats')

surf=cv2.xfeatures2d.SURF_create()
#surf.setExtended(True)

faltten=open('faltten','a')

print('Starting extracting featurs-cat')
temp_i=0
for path in cats:
	
	pic=cv2.imread(path)
	pic=cv2.cvtColor(pic,cv2.COLOR_RGB2GRAY)
	kp, des = surf.detectAndCompute(pic,None)
	
	for i in des:
		for j in i:
			faltten.write(str(j)+' ')
		faltten.write('\n')
	temp_i+=1
	if temp_i%1000==0:
		print('Done...',temp_i)
