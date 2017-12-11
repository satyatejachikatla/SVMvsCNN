import numpy as np 

desc=open('faltten','r').readlines()

l=len(desc)
for i in range(l):
	desc[i]=np.float32(desc[i].strip().split())
	if(i%10000==0):
		print(i)
np.save('desc',desc)