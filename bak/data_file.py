from glob import glob
from random import shuffle


Flowers=glob('train/Flowers/*.jpg')[:5000]
Cars=glob('train/Cars/*.jpg')[:5000]


for i in range(len(Flowers)):
	Flowers[i]=[Flowers[i],'0']
for i in range(len(Cars)):
	Cars[i]=[Cars[i],'1']

shuffle(Flowers)
shuffle(Cars)

train_points=900
test_points=100
for i in range(5):
        x=Flowers[i*1000:i*1000+train_points]+Cars[i*1000:i*1000+train_points]
        y=Flowers[i*1000+train_points:i*1000+train_points+test_points]+Cars[i*1000+train_points:i*1000+train_points+test_points]

        shuffle(x)
        shuffle(y)
        
        file=open('Div/CNN/train'+str(i)+'.txt','a')
        file_=open('Div/SVM/train'+str(i)+'.txt','a')
        for i in x:
                file.write(i[0].split('.jpg')[0]+'_resized.jpg '+i[1]+'\n')
                file_.write(i[0]+' '+i[1]+'\n')
        file.close()
        file_.close()
        
        file=open('Div/CNN/test'+str(i)+'.txt','a')
        file_=open('Div/SVM/test'+str(i)+'.txt','a')
        for i in y:
                file.write(i[0].split('.jpg')[0]+'_resized.jpg '+i[1]+'\n')
                file_.write(i[0]+' '+i[1]+'\n')
        file.close()
        file_.close()
