from PIL import Image
import glob
import numpy as np 

def scale(image, max_size, method=Image.ANTIALIAS):
    """
    resize 'image' to 'max_size' keeping the aspect ratio 
    and place it in center of white 'max_size' image 
    """
    im_aspect = float(image.size[0])/float(image.size[1])
    out_aspect = float(max_size[0])/float(max_size[1])
    if im_aspect >= out_aspect:
        scaled = image.resize((max_size[0], int((float(max_size[0])/im_aspect) + 0.5)), method)
    else:
        scaled = image.resize((int((float(max_size[1])*im_aspect) + 0.5), max_size[1]), method)
 
    offset = (int((max_size[0] - scaled.size[0]) / 2), int((max_size[1] - scaled.size[1]) / 2))
    back = Image.new("RGB", max_size, "black")
    back.paste(scaled, offset)
    return back


iconMap = glob.glob("cats/*")
iconMap+= glob.glob("dogs/*")

for i in range(len(iconMap)):
	image=Image.open(iconMap[i])
	scale(image,(256,256)).save(iconMap[i].split('.jpg')[0]+'_resized.jpg')
	print(i)
