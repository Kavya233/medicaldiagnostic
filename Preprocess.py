import h5py
import cv2
import numpy as np
from matplotlib import pyplot as plt

def process(path):
	files={}
	array = {}
	cnt=0
	for i in range(1,3065):
	    filepath=path +"/"+ str(i) + '.mat'
	    f=h5py.File(filepath)
	    x=f['cjdata']
	    img=np.array(x['image'])
	
	    array['img']=np.array(x['image'])
	    array['label']=int(np.array(x['label'])[0][0])
	    array['tumormask'] = np.array(x['tumorMask'])
	    print(i)
	    if array['label'] == 1 :
	        cv2.imwrite(r"data\L1/" +str(i) + '.jpg', img)
	    if array['label'] == 2 :
	        cv2.imwrite(r"data\L2/" +str(i) + '.jpg', img)
	    if array['label'] == 3 :
	        cv2.imwrite(r"data\L3/" + str(i) + '.jpg', img)
	    files[i]=array
	    
	    