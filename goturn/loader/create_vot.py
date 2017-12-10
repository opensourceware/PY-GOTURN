import cv2
import os
import pickle
import numpy as np

vot_dir = "/home/manpk/Desktop/vision/project/vot2014/"
filenames = os.listdir(vot_dir)
filenames.pop(filenames.index('list.txt'))

for file in filenames:
	li = os.listdir(vot_dir+file)
	img1 = cv2.imread(vot_dir+file+'/'+li[0])
	arr = np.zeros((100, 3, img1.shape[0], img1.shape[1]))
	for i, f in enumerate(li):
		if f.endswith('.jpg'):
			image = cv2.imread(vot_dir+file+'/'+f)
			arr[i%100] = image.reshape(3, img1.shape[0], img1.shape[1])
		if (i%100==99) or (i==len(li)-1):
			with open(vot_dir+"data/"+file+str(i)+".pkl", "w") as f:
				pickle.dump(arr, f)
			print(file)

