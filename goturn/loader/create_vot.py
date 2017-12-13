import cv2
import os
import pickle
import numpy as np
from goturn.helper.image_proc import cropPadImage
from goturn.helper.BoundingBox import BoundingBox

def load_label(file):
	with open(file, 'r') as f:
		data = f.read()
	labels = [map(float, line.split(',')) for line in data.split('\n')[:-1]]
	bbox_labels = []
	for i in range(len(labels)):
		x1 = min(labels[i][0], min(labels[i][2], min(labels[i][4], labels[i][6]))) - 1
		y1 = min(labels[i][1], min(labels[i][3], min(labels[i][5], labels[i][7]))) - 1
		x2 = max(labels[i][0], max(labels[i][2], max(labels[i][4], labels[i][6]))) - 1
		y2 = max(labels[i][1], max(labels[i][3], max(labels[i][5], labels[i][7]))) - 1
		bbox = BoundingBox(x1, y1, x2, y2)
		bbox_labels.append(bbox)
	return bbox_labels

def preprocess(image):
	"""
	Resizes the image to 227X227 to feed a fixed input to the network.
	Mean subtraction is part of original method, hence kept.
	Reshapes image to put color channels first.
	"""
	# print(image.shape)
	image = cv2.resize(image, (227, 227), interpolation=cv2.INTER_CUBIC)
	image = np.float32(image)
	mean = [104, 117, 123]
	image -= np.array(mean)
	# cv2.imshow('Results', image)
	# cv2.waitKey(0)
	image = np.transpose(image, [2, 0, 1])
	return image

vot_dir = "/home/manpk/Desktop/vision/project/vot2014/"
filenames = os.listdir(vot_dir)
filenames.pop(filenames.index('list.txt'))
filenames.pop(filenames.index('data'))

for file in filenames:

	if file in ["bicycle", "diving", "hand1", "motocross", "sphere", "trellis", "ball", "torus", "surfing", "skating", "jogging", "hand2", "fernando", "drunk", "car", "bolt", "fish1", "david"]:
		continue
	li = [f for f in os.listdir(vot_dir+file) if f.endswith('.jpg')]

	cropped_images = np.zeros((100, 3, 227, 227))
	cropped_target = np.zeros((100, 3, 227, 227))
	bbox_labels = load_label(vot_dir+file+'/groundtruth.txt')
	print(len(bbox_labels))
	ground_truth = np.zeros((100, 4))

	for i, f in enumerate(li):
		image = cv2.imread(vot_dir+file+'/'+f)
		search_region, search_location, edge_spacing_x, edge_spacing_y = cropPadImage(bbox_labels[i], image)
		# print(bbox_labels[i].y1, bbox_labels[i].y2)
		target = image[max(0, int(bbox_labels[i].y1)-10):min(image.shape[0], int(bbox_labels[i].y2)+10), max(0, int(bbox_labels[i].x1)-10):min(image.shape[1], int(bbox_labels[i].x2)+10), :]
		search_region = preprocess(search_region)
		target = preprocess(target)
		recentered = bbox_labels[i].recenter(search_location, edge_spacing_x, edge_spacing_y, BoundingBox(0.0, 0.0, 0.0, 0.0))
		cropped_images[i%100] = search_region
		cropped_target[i%100] = target
		ground_truth[i%100, :] = np.array([recentered.x1, recentered.y1, recentered.x2, recentered.y2])
		if (i%100==99) or (i==len(li)-1):
			print(file)
			with open(vot_dir+"data/"+file+str(i)+".search.pkl", "w") as f:
				pickle.dump(cropped_images[:(i%100)+1], f)
			with open(vot_dir+"data/"+file+str(i)+".target.pkl", "w") as f:
				pickle.dump(cropped_target[:(i%100)+1], f)
			with open(vot_dir+"data/"+file+str(i)+".groundtruth.pkl", "w") as f:
				pickle.dump(ground_truth[:(i%100)+1], f)
