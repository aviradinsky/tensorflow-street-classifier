#%%

from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import time
import random
import multiprocessing


#%%

def display_crops(crops: np.array, count=5):
	for i in range(count):
		plt.figure()
		plt.imshow(crops[i])
		plt.show()

#%%

def display_bounding_boxes(img: np.array, points: list):
	# copy image so as not to draw bounding boxes on original image
	copy = img.copy()

	# to display, loop through all of the boxes in batches of 40
	for i in range(5):
		if i * 40 >= len(points):
			break
		# display bounding boxes in groups of 40
		for j in range(40): 
			if i * 40 + j >= len(points):
				break

			current_rect = points[i * 40 + j]
			
			colors = []
			for _ in range(3):
				colors.append(random.randint(0,255))

			copy = cv.rectangle(copy, pt1=current_rect[0], pt2=current_rect[1],
							color=tuple(colors), thickness=4)

		plt.figure()
		plt.imshow(copy)
		plt.show()

		# copy = cv.cvtColor(cv.imread('./test_images/predict_img.jpg'), cv.COLOR_BGR2RGB)
		copy = img.copy()

#%%

def selective_search(img: np.array, display_boxes=False):
	"""Runs a selective search implementation on the inputted image.
	Returns a list of numpy.array of crops from the selective
	search algorithm.

	Args:
		img (np.array): input image.
		display_boxes (bool, optional): Defaults to False.

	Returns:
        list: a list tuples. Each crop is represented by a tuple of 
        length 2, where the array of the crop is stored at position
        [0] and its bbox in the original image is stored at 
        position [1].  The order of the bbox is:
                                (left, top, right, bottom)
	"""
	# speed-up using multithreads
	cv.setUseOptimized(True);
	cv.setNumThreads(multiprocessing.cpu_count());

	# initialize OpenCV's selective search implementation and set the
	# input image
	ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
	ss.setBaseImage(img)
	# check to see if we are using the *fast* but *less accurate* version
	# of selective search
	# if args["method"] == "fast":
	print("[INFO] using *fast* selective search")
	ss.switchToSelectiveSearchFast()
	# otherwise we are using the *slower* but *more accurate* version
	# else:
	# print("[INFO] using *quality* selective search")
	# ss.switchToSelectiveSearchQuality()
	
	# run selective search on the input image
	start = time.time()
	rects = ss.process()
	end = time.time()

	# show how along selective search took to run along with the total
	# number of returned region proposals
	print("[INFO] selective search took {:.4f} seconds".format(end - start))
	print("[INFO] {} total region proposals".format(len(rects)))

	# collect the points defining the bounding boxes
	points = []
	crops = []

	# loop over the rectangles generated by selective search
	for (x, y, w, h) in rects:
		# slice array to get crop
		crop = img[y:y + h, x:x + w]
		# create bbox tuple (left, upper, right, lower)
		left = x
		upper = y
		right = x + w
		lower = y + h
		bbox = (left, upper, right, lower)
		crops.append((crop, bbox))
		if display_boxes:
			# convert bounding boxes from (x, y, w, h) to (left,
			# upper), (right, lower)
			points.append(((x, y), (x + w, y + h)))

	if display_boxes:
		display_bounding_boxes(img, points)

	return crops

#%%

def test():
	image = cv.cvtColor(cv.imread('./test_images/predict_img.jpg'), cv.COLOR_BGR2RGB)
	crops = selective_search(image, display_boxes=True)
	# get just the pictures
	crops = [img for img, tup in crops]
	display_crops(crops)

#%%

if __name__ == '__main__':
	test()

# %%
