# %%

from matplotlib import pyplot as plt

from PIL import Image

from tensorflow.image import non_max_suppression as tfnms
import tensorflow as tf
import numpy as np
import cv2 as cv
import random
import sys
import os

import params

# %%

def display_image(img):
    plt.imshow(img.astype('uint8'))
    plt.show()

# %%

# square = np.zeros([1024,1024,3])
# display_image(square)
# img = None

# boxes = [
#     (5,5,300,1000),
#     (400,740,893,295),
#     (374,640,817,490),
#     (300,845,800,400),
#     (600,800,922,663),
#     (507,565,826,267)
# ]

# scores = [
#     .999,
#     .99,
#     .94,
#     .73,
#     .53,
#     .34
# ]

# max_output_size = 3

# for i in range(len(boxes)):
#     box = boxes[i]
#     color = []
#     for _ in range(3):
#             color.append(random.randint(0,255))
#     pt1 = (box[0], box[1])
#     pt2 = (box[2], box[3])
#     img = cv.rectangle(square, pt1, pt2, color, thickness=3)

# display_image(img)

# # %%

# reverse = [(b, a, d, c) for (a, b, c, d) in boxes]
# nparray = np.array(reverse)
# boxes_tensor = tf.constant(nparray, dtype='float32')
# print(boxes_tensor)
# scores_tensor = tf.constant(scores, dtype='float32')

# selected_indices = tfnms(boxes_tensor, scores_tensor, max_output_size)
# selected_boxes = tf.gather(boxes, selected_indices)
# selected_boxes = [x.numpy() for x in selected_boxes]
# reshaped = np.reshape(selected_boxes, [len(selected_boxes) * 4,])

# results = []

# for i in range(0, len(reshaped), 4):
#     results.append((reshaped[i], reshaped[i + 1],
#                    reshaped[i + 2], reshaped[i + 3]))

# print('results: ', results)
# print(boxes)

# test_img = np.zeros((1024, 1024, 3))

# for i in results:
#     pt1 = (i[0], i[1])
#     pt2 = (i[2], i[3])
#     color = []
#     for _ in range(3):
#             color.append(random.randint(0,255))
#     test_img = cv.rectangle(test_img, pt1, pt2, color, thickness=3)

# display_image(test_img)
# %%

def nms(bboxes: list, scores: list, max_output_size=1):
    """Runs tensorflow's non max suppression implentation on the
    given bboxes.

    Args:
        bboxes (list): list of bboxes with coordinates in order of:
                                            (left, top, right, bottom)
        scores (list): list of prediction probabilities for each bbox
        max_output_size (int, optional): max output size. Defaults to 1.
    
    Returns:
        bboxes: a list of bbox tuples returned by the tf nms algorithm.
    """
    print('len bboxes: ' + str(len(bboxes)))
    print('len scores: ' + str(len(scores)))
    # revers x and y coordinates to align with tf input reqs
    reverse = [(b, a, d, c) for (a, b, c, d) in boxes]
    # convert to tensor
    bbox_array = np.array(reverse)
    boxes_tensor = tf.constant(bbox_array, dtype='float32')
    scores_tensor = tf.constant(scores, dtype='float32')
    # get results from nms as list of numbers
    selected_indices = tfnms(boxes_tensor, scores_tensor, max_output_size)
    selected_boxes = tf.gather(boxes, selected_indices)
    selected_boxes = [x.numpy() for x in selected_boxes]
    reshaped = np.reshape(selected_boxes, [len(selected_boxes) * 4,])
    # convert number list to groups of 4, representing each bbox
    results = []
    for i in range(0, len(reshaped), 4):
        results.append((reshaped[i], reshaped[i + 1],
                    reshaped[i + 2], reshaped[i + 3]))    
    return results

# %%

def tf_index_select(input_, dim, indices):
    """
    input_(tensor): input tensor
    dim(int): dimension
    indices(list): selected indices list
    """
    shape = input_.get_shape().as_list()
    if dim == -1:
        dim = len(shape)-1
    shape[dim] = 1

    tmp = []
    for idx in indices:
        begin = [0]*len(shape)
        begin[dim] = idx
        tmp.append(tf.slice(input_, begin, shape))
    res = tf.concat(tmp, axis=dim)

    return res

def nms_tensorflow(P : tf.Tensor ,thresh_iou : float):
    """
    Apply non-maximum suppression to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the image 
            along with the class predscores, Shape: [num_boxes,5].
        thresh_iou: (float) The overlap thresh for suppressing unnecessary boxes.
    Returns:
        A list of filtered boxes, Shape: [ , 5]
    """

    # we extract coordinates for every 
    # prediction box present in P
    x1 = P[:, 0]
    y1 = P[:, 1]
    x2 = P[:, 2]
    y2 = P[:, 3]

    # we extract the confidence scores as well
    scores = P[:, 4]

    # calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)
    
    # sort the prediction boxes in P
    # according to their confidence scores
    order = tf.argsort(scores)

    # initialise an empty list for 
    # filtered prediction boxes
    keep = []
    

    while len(order) > 0:
        
        # extract the index of the 
        # prediction with highest score
        # we call this prediction S
        idx = order[-1]

        # push S in filtered predictions list
        keep.append(P[idx])

        # remove S from P
        order = order[:-1]

        # sanity check
        if len(order) == 0:
            break
        
        # select coordinates of BBoxes according to 
        # the indices in order
        xx1 = tf_index_select(x1,dim = 0, indices = order)
        xx2 = tf_index_select(x2,dim = 0, indices = order)
        yy1 = tf_index_select(y1,dim = 0, indices = order)
        yy2 = tf_index_select(y2,dim = 0, indices = order)

        # find the coordinates of the intersection boxes
        xx1 = tf.math.maximum(xx1, x1[idx])
        yy1 = tf.math.maximum(yy1, y1[idx])
        xx2 = tf.math.maximum(xx2, x2[idx])
        yy2 = tf.math.maximum(yy2, y2[idx])

        # find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1
        
        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = tf.clip_by_value(w, 0.0, np.inf)
        h = tf.clip_by_value(h, 0.0, np.inf)

        # find the intersection area
        inter = w*h

        # find the areas of BBoxes according the indices in order
        rem_areas = tf_index_select(areas, dim = 0, indices = order) 

        # find the union of every prediction T in P
        # with the prediction S
        # Note that areas[idx] represents area of S
        union = (rem_areas - inter) + areas[idx]
        # *********HEREING LIES THE BUG - the IoU should never be above 1, and yet it is. Something is wrong in either
        # the calculation of the areas, intersections, or unions
        
        # find the IoU of every prediction in P with S
        IoU = inter / union

        # keep the boxes with IoU less than thresh_iou
        mask = IoU < thresh_iou
        order = order[mask]
    
    return keep

def test():
    # Let P be the following
    P = tf.constant([
        [1, 1, 3, 3, 0.95],
        [1, 1, 3, 4, 0.93],
        [1, 0.9, 3.6, 3, 0.98],
        [1, 0.9, 3.5, 3, 0.97]
    ])
    filtered_boxes = nms_tensorflow(P,0.8)
    print(filtered_boxes)

# %%

test()

# %%
