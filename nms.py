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

def nms(bboxes: list, scores: list, max_output_size=10, iou_threshold=.25):
    """Runs tensorflow's non max suppression implentation on the
    given bboxes.

    Args:
        bboxes (list): list of bboxes with coordinates in order of:
                                            (left, top, right, bottom)
        scores (list): list of prediction probabilities for each bbox
        max_output_size (int, optional): max output size. Defaults to 1.
    
    Returns:
        indices: a list of indices of bboxes chosen by the nms algorithm
    """
    # reverse x and y coordinates to align with tensorflow input reqs
    reverse = [(b, a, d, c) for (a, b, c, d) in bboxes]
    # convert to tensor
    bbox_array = np.array(reverse)
    boxes_tensor = tf.constant(bbox_array, dtype='float32')
    scores_tensor = tf.constant(scores, dtype='float32')
    # get results from nms as list of numbers
    selected_indices = tfnms(boxes_tensor, scores_tensor, 
                             tf.constant(max_output_size), 
                             iou_threshold=iou_threshold)
    return selected_indices.numpy()

# %%
