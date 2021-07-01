import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from PIL import Image, ImageDraw as D
from operator import mul
import json
import os

file = open("writeData.json")

data = json.load(file) # data is a list of dictionaries
print (type(data))
for example in data: # example is the dictionary 
    print(example.items())
    print("\n")
    print("bbox: " , example['objects']['bbox'])
    print("\n")
