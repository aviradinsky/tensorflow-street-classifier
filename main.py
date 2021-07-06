# %%
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
# %%
# loading the data
chosen_labels = [
    1, # bike
    3, # motorcycle
    5, # bus
    7, # truck
    2, # car
    6, # train
    0, # person
    9, # traffic light
    11, # stop sign
    10 # fire hydrant
]
data = tfds.load('coco').get('train')
# print(type(data))  <class 'tensorflow.python.data.ops.dataset_ops.PrefetchDataset'>
# %%
# this is just a holder until i use all of the data
i = 1
for sample in data:
    if i > 5:
        break
    else:
        i += 1
    #print(type(sample)) dict
    #print(sample.keys()) IMAGE, IMAGE/FILENAME, IMAGE/ID, OBJECTS
    image = sample.get('image')
    objects = sample.get('objects')
    # print(objects.keys()) 'area', 'bbox', 'id', 'is_crowd', 'label'
    all_labels = objects.get('label').numpy()
    #plt.figure()
    #plt.imshow(image)
    #plt.show()
    for j, label in enumerate(all_labels):
        if label in chosen_labels:
            bbox = objects.get('bbox').numpy()[j]
            top_point = bbox[0]*image.shape[0]
            left_point = bbox[1]*image.shape[1]
            bottom_point= bbox[2]*image.shape[0]
            right_point= bbox[3]*image.shape[1]

            #print(f'{bbox[3]*image.shape[1]}')
            cropped_image = tf.image.crop_to_bounding_box(image,
                offset_height=int(top_point),
                offset_width=int(left_point),
                target_height=int(bottom_point - top_point),
                target_width=int(right_point - left_point)
            )
            plt.figure()
            plt.imshow(cropped_image)
            plt.show()
            pass

# %%

# %%
