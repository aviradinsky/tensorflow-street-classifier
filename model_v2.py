# %%
import tensorflow as tf
import tensorflow_datasets as tfds
import os
# %%
"""
global variables
"""
directory = f'{os.getcwd()}/data/'
chosen_labels = [
    1,  # bike
    3,  # motorcycle
    5,  # bus
    7,  # truck
    2,  # car
    6,  # train
    0,  # person
    9,  # traffic light
    11,  # stop sign
    10  # fire hydrant
]
# %%

def crop_tensor_by_nth_bbox(tensor, nth_bbox):
    """
    given an image with bboxs this method returns the cropped image
    for the nth bbox
    """


    bbox = tensor.get('objects').get('bbox').numpy()[nth_bbox]
    image = tensor.get('image')

    top_line = bbox[0]*image.shape[0]
    left_line = bbox[1]*image.shape[1]
    bottom_line = bbox[2]*image.shape[0]
    right_line = bbox[3]*image.shape[1]

    offset_height = int(top_line)
    offset_width = int(left_line)
    target_height = int(bottom_line - top_line)
    target_width = int(right_line - left_line)

    """
    if target_height == 0 or target_width == 0 or top_line == 0 or right_line == 0:
        print('garbage is going in')
    """
    return tf.image.crop_to_bounding_box(image,
                                                    offset_height=offset_height,
                                                    offset_width=offset_width,
                                                    target_height=target_height,
                                                    target_width=target_width
                                                    )
    pass

def slice_into_4ths(image) -> list:
    x,y,z = image.shape

    x,y = int(x/2),int(y/2)

    top_left = tf.slice(image, [0, 0, 0], [x, y, 3])
    bottom_left = tf.slice(image, [x, 0, 0], [x, y, 3])
    top_right = tf.slice(image, [0, y, 0], [x, y, 3])
    bottom_right = tf.slice(image, [x, y, 0], [x, y, 3])

    return [
        top_right,
        top_left,
        bottom_left,
        bottom_right
    ]



def get_image_manipulations_and_path(image,count) -> list:
    tensor = image
    objects = image.get('objects')
    all_labels = objects.get('label').numpy()
    image = image.get('image') 
    ret = []
    path_buffer: str= ''
    if len(set(all_labels) & set(chosen_labels)) == 0:
        """
        this is background and will be randomly cropped to increase number
        """
        images = slice_into_4ths(image)
        for i in images:
            if count % 8 == 0:
                path = f'{directory}/test/10/{count}.jpeg'
            else:
                path = f'{directory}/10/{count}.jpeg'
            count += 1
            ret.append((i,path))
    else:
        """
        this is images that have objects we want, they will be cropped
        """
        for ind, label in enumerate(all_labels):
            if label in chosen_labels:
                image = crop_tensor_by_nth_bbox(tensor,ind)
                if count % 8 == 0:
                    path = f'{directory}/test/{chosen_labels.index(label)}/{count}.jpeg'
                else:
                    path = f'{directory}/{chosen_labels.index(label)}/{count}.jpeg'
                count += 1
                ret.append((image,path))
    return (ret,count)


def set_data_in_directories() -> None:
    """
    checking to see if the path ('data/' and 'data/test') exists, if it does no code will run
    """

    if os.path.exists(f'{directory}/0'):
        # this means that the data was already loaded
        print(f'Data is already in the data folders')
        return
    else:
        for i in range(len(chosen_labels) + 1):
            os.makedirs(f'{directory}/{i}')
            os.makedirs(f'{directory}/test/{i}')
    
    """ 
    there are 4 keys but test and test2015 dont have labels
    """


    valid_keys = ('train','validation')
    count = 0
    for key in valid_keys:
        for image in tfds.load('coco',shuffle_files=True).get(key):
            data = get_image_manipulations_and_path(image,count)

            for manipulation, path in data[0]:
                tf.keras.preprocessing.image.save_img(path, manipulation)
                count = data[1] 
    pass
# %%
image = tfds.load('coco').get('train').take(1)
import matplotlib.pyplot as plt
for j in image:
    j = j.get('image')
    for i in slice_into_4ths(j):
        plt.figure()
        plt.imshow(i)
        plt.show()
# %%

set_data_in_directories()

# %%
