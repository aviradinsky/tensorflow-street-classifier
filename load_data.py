# %%
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.backend import convert_inputs_if_ragged
import tensorflow_datasets as tfds
import os
# %%
def crop_tensor_by_nth_bbox(tensor, nth_bbox):
    """
    crop a tensor in the dataset by the nth tensor
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

    if offset_height == 0 or offset_width == 0 or target_height <= 17 or target_width <= 17:
        return None
    else:
        return tf.image.crop_to_bounding_box(image,
                                                    offset_height=offset_height,
                                                    offset_width=offset_width,
                                                    target_height=target_height,
                                                    target_width=target_width
                                                    )
# %%
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
        bottom_right,
    ]
# %%
lines = open('misc/objects-label.labels.txt','r').readlines()
all_possible_labels = [line.strip() for line in lines]
def main(
    directory = f'{os.getcwd()}',
    chosen_labels_string = [
        'bicycle',
        'motorcycle',
        'bus',
        'truck',
        'car',
        'train',
        'person',
        'traffic light',
        'stop sign',
        'fire hydrant']):
    for label in chosen_labels_string:
        if label not in all_possible_labels:
            raise Exception(f'No such label as {label}')

    directory = f'{directory}/data'

    dont_need_to_continue = False

    for string in chosen_labels_string:
        if os.path.exists(f'{directory}/test/{string}/'):
            dont_need_to_continue = True
            break
        if os.path.exists(f'{directory}/train/{string}/'):
            dont_need_to_continue = True
            break

    if dont_need_to_continue:
        print('Directory structure for data, detected\nno longer creating data')
        return
    count_of_labels_dict = {}

    for string in chosen_labels_string:
        s = string.replace(' ', '_')
        count_of_labels_dict[s] = 0
        os.makedirs(f'{directory}/test/{s}/')
        os.makedirs(f'{directory}/train/{s}/')
    os.makedirs(f'{directory}/test/background/')
    os.makedirs(f'{directory}/train/background/')
    count_of_labels_dict['background'] = 0
    print(count_of_labels_dict)
    

    total_labels = len(chosen_labels_string) + 1
    """
    writing the chosen labels into an array with their number value
    """
    chosen_labels_int = []
    for label in chosen_labels_string:
        if label in all_possible_labels:
            chosen_labels_int.append(all_possible_labels.index(label))
    print(chosen_labels_int)
    all_data = tfds.load('coco')
    """
    all_data.keys() = 'test', 'test2015', 'train', 'validation'
    """
    valid_keys = ('train','validation')
    number_of_images_so_far = 0
    for key in valid_keys:
        for data in all_data.get(key):
            images_to_send = []
            image = data.get('image')
            number_form_labels = data.get('objects').get('label').numpy()

            for i,int_label in enumerate(number_form_labels):
                if int_label in chosen_labels_int:
                    images_to_send.append((crop_tensor_by_nth_bbox(data,i),all_possible_labels[int_label]))
            if len(images_to_send) == 0:
                for fourth in slice_into_4ths(image):
                    images_to_send.append((fourth,'background'))

            for send in images_to_send:
                image = send[0]
                if image is None:
                    continue
                label = send[1].replace(' ','_')
                if count_of_labels_dict[label] > 5000:
                    continue
                else:
                    if label == 'background':
                        count_of_labels_dict[label] += 0.25
                    else:
                        count_of_labels_dict[label] += 1
                if count_of_labels_dict[label] % 8 == 0:
                    label = f'test/{label}'
                else:
                    label = f'train/{label}'
                tf.keras.preprocessing.image.save_img(f'{directory}/{label}/{number_of_images_so_far}.jpg',image)
                number_of_images_so_far += 1
                if number_of_images_so_far % 1000 == 0:
                    print(f'{number_of_images_so_far = }')
# %%
main()
# %%
