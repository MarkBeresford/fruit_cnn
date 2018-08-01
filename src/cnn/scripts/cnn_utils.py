import numpy as np
import os
from pathlib import Path
from PIL import Image
from random import shuffle


classes = ['apple_braeburn', 'apple_golden_1', 'apple_golden_2', 'apple_golden_3', 'apple_granny_smith', 'apple_red_1',
           'apple_red_2', 'apple_red_3', 'apple_red_delicious', 'apple_red_yellow', 'apricot', 'avocado',
           'avocado_ripe', 'banana', 'banana_red', 'cactus_fruit', 'cantaloupe_1', 'cantaloupe_2', 'carambula',
           'cherry_1', 'cherry_2', 'cherry_rainier', 'cherry_wax_black', 'cherry_wax_red', 'cherry_wax_yellow',
           'clementine', 'cocos', 'dates', 'granadilla', 'grape_pink', 'grape_white', 'grape_white_2', 'grape_white_2',
           'grapefruit_pink', 'grapefruit_white', 'guava', 'huckleberry', 'kaki', 'kiwi', 'kumquats', 'lemon',
           'lemon_meyer', 'limes', 'lychee', 'mandarine', 'mango', 'maracuja', 'melon_piel_de_sapo', 'mulberry',
           'nectarine', 'orange', 'papaya', 'passion_fruit', 'peach', 'peach_flat', 'pear', 'pear_abate',
           'pear_monster', 'pear_williams', 'pepino', 'physalis', 'physalis_with_husk', 'pineapple', 'pineapple_mini',
           'pitahaya_red', 'plum', 'pomegranate', 'quince', 'rambutan', 'raspberry', 'salak', 'strawberry',
           'strawberry_wedge', 'tamarillo', 'tangelo', 'walnut']
NUM_LABELS = len(classes)
SRC_FOLDER_PATH = str(Path(__file__).parents[2])
NUM_CHANNELS = 3
BATCH_SIZE = 20
IMAGE_SIZE = 100
SESSION_NUM = '0'
dropout_rate = 0.5
learning_rate = 0.0001
TRAINING_DATA_FILE_PATH = os.path.join(SRC_FOLDER_PATH, 'fruits-360/Training')
TEST_DATA_FILE_PATH = os.path.join(SRC_FOLDER_PATH, 'fruits-360/Test')
CHECKPOINT_PATH = os.path.join(SRC_FOLDER_PATH, 'cnn/model_checkpoint', SESSION_NUM)
SUMMERY_PATH = os.path.join(SRC_FOLDER_PATH, 'cnn/summary', SESSION_NUM)


def get_pixels_from_file_paths_cnn(file_paths, training):
    training_data_bytearray = []
    for file_path in file_paths:
        jpgfile = Image.open(file_path)
        images = []
        orig_pixels = get_pixel_values(jpgfile)
        images.append(orig_pixels)
        if training:
            brighter_pixels = get_darker_and_lighter_images('brighten', jpgfile)
            images.append(brighter_pixels)
            darken_pixels = get_darker_and_lighter_images('darken', jpgfile)
            images.append(darken_pixels)
        for image in images:
            img_array = []
            for row in image:
                row_array = []
                for pixel_set in row:
                    pixel_set_floats = []
                    for pixel in pixel_set:
                        pixel_set_floats.append(float(pixel))
                    row_array.append(pixel_set_floats)
                img_array.append(row_array)
            training_data_bytearray.append(img_array)
    return training_data_bytearray


def get_darker_and_lighter_images(action, jpgfile):
    pixels = list(jpgfile.getdata())
    width, height = jpgfile.size
    new_image_list = []

    brightness_multiplier = 1.0
    extent = 0.5

    if action is 'brighten':
        brightness_multiplier += extent
    else:
        brightness_multiplier -= extent

    # for each pixel, append the brightened or darkened version to the new image list
    for pixel in pixels:
        new_pixel = (int(pixel[0] * brightness_multiplier),
                     int(pixel[1] * brightness_multiplier),
                     int(pixel[2] * brightness_multiplier))

        new_image_list.append(new_pixel)
    new_image_list = [new_image_list[i * width:(i + 1) * width] for i in xrange(height)]
    return new_image_list


def get_pixel_values(jpgfile):
    pixels = list(jpgfile.getdata())
    width, height = jpgfile.size
    pixels = [pixels[i * width:(i + 1) * width] for i in xrange(height)]
    return pixels


def get_jpg_paths(directory):
    file_paths = []
    for dirName, subdirList, fileList in os.walk(directory, topdown=False):
        for file in fileList:
            if ".jpg" in file.lower():  # check whether the file's a jpg image
                file_paths.append(os.path.join(dirName, file))
    return file_paths


def shuffle_list(*ls):
    l = list(zip(*ls))
    shuffle(l)
    return zip(*l)


def get_file_paths_and_labels(file_paths, labels, batch, BATCH_SIZE):
    first_instance = batch * BATCH_SIZE
    batch_file_paths = file_paths[first_instance:first_instance + BATCH_SIZE]
    batch_labels = labels[first_instance:first_instance + BATCH_SIZE]
    return batch_file_paths, batch_labels


def black_background_thumbnail(path_to_image, thumbnail_size=(IMAGE_SIZE, IMAGE_SIZE)):
    background = Image.new('RGBA', thumbnail_size, "black")
    source_image = Image.open(path_to_image).convert("RGBA")
    source_image.thumbnail(thumbnail_size)
    (w, h) = source_image.size
    background.paste(source_image, ((thumbnail_size[0] - w) / 2, (thumbnail_size[1] - h) / 2))
    return background


def get_paths(path):
    training_labels = []
    file_paths = []
    counter = 0
    previous_dir = ''
    for dirName, subdirList, fileList in os.walk(path, topdown=False):
        for file in fileList:
            labels_array = np.zeros(len(classes))
            if previous_dir == '':
                previous_dir = dirName
            elif dirName != previous_dir:
                counter += 1
                previous_dir = dirName
            if ".jpg" in file.lower():  # check whether the file's a jpg image
                file_paths.append(os.path.join(dirName, file))
                np.put(labels_array, counter, 1)
                training_labels.append(labels_array)
    return file_paths, training_labels
