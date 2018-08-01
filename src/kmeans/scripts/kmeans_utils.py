import os
import numpy as np
from PIL import Image
from itertools import izip
from pathlib import Path


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
TRAINING_DATA_FILE_PATH = os.path.join(SRC_FOLDER_PATH, 'fruits-360/Training')
TEST_DATA_FILE_PATH = os.path.join(SRC_FOLDER_PATH, 'fruits-360/Test')
n_components = 100
BATCH_SIZE = n_components + 1


def get_pixel_values(jpgfile):
    pixels = list(jpgfile.getdata())
    width, height = jpgfile.size
    pixels = [pixels[i * width:(i + 1) * width] for i in xrange(height)]
    return pixels


def get_pixels_from_file_paths_kmeans(file_paths, training):
    training_data_bytearray = []
    for file_path in file_paths:
        jpgfile = Image.open(file_path)
        images = []
        orig_pixels = get_pixel_values(jpgfile)
        orig_pixels = np.asarray(orig_pixels)
        orig_pixels = orig_pixels.ravel()
        images.append(orig_pixels)
        if training:
            brighter_pixels = get_darker_and_lighter_images('brighten', orig_pixels)
            brighter_pixels = np.asarray(brighter_pixels)
            brighter_pixels = brighter_pixels.ravel()
            images.append(brighter_pixels)
            darken_pixels = get_darker_and_lighter_images('darken', orig_pixels)
            darken_pixels = np.asarray(darken_pixels)
            darken_pixels = darken_pixels.ravel()
            images.append(darken_pixels)
        for image in images:
            averaged_image = []
            for pixel_1, pixel_2, pixel_3 in izip(*[iter(image)] * 3):
                averaged_image.append(float((pixel_1 + pixel_2 + pixel_3) / 3))
            training_data_bytearray.append(averaged_image)
    return training_data_bytearray


def get_darker_and_lighter_images(action, pixels):
    new_image_list = []

    brightness_multiplier = 1.0
    extent = 0.5

    if action is 'brighten':
        brightness_multiplier += extent
    else:
        brightness_multiplier -= extent
    for pixel in pixels:
        new_pixel = pixel * brightness_multiplier
        new_image_list.append(new_pixel)
    return new_image_list


def get_paths(path):
    training_labels = []
    file_paths = []
    counter = 0
    previous_dir = ''
    for dirName, subdirList, fileList in os.walk(path, topdown=False):
        for file in fileList:
            # labels_array = np.zeros(len(classes))
            if previous_dir == '':
                previous_dir = dirName
            elif dirName != previous_dir:
                counter += 1
                previous_dir = dirName
            if ".jpg" in file.lower():  # check whether the file's a jpg image
                file_paths.append(os.path.join(dirName, file))
                training_labels.append(counter)
    return file_paths, training_labels


def get_file_paths_and_labels(file_paths, labels, batch, BATCH_SIZE):
    first_instance = batch * BATCH_SIZE
    batch_file_paths = file_paths[first_instance:first_instance + BATCH_SIZE]
    batch_labels = labels[first_instance:first_instance + BATCH_SIZE]
    return batch_file_paths, batch_labels
