import numpy as np
import os
from sklearn.utils import shuffle
from PIL import Image


classes = ['apple_braeburn', 'apple_golden_1', 'apple_golden_2', 'apple_golden_3', 'apple_granny_smith', 'apple_red_1',
           'apple_red_2', 'apple_red_3', 'apple_red_delicious', 'apple_red_yellow', 'apricot', 'avocado',
           'avocado_ripe', 'banana', 'banana_red', 'cactus_fruit', 'cantaloupe_1', 'cantaloupe_2', 'carambula',
           'cherry_1', 'cherry_2', 'cherry_rainier', 'cherry_wax_black', 'cherry_wax_red', 'cherry_wax_yellow',
           'clementine', 'cocos', 'dates', 'granadilla', 'grape_pink', 'grape_white', 'grape_white_2',
           'grapefruit_pink', 'grapefruit_white', 'guava', 'huckleberry', 'kaki', 'kiwi', 'kumquats', 'lemon',
           'lemon_meyer', 'limes', 'lychee', 'mandarine', 'mango', 'maracuja', 'melon_piel_de_sapo', 'mulberry',
           'nectarine', 'orange', 'papaya', 'passion_fruit', 'peach', 'peach_flat', 'pear', 'pear_abate',
           'pear_monster', 'pear_williams', 'pepino', 'physalis', 'physalis_with_husk', 'pineapple', 'pineapple_mini',
           'pitahaya_red', 'plum', 'pomegranate', 'quince', 'rambutan', 'raspberry', 'salak', 'strawberry',
           'strawberry_wedge', 'tamarillo', 'tangelo', 'tomato_1', 'tomato_2', 'tomato_3', 'tomato_4',
           'tomato_cherry_red', 'tomato_maroon', 'walnut']
NUM_LABELS = len(classes)
SRC_FOLDER_PATH = os.path.dirname(os.getcwd())
IMAGE_SIZE = 100
TRAINING_DATA_FILE_PATH = os.path.join(SRC_FOLDER_PATH, 'fruits-360/Training')
TEST_DATA_FILE_PATH = os.path.join(SRC_FOLDER_PATH, 'fruits-360/Test')


def get_darker_and_lighter_images(action, jpgfile, images_array):
    pixels = list(jpgfile.getdata())
    width, height = jpgfile.size

    brightness_multiplier = 1.0
    extent = 0.5

    if action is 'brighten':
        brightness_multiplier += extent
    else:
        brightness_multiplier -= extent

    modified_image = []
    for pixel in pixels:
        modified_pixel = [pixel_value * brightness_multiplier for pixel_value in pixel]
        modified_image.append(modified_pixel)

    modified_image = [modified_image[i * width:(i + 1) * width] for i in range(height)]
    images_array.append(modified_image)
    return images_array



def get_pixel_values(jpgfile):
    pixels = list(jpgfile.getdata())
    width, height = jpgfile.size
    pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
    return pixels


def get_jpg_paths(directory):
    file_paths = []
    for dirName, subdirList, fileList in os.walk(directory, topdown=False):
        for file in fileList:
            if ".jpg" in file.lower():  # check whether the file's a jpg image
                file_paths.append(os.path.join(dirName, file))
    return file_paths


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
    background.paste(source_image, (int((thumbnail_size[0] - w) / 2), int((thumbnail_size[1] - h) / 2)))
    return background


def get_paths(path):
    labels = []
    paths = []
    counter = 0
    previous_dir = ''
    for dirName, subdirList, fileList in os.walk(path, topdown=False):
        for file in fileList:
            labels_array_zeroes = np.zeros(len(classes))
            if previous_dir == '':
                previous_dir = dirName
            elif dirName != previous_dir:
                counter += 1
                previous_dir = dirName
            if ".jpg" in file.lower():  # check whether the file's a jpg image
                paths.append(os.path.join(dirName, file))
                np.put(labels_array_zeroes, counter, 1)
                labels.append(labels_array_zeroes)
    paths, labels = shuffle(paths, labels)
    return paths, labels


def get_pixels_from_file_paths(file_paths, training):
    for file_path in file_paths:
        jpgfile = Image.open(file_path)
        orig_pixels = get_pixel_values(jpgfile)
        images = [orig_pixels]
        if training:
            images = get_darker_and_lighter_images('brighten', jpgfile, images)
            images = get_darker_and_lighter_images('darken', jpgfile, images)
        images = np.asfarray(images)
    return images
