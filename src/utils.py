import math
from random import shuffle
from PIL import Image
import numpy as np
import os


classes = ['apple_braeburn', 'apple_golden_1', 'apple_golden_2', 'apple_golden_3', 'apple_granny_smith', 'apple_red_1', 'apple_red_3', 'apple_red_delicious', 'apple_red_yellow', 'apricot', 'avocado', 'avocado_ripe', 'banana', 'banana_red', 'cactus_fruit', 'cantaloupe_1', 'cantaloupe_2', 'carambula', 'cherry_1', 'cherry_2', 'cherry_rainier', 'clementine', 'cocos', 'dates', 'granadilla', 'grape_pink', 'grape_white', 'grape_white_2', 'grapefruit_pink', 'grapefruit_white', 'guava', 'huckleberry', 'kaki', 'kiwi', 'kumquats', 'lemon', 'lemon_meyer', 'limes', 'litchi', 'mandarine', 'mango', 'maracuja', 'nectarine', 'orange', 'papaya', 'passion_fruit', 'peach', 'peach_flat', 'pear', 'pear_abate', 'pear_monster', 'pear_williams', 'pepino', 'pineapple', 'pitahaya_red', 'plum', 'pomegranate', 'quince', 'raspberry', 'salak', 'strawberry', 'apple_red_2', 'tamarillo', 'tangelo']
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
TRAINING_DATA_FILE_PATH = os.path.join(PROJECT_ROOT, 'fruits-360/Training')
VALIDATION_DATA_FILE_PATH = os.path.join(PROJECT_ROOT, 'fruits-360/Validation')
SESSION_NUM = '0'
BATCH_SIZE = 60
IMAGE_SIZE = 100
NUM_LABELS = len(classes)
learning_rate = 0.0001
NUM_CHANNELS = 3
SEED = 42
NUM_RUN_IMAGES = 1


def get_pixel_values(path):
    jpgfile = Image.open(path)
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


def get_validation_paths(path):
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


def get_test_train_paths(path):
    training_file_paths = []
    test_file_paths = []
    training_labels = []
    test_labels = []
    dir_file_paths = []
    dir_labels = []
    counter = 0
    previous_dir = ''
    for dirName, subdirList, fileList in os.walk(path, topdown=False):
        for file in fileList:
            labels_array = np.zeros(len(classes))
            if previous_dir == '':
                previous_dir = dirName
            elif dirName != previous_dir:
                num_train = int(math.ceil(len(dir_file_paths) * 0.8))
                num_test = int(math.floor(len(dir_file_paths) * 0.2))
                training_file_paths.append(dir_file_paths[:num_train])
                test_file_paths.append(dir_file_paths[:num_test])
                test_labels.append(dir_labels[:num_test])
                training_labels.append(dir_labels[:num_train])
                counter += 1
                previous_dir = dirName
                dir_file_paths = []
                dir_labels = []
            if ".jpg" in file.lower():  # check whether the file's a jpg image
                dir_file_paths.append(os.path.join(dirName, file))
                np.put(labels_array, counter, 1)
                dir_labels.append(labels_array)
    training_labels = [item for sublist in training_labels for item in sublist]
    test_file_paths = [item for sublist in test_file_paths for item in sublist]
    training_file_paths = [item for sublist in training_file_paths for item in sublist]
    test_labels = [item for sublist in test_labels for item in sublist]
    return training_file_paths, training_labels, test_file_paths, test_labels


def get_pixs_from_file_paths(file_paths):
    training_data_bytearray = []
    for file_path in file_paths:
        img_array = []
        for row in get_pixel_values(file_path):
            row_array = []
            for pixel_set in row:
                pixel_set_floats = []
                for pixel in pixel_set:
                    pixel_set_floats.append(float(pixel))
                row_array.append(pixel_set_floats)
            img_array.append(row_array)
        training_data_bytearray.append(img_array)
    return training_data_bytearray


def shuffle_list(*ls):
    l = list(zip(*ls))
    shuffle(l)
    return zip(*l)


def get_pixels_and_labels(file_paths, labels, batch, BATCH_SIZE):
    first_instance = batch * BATCH_SIZE
    batch_file_paths = file_paths[first_instance:first_instance + BATCH_SIZE]
    batch_labels = labels[first_instance:first_instance + BATCH_SIZE]
    return batch_file_paths, batch_labels


def black_background_thumbnail(path_to_image, thumbnail_size=(IMAGE_SIZE, IMAGE_SIZE)):
    background = Image.new('RGBA', thumbnail_size, "black")
    source_image = Image.open(path_to_image).convert("RGBA")
    source_image.thumbnail(thumbnail_size)
    (w, h) = source_image.size
    background.paste(source_image, ((thumbnail_size[0] - w) / 2, (thumbnail_size[1] - h) / 2 ))
    return background
