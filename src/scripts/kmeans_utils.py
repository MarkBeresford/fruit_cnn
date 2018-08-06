from common_utils import *
from PIL import Image
from itertools import izip
from sklearn.preprocessing import normalize


n_components = 100
BATCH_SIZE = n_components + 1


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
    training_data_bytearray = normalize(training_data_bytearray)
    return training_data_bytearray


def get_paths_kmeans(path):
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
