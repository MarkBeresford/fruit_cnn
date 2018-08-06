from common_utils import *
import matplotlib.pyplot as plt

n_inputs = 10000
n_hidden = 100
n_outputs = n_inputs
learning_rate = 0.01
BATCH_SIZE = 10
SESSION_NUM = '0'
CHECKPOINT_PATH = os.path.join(SRC_FOLDER_PATH, 'autoencoder/model_checkpoint', SESSION_NUM)
SUMMERY_PATH = os.path.join(SRC_FOLDER_PATH, 'autoencoder/summary', SESSION_NUM)
CODINGS_PATH = os.path.join(SRC_FOLDER_PATH, 'autoencoder/codings')
IMAGES_PATH = os.path.join(SRC_FOLDER_PATH, 'autoencoder/images')


def reshape_batch(batch_paths):
    reshaped_batch = []
    for file_path in batch_paths:
        jpgfile = Image.open(file_path)
        orig_pixels = get_pixel_values(jpgfile)
        orig_pixels_np = np.asarray(orig_pixels)
        averaged_pixs = np.mean(orig_pixels_np, axis=2, dtype=int)
        reshaped_pixels = np.reshape(averaged_pixs, 10000)
        reshaped_batch.append(reshaped_pixels)
    reshaped_batch_np = np.asarray(reshaped_batch)
    return reshaped_batch_np


def plot_image(image, path, image_num, shape=[100, 100]):
    fig = plt.figure()
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")
    if not os.path.isdir(path):
        os.makedirs(path)
    fig.savefig(os.path.join(path, image_num))


