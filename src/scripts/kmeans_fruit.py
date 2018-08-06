from kmeans_utils import *
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.decomposition import IncrementalPCA
from tqdm import *
import logging
import math
import pickle
import sys


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_pca_pixels_and_labels(file_paths, labels, folder):
    logger.info('Running Principle Component Analysis.')

    num_files = len(file_paths)
    num_batches = int(math.ceil(num_files / float(BATCH_SIZE)))
    inc_pca = IncrementalPCA(n_components=n_components)
    x_transform = np.empty(shape=(num_files, n_components))
    pixel_path = os.path.join(SRC_FOLDER_PATH, 'kmeans/tmp', folder)
    labels_path = os.path.join(SRC_FOLDER_PATH, 'kmeans/tmp', folder)
    if not os.path.isdir(pixel_path):
        os.makedirs(pixel_path)
    if not os.path.isdir(labels_path):
        os.makedirs(labels_path)

    logger.info('There are %s Batches.' % str(num_batches))
    # The times three is to account for the darker and lighter images produced for each image
    logger.info('There are %s examples in this PCA.' % str(len(file_paths) * 3))

    pbar = tqdm(total=num_batches, position=1)
    for batch in range(num_batches):
        pbar.update(1)
        batch_file_paths, batch_labels = get_file_paths_and_labels(file_paths, labels, batch, BATCH_SIZE)
        batch_x = get_pixels_from_file_paths_kmeans(batch_file_paths, training=True)
        # PCA fitting
        inc_pca.partial_fit(batch_x)
        partial_x_transform = inc_pca.transform(batch_x)
        if batch == 0:
            x_transform = partial_x_transform
        else:
            x_transform = np.vstack((x_transform, partial_x_transform))
    logger.info('PCA saved: %s instances.' % len(x_transform))

    logger.info('explained_variance_')
    logger.info(inc_pca.explained_variance_)
    logger.info('explained_variance_ratio_')
    logger.info(inc_pca.explained_variance_ratio_)

    pixel_file = open(os.path.join(pixel_path, 'pixels.p'), 'w')
    pickle.dump(x_transform, pixel_file)  # dump data to f
    pixel_file.close()
    labels_file = open(os.path.join(labels_path, 'labels.p'), 'w')
    pickle.dump(labels, labels_file)  # dump data to f
    labels_file.close()
    logger.info("PCA complete!")


def train_kmeans(pca_training_pix_path, pca_test_pix_path, pca_test_labels_path):
    clf = KMeans(n_clusters=NUM_LABELS, random_state=0)
    logger.info('Reading in Data.')
    with open(pca_training_pix_path, 'rb') as pixels_file:
        train_pixels = pickle.load(pixels_file)
    pixels_file.close()
    with open(pca_test_pix_path, 'rb') as pixels_file:
        test_pixels = pickle.load(pixels_file)
    pixels_file.close()
    with open(pca_test_labels_path, 'rb') as labels_file:
        test_labels_pickled = pickle.load(labels_file)
    labels_file.close()
    # Extend labeled data to include bright and light images
    extended_test_labels_pickled = []
    for label in test_labels_pickled:
        extended_test_labels_pickled += [label] * 3
    logger.info("Starting Training...")
    clf.fit(train_pixels)
    logger.info("Training complete!")
    logger.info("Predicting...")
    predictions = clf.predict(test_pixels)
    batch_accuracy = accuracy_score(extended_test_labels_pickled, predictions)
    logger.info("Test set accuracy: {:.3f}".format(batch_accuracy))


if __name__ == "__main__":
    train_folder = 'train_%s_comps' % n_components
    test_folder = 'test_%s_comps' % n_components
    if sys.argv[1] == 'pca':
        # Training PCA
        training_file_paths, training_labels = get_paths_kmeans(TRAINING_DATA_FILE_PATH)
        get_pca_pixels_and_labels(training_file_paths, training_labels, train_folder)
        # Test PCA
        test_file_paths, test_labels = get_paths_kmeans(TEST_DATA_FILE_PATH)
        get_pca_pixels_and_labels(test_file_paths, test_labels, test_folder)
    if sys.argv[1] == 'train':
        train_pixel_path = os.path.join(SRC_FOLDER_PATH, 'kmeans/tmp', train_folder, 'pixels.p')
        test_pixel_path = os.path.join(SRC_FOLDER_PATH, 'kmeans/tmp', test_folder, 'pixels.p')
        test_labels_path = os.path.join(SRC_FOLDER_PATH, 'kmeans/tmp', test_folder, 'labels.p')
        train_kmeans(train_pixel_path, test_pixel_path, test_labels_path)
