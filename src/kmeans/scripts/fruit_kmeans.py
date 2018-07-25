from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.decomposition import IncrementalPCA
from kmeans_utils import *
import sys
import pickle


def get_pca_pixels_and_labels(file_paths, labels, folder):
    print('Running Principle Component Analysis.')
    num_batches = int(len(file_paths) / BATCH_SIZE)
    inc_pca = IncrementalPCA(n_components=n_components)
    x_transform = np.empty(shape=(BATCH_SIZE, n_components))
    pixel_path = os.path.join(SRC_FOLDER_PATH, 'kmeans/tmp', folder)
    labels_path = os.path.join(SRC_FOLDER_PATH, 'kmeans/tmp', folder)
    if not os.path.isdir(pixel_path):
        os.mkdir(pixel_path)
    if not os.path.isdir(labels_path):
        os.mkdir(labels_path)

    print('There are %s Batches' % str(num_batches))

    for batch in range(num_batches):
        print('PCA,  BATCH NUM : {:}'.format(batch + 1))
        batch_file_paths, batch_labels = get_file_paths_and_labels(file_paths, labels, batch, BATCH_SIZE)
        batch_x = get_pixels_from_file_paths_kmeans(batch_file_paths, training=True)
        # PCA fitting
        inc_pca.partial_fit(batch_x)
        partial_x_transform = inc_pca.transform(batch_x)
        if batch == 0:
            x_transform = partial_x_transform
        else:
            x_transform = np.vstack((x_transform, partial_x_transform))
    print(inc_pca.explained_variance_ratio_)
    pixel_file = open(os.path.join(pixel_path, 'pixels2.p'), 'w')
    pickle.dump(x_transform, pixel_file)  # dump data to f
    pixel_file.close()
    labels_file = open(os.path.join(labels_path, 'labels2.p'), 'w')
    pickle.dump(labels, labels_file)  # dump data to f
    labels_file.close()
    print("PCA complete!")


def train_kmeans(pca_training_pix_path, pca_test_pix_path, pca_test_labels_path):
    clf = KMeans(n_clusters=NUM_LABELS, random_state=0)
    print('Reading in Data.')
    with open(pca_training_pix_path, 'rb') as pixels_file:
        train_pixels = pickle.load(pixels_file)
    pixels_file.close()
    with open(pca_test_pix_path, 'rb') as pixels_file:
        test_pixels = pickle.load(pixels_file)
    pixels_file.close()
    with open(pca_test_labels_path, 'rb') as pixels_file:
        test_labels_pickled = pickle.load(pixels_file)
    pixels_file.close()
    print("Starting Training...")
    clf.fit(train_pixels)
    print("Training complete!")
    total_test_batches = int(len(test_labels_pickled) / BATCH_SIZE)
    print('There are {:} test batches.'.format(total_test_batches))
    batch_accuracys = []
    for batch in range(total_test_batches):
        predictions = clf.predict(test_pixels)
        batch_accuracy = accuracy_score(test_labels_pickled[:len(predictions)], predictions)
        print('Batch Accuracy Score {:}'.format(batch_accuracy))
        batch_accuracys.append(batch_accuracy)
    av_test_acc = sum(batch_accuracys) / len(batch_accuracys)
    print("Test set accuracy: {:.3f}".format(av_test_acc))


if __name__ == "__main__":
    if sys.argv[1] == 'pca':
        training_file_paths, training_labels = get_paths(TRAINING_DATA_FILE_PATH)
        get_pca_pixels_and_labels(training_file_paths, training_labels, 'train')
        test_file_paths, test_labels = get_paths(TEST_DATA_FILE_PATH)
        get_pca_pixels_and_labels(test_file_paths, test_labels, 'test')
    if sys.argv[1] == 'train':
        train_pixel_path = os.path.join(SRC_FOLDER_PATH, 'kmeans/tmp/train/pixels.p')
        test_pixel_path = os.path.join(SRC_FOLDER_PATH, 'kmeans/tmp/test/pixels.p')
        test_labels_path = os.path.join(SRC_FOLDER_PATH, 'kmeans/tmp/test/labels.p')
        train_kmeans(train_pixel_path, test_pixel_path, test_labels_path)
