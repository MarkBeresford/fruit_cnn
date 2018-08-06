import sys
from autoencoder_utils import *
import tensorflow as tf
import logging
from tqdm import *


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train_ae():
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, n_inputs])
    hidden = tf.layers.dense(x, n_hidden)
    outputs = tf.layers.dense(hidden, n_inputs)
    codings = outputs
    reconstruction_loss = tf.reduce_mean(tf.square(outputs - x))

    optimiser = tf.train.AdamOptimizer(learning_rate)
    training_op = optimiser.minimize(reconstruction_loss)

    init_op = tf.global_variables_initializer()

    # Get paths and labels
    training_file_paths, training_labels = get_paths(TRAINING_DATA_FILE_PATH)
    test_file_paths, test_labels = get_paths(TEST_DATA_FILE_PATH)

    # TODO: Save Weights using saverg
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        merged_summary = tf.summary.merge_all()
        if not os.path.isdir(SUMMERY_PATH):
            os.makedirs(SUMMERY_PATH)
        writer = tf.summary.FileWriter(SUMMERY_PATH)
        writer.add_graph(sess.graph)
        num_train_batches = int(len(training_labels) / BATCH_SIZE)
        logger.info('There are %s training batches.' % str(num_train_batches))
        train_pbar = tqdm(total=num_train_batches, position=1)
        for batch in range(num_train_batches):
            train_pbar.update(1)
            training_batch_file_paths, training_batch_labels = get_file_paths_and_labels(training_file_paths, training_labels, batch, BATCH_SIZE)
            training_batch_file_paths_reshaped = reshape_batch(training_batch_file_paths)
            sess.run(training_op, feed_dict={x: training_batch_file_paths_reshaped})

        num_test_batches = int(len(test_labels) / BATCH_SIZE)
        logger.info('There are %s training batches.' % str(num_test_batches))
        test_pbar = tqdm(total=num_test_batches, position=1)
        for batch in range(num_test_batches):
            test_pbar.update(1)
            test_batch_file_paths, test_batch_labels = get_file_paths_and_labels(test_file_paths,
                                                                                         test_labels, batch, BATCH_SIZE)

            test_batch_file_paths_reshaped = reshape_batch(test_batch_file_paths)
            coding_val = codings.eval(feed_dict={x: test_batch_file_paths_reshaped})
            if batch % (num_test_batches / 10) == 0:
                plot_image(coding_val[0], CODINGS_PATH, str(batch))
                plot_image(test_batch_file_paths_reshaped[0], IMAGES_PATH, str(batch))

if __name__ == "__main__":
    if sys.argv[1] == 'train':
        train_ae()
