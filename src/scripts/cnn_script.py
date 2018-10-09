import logging
import tensorflow as tf
import sys
from cnn_utils import *
from tqdm import *



def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name="conv"):
    with tf.name_scope(name):
        conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]
        #   tf.truncated_normal -> Normal distrubtion
        weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name + 'W')
        bias = tf.Variable(tf.truncated_normal([num_filters]), name=name + 'b')
        #   tf.nn.conv2d erforms converlution between the input_data and the weights
        out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')
        out_layer += bias
        #   apply a ReLU non-linear activation
        out_layer = tf.nn.relu(out_layer)
        #   ksize is the pooling filter size
        ksize = [1, pool_shape[0], pool_shape[1], 1]
        strides = [1, 2, 2, 1]
        #   padding='SAME' means after pooling the tensors will be the same size
        out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("bias", bias)
        tf.summary.histogram("activation", out_layer)
        return out_layer


def create_new_dense_layer(input_data, input_size, output_size, name="dense"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.03), name='W')
        b = tf.Variable(tf.truncated_normal([output_size], stddev=0.01), name='b')
        dense_layer = tf.matmul(input_data, w) + b
        return dense_layer


def train_cnn():
    training = tf.placeholder_with_default(True, shape=(), name='training')
    x = tf.placeholder(tf.float32, [None, 100, 100, 3], name='x')
    x_shaped = tf.reshape(x, [-1, 100, 100, 3], name='x_reshaped')
    y = tf.placeholder(tf.float32, [None, NUM_LABELS], name='labels')

    # create both convolutional layers
    layer1 = create_new_conv_layer(x_shaped, NUM_CHANNELS, 32, [5, 5], [8, 8], name='l1')
    layer2 = create_new_conv_layer(layer1, 32, 64, [5, 5], [8, 8], name='l2')

    # 25 * 25 because the pool layer will produce a 25 * 25 matrix
    flattened = tf.reshape(layer2, [-1, 25 * 25 * 64], name='flatterned')
    flattened_dropped = tf.layers.dropout(flattened, dropout_rate, training=training)

    # setup some weights and bias values for this layer, then activate with ReLU
    # 1000 neurons in hidden layers
    dense_layer1 = create_new_dense_layer(flattened_dropped, 25 * 25 * 64, 1000, name='dl1')
    dense_layer1_dropped = tf.layers.dropout(dense_layer1, dropout_rate, training=training)
    dense_layer1_rnn = tf.nn.relu(dense_layer1_dropped, name='relu')

    # another layer with softmax activations
    dense_layer2 = create_new_dense_layer(dense_layer1_rnn, 1000, NUM_LABELS, name='dl2')
    y_ = tf.nn.softmax(dense_layer2, name='predictions')

    with tf.name_scope('xent'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y),
                                       name='xent')
    tf.summary.scalar("cross_entropy", cross_entropy)

    # Get training file paths and labels
    training_file_paths, training_labels = get_paths(TRAINING_DATA_FILE_PATH)

    with tf.name_scope('train'):
        optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1), name='x_pred')
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    tf.summary.scalar("accuracy", accuracy)

    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        merged_summary = tf.summary.merge_all()
        if not os.path.isdir(SUMMERY_PATH):
            os.makedirs(SUMMERY_PATH)
        writer = tf.summary.FileWriter(SUMMERY_PATH)
        writer.add_graph(sess.graph)

        total_train_batches = int(len(training_labels) / BATCH_SIZE)
        logger.info('There are %s training batches.'% str(total_train_batches))
        pbar = tqdm(total=total_train_batches, position=1)
        for batch in range(total_train_batches):
            pbar.update(1)
            batch_file_paths, batch_labels = get_file_paths_and_labels(training_file_paths, training_labels, batch, BATCH_SIZE)
            # This will generate three sets of pixels, one for the original image, one for the 'darker' image and one
            # for the 'lighter' image.
            batch_x = get_pixels_from_file_paths(batch_file_paths, training=True)
            multi_batch_labels = []
            # This is nessasary to replicate the labels for the 'darker' and 'lighter' images
            for label in batch_labels:
                multi_batch_labels.extend([label] * 3)
            batch_y = np.asarray(multi_batch_labels)
            s = sess.run(merged_summary, feed_dict={x: batch_x, y: batch_y})
            writer.add_summary(s, batch)
            _, cost = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
        if not os.path.isdir(CHECKPOINT_PATH):
            os.makedirs(CHECKPOINT_PATH)

        saver.save(sess, os.path.join(CHECKPOINT_PATH, "fruit_ml"))
        logger.info("Training complete!")


def restore_cnn():
    logger.info('Restoring Model.')
    meta_data_file = os.path.join(CHECKPOINT_PATH, 'fruit_ml.meta')
    checkpoint = tf.train.import_meta_graph(meta_data_file)
    logger.info('Model Restored.')
    return checkpoint


def test_cnn():
    test_file_paths, test_labels = get_paths(TEST_DATA_FILE_PATH)
    total_test_batches = int(len(test_labels) / BATCH_SIZE)

    test_accs = []
    with tf.Session() as sess:
        cnn_checkpoint = restore_cnn()
        cnn_checkpoint.restore(sess, tf.train.latest_checkpoint(CHECKPOINT_PATH))
        logger.info('There are {} test batches.'.format(str(total_test_batches)))
        pbar = tqdm(total=total_test_batches, position=1)
        for batch in range(total_test_batches):
            pbar.update(1)
            batch_file_paths, batch_labels = get_file_paths_and_labels(test_file_paths, test_labels, batch, BATCH_SIZE)
            batch_x = get_pixels_from_file_paths(batch_file_paths, training=False)
            batch_y = np.asarray(batch_labels)
            test_acc = sess.run("accuracy/accuracy:0",
                                feed_dict={"x:0": batch_x, "labels:0": batch_y, 'training:0': False})
            test_accs.append(test_acc)
        av_test_acc = sum(test_accs) / len(test_accs)
    logger.info('\n')
    logger.info('Test set accuracy: {}%'.format(str(av_test_acc * 100)))


def generate_predictions_using_cnn(image_directory):
    with tf.Session() as sess:
        cnn_checkpoint = restore_cnn()
        cnn_checkpoint.restore(sess, tf.train.latest_checkpoint(CHECKPOINT_PATH))
        file_paths = get_jpg_paths(image_directory)
        resized_file_paths = []
        resized_directory = os.path.join(image_directory, 'resized_files')
        if not os.path.isdir(resized_directory):
            os.makedirs(resized_directory)

        # Makes images have 100x100 pixels
        for original_file_path in file_paths:
            resized_file_path = os.path.join(resized_directory, original_file_path.split('/')[-1])
            try:
                im = black_background_thumbnail(original_file_path)
                rgb_im = im.convert('RGB')
                rgb_im.save(resized_file_path)
                resized_file_paths.append(resized_file_path)
            except IOError:
                logger.error("Failed to convert image : '%s'" % original_file_path)
        pixels_single_array = get_pixels_from_file_paths(resized_file_paths, training=False)

        predictions = sess.run('predictions:0', feed_dict={'x:0': pixels_single_array})
        for prediction in range(len(predictions)):
            image_num = prediction + 1
            logger.info('#######################  IMAGE NUMBER : {}   #######################'.format(image_num))
            logger.info('Predictions:')
            for fruit_num in range(len(classes)):
                logger.info('%s : %s' % (classes[fruit_num], predictions[prediction][fruit_num]))


def utilise_cnn(user_function):
    if user_function == 'predict':
        logger.info('Runing Prediction Method.')
        input_image_directory = sys.argv[2]
        generate_predictions_using_cnn(input_image_directory)
    elif user_function == 'train':
        logger.info('Runing Training Method.')
        train_cnn()
    elif user_function == 'test':
        logger.info('Runing Testing Method.')
        test_cnn()


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    user_function = sys.argv[1]
    utilise_cnn(user_function)
