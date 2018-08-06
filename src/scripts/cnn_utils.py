from common_utils import *

NUM_CHANNELS = 3
BATCH_SIZE = 20
SESSION_NUM = '0'
dropout_rate = 0.5
learning_rate = 0.0001
CHECKPOINT_PATH = os.path.join(SRC_FOLDER_PATH, 'cnn/model_checkpoint', SESSION_NUM)
SUMMERY_PATH = os.path.join(SRC_FOLDER_PATH, 'cnn/summary', SESSION_NUM)
