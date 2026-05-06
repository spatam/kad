# %%
from utils.tcnn import TCN
from utils import dataset as dt
from utils.trainloop import *

# %% TCN

TRAIN_DATA_DIR = "./dataset_split/train/dataset_5"
VAL_DATA_DIR = "./dataset_split/val"
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
MODEL_NAME = './weights/tcn_best.pth'
LABEL_ENC = './labels/label_encoder.pkl'
PATIENCE = 10


train_files, test_files, train_labels, test_labels, num_classes, le = dt.prepare_data(
        TRAIN_DATA_DIR, VAL_DATA_DIR
    )

train_ds = dt.KeyAudioDataset(train_files, train_labels, is_train=True)

sample_data, _ = train_ds[0]
input_shape = sample_data.shape


MODEL = TCN(num_classes, input_shape[1:])

hist = train_model(TRAIN_DATA_DIR,
VAL_DATA_DIR,
BATCH_SIZE,
EPOCHS,
LEARNING_RATE,
MODEL_NAME,
LABEL_ENC,
MODEL,
PATIENCE)

plot_results(hist)


# %% LSTM

from utils.lstm import LSTM

MODEL = LSTM(num_classes, input_shape[1])
MODEL_NAME = 'lstm_best.pth'
EPOCHS = 100
LEARNING_RATE = 0.0001
MODEL_NAME = './weights/lstm_best.pth'
LABEL_ENC = './labels/lstm_encoder.pkl'
PATIENCE = 20


hist = train_model(TRAIN_DATA_DIR,
VAL_DATA_DIR,
BATCH_SIZE,
EPOCHS,
LEARNING_RATE,
MODEL_NAME,
LABEL_ENC,
MODEL,
PATIENCE)

plot_results(hist)


# %% CRNN

from utils.crnn import CRNN

MODEL_NAME = './weights/crnn_best.pth'
LABEL_ENC = './labels/crnn_encoder.pkl'
MODEL = CRNN(num_classes, input_shape[1])

hist = train_model(TRAIN_DATA_DIR,
VAL_DATA_DIR,
BATCH_SIZE,
EPOCHS,
LEARNING_RATE,
MODEL_NAME,
LABEL_ENC,
MODEL,
PATIENCE)

plot_results(hist)

# %%
