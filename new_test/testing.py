# %%
from utils.testloop import evaluate_passwords

# %% Testing TCN

from utils.tcnn import TCN


MODEL_WEIGHTS = './weights/tcn_best.pth'
ENCODER_PATH = './labels/label_encoder.pkl'
TEST_BASE_DIR = "dataset_split/test"
INPUT_SHAPE = (128, 47)
NUM_CLASSES = 36
MODEL = TCN(NUM_CLASSES, INPUT_SHAPE)

evaluate_passwords(MODEL_WEIGHTS, ENCODER_PATH, TEST_BASE_DIR, MODEL)


# %% Testing LSTM

from utils.lstm import LSTM

MODEL_WEIGHTS = "./weights/lstm_best.pth"
ENCODER_PATH = "./labels/lstm_encoder.pkl"
MODEL = LSTM(NUM_CLASSES, INPUT_SHAPE[0])

evaluate_passwords(MODEL_WEIGHTS, ENCODER_PATH, TEST_BASE_DIR, MODEL)

# %% Testing CRNN

from utils.crnn import CRNN

MODEL_WEIGHTS = "./weights/crnn_best.pth"
ENCODER_PATH = "./labels/crnn_encoder.pkl"
MODEL = CRNN(NUM_CLASSES, INPUT_SHAPE[0])

evaluate_passwords(MODEL_WEIGHTS, ENCODER_PATH, TEST_BASE_DIR, MODEL)

# %%
