import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

# The are placeholders
SAMPLE_RATE = 48000
DURATION = 0.5

class KeyAudioDataset(Dataset):
    def __init__(self, file_paths, labels, sr=SAMPLE_RATE, duration=DURATION, is_train=False, is_plot=False):
        self.file_paths = file_paths            # Location where audios are
        self.labels = labels                    # Labels for each file
        self.sr = sr                            # Sample rate ('resolution of audio')
        self.target_len = int(sr * duration)    # Length of the audio is given by resolution times duration
        self.is_train = is_train
        self.is_plot = is_plot

        if self.is_train:
            self.mask_f = torchaudio.transforms.FrequencyMasking(5)
            self.mask_t = torchaudio.transforms.TimeMasking(2)

        self.mel_transform = nn.Sequential(torchaudio.transforms.MelSpectrogram(
        sample_rate=self.sr,    # Torchaudio default: 16kHz. We use 48kHz,
        n_fft=2048,             # Torchaudio default: 400. 2048 is the default in Librosa
        hop_length=512,         # Torchaudio default: 200. 512 is the default in Librosa
        n_mels=128              # Standard Mel count
        ), torchaudio.transforms.AmplitudeToDB())

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        # We initially load the audio along with its original sample rate
        y, orig_sr = torchaudio.load(path, normalize=True) # We load the audio using torchaudio resulting in (channels, time)

        # If sample rate is different from the one manually set from us, it has to be resempled
        if orig_sr != self.sr:
            y = torchaudio.functional.resample(waveform=y, orig_freq=orig_sr, new_freq=self.sr) # We take the loaded tensor with original frequency and change it into sr


        # Stereo audios store L,R audios in 2 different channels. 
        # If any of our WAV file is in stereo, we need to merge the first channel
        if y.shape[0] > 1:
            y = torch.mean(y, dim=0, keepdim=True) # Keepdim True allow us to just merge the first dimensions (merge LR channels) and leave the others unchanged

        # We want audio length to be consistent across different samples
        num_samples = y.shape[1] # torchaudio.load returns (channel, time), we need to focus on time

        # If our audio is larger than the desired size...
        if num_samples > self.target_len:
            y = y[:, :self.target_len] # We slice the time dimension up to the desired length

        # If its smaller
        elif num_samples < self.target_len:
            padding = self.target_len - num_samples # We compute the padding as how many ms we need to reach the desired dimension
            y = torch.nn.functional.pad(y, (0, padding)) # This function increase the size of the tensor by padding (All the last pad val are 0s)

        # We apply the mel transform defined in the dunder init
        mel = self.mel_transform(y)
        
        if self.is_plot:
            mean = mel.mean()
            std = mel.std()
            mel = (mel - mean) / (std + 1e-7)
        
        if self.is_train:
            mel = self.mask_f(mel)
            mel = self.mask_t(mel)

        # Return two values: the mel frequency of the audio, and the label
        return mel, torch.tensor(label, dtype=torch.long)


def prepare_data(train_data_dir, test_data_dir):
    # We initialize the lists for storing training audio path and labels
    file_paths_train = []
    labels_train = []
    
    for f in os.listdir(train_data_dir): # We iterate through the whole dataset (it contains only letters or numbers)
        if f.endswith(".wav"): 
            file_paths_train.append(os.path.join(train_data_dir, f))    # Append each pathfile to the predefined list
            label = f.split("_")[1]                                     # Training data structure names = aug_ClassType_NumOfClick_NumOfAug.wav
            labels_train.append(label)                                  # We append the ClassType to the label list

    # The same is done for the test set
    file_paths_test = []
    labels_test = []

    for f in os.listdir(test_data_dir):
        if f.endswith(".wav"):
            file_paths_test.append(os.path.join(test_data_dir, f))
            label = f.split("_")[0]                                     # Test data structure names = ClassType_NumOfClick.wav
            labels_test.append(label)                                   # We append the ClassType

    # Label Econding. We want to turn the label classes (which are all the buttons of the keyboard), 
    # into standardized int (i.e. encoded_lab 'j' must be the same in both test and train set)
    le = LabelEncoder()
    le.fit(labels_test)                            # We fit the encoder in the train
    y_train_encoded = le.transform(labels_train)    # And apply the same encode on train
    y_test_encoded = le.transform(labels_test)      # And test

    # Let's notice that train/test splitting was done using the stratify flag put as True,
    # meaning that fitting the encoder on train labels is the same thing as doing it in the test labels

    # We turn paths into np arrays for simplicity
    X_train_paths = np.array(file_paths_train)
    X_test_paths = np.array(file_paths_test)

    # We print info of the datas
    print(f"Data Loaded:")
    print(f"Training: {len(X_train_paths)} Augmented files")
    print(f"Testing:  {len(X_test_paths)} Original files")
    print(f"Classes:  {len(np.unique(labels_train))}")

    # Return results in a sklearn test_train_split-like output
    return X_train_paths, X_test_paths, y_train_encoded, y_test_encoded, len(np.unique(labels_train)), le
