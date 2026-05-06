# Peak Detection Splitting and Data Aug

# %% Library Import

from utils.data_aug import *
from utils.peak_detection import *
import os
from sklearn.model_selection import train_test_split
import shutil


# %% 1. Extract single presses from audio files

indir = os.path.join('.','mechanical_keyboard_dataset')
audio = os.listdir(indir)
w_type = 'cmor1.0-0.5'
outdir = os.path.join('.','out_dataset')
os.makedirs(outdir, exist_ok=True)

for wav in audio:
    file = os.path.join(indir, wav)
    peak_detection(file, w_type, outdir)

# %% 2. Data Augmentation

data_dir = os.path.join('.','out_dataset')              # This is where the full dataset is
train_dir = os.path.join('.', 'dataset_split', 'train') # Output: Training Data
val_dir = os.path.join('.', 'dataset_split', 'val')     # Output: Validating Data

# Create output directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

files = [f for f in os.listdir(data_dir) if f.endswith(".wav")]
file_paths = []
labels = []

for f in files:
    parts = f.split("_")
    label_str = parts[0]
    if label_str == 'w\u200b':
        label_str = 'w'
    file_paths.append(os.path.join(data_dir, f))
    labels.append(label_str)

X_train, X_val, y_train, y_test = train_test_split(file_paths, labels, test_size=0.2, stratify=labels)

def copy_files(file_list, dest_dir):
    for src_path in file_list:
        filename = os.path.basename(src_path)
        dst_path = os.path.join(dest_dir, filename)
        shutil.copy(src_path, dst_path)

print("Copying training files...")
copy_files(X_train, train_dir)

print("Copying valdation files...")
copy_files(X_val, val_dir)

# %%

indir = os.path.join('.', 'dataset_split', 'train') 
audio = os.listdir(indir)
w_type = 'cmor1.0-0.5'

for i in range(2, 6):
    outdir = os.path.join('.', 'dataset_split', 'train', f'dataset_{i}') 
    os.makedirs(outdir, exist_ok=True)
    data_augmentation(
        indir, outdir, i, noise_factor_low=30,
        noise_factor_high=15,
        )


# %% 3. Making test set

indir = os.path.join('.','TestAudioLong')
audio = os.listdir(indir)
w_type = 'cmor1.0-0.5'
outdir = os.path.join('.','dataset_split', 'test')
os.makedirs(outdir, exist_ok=True)

for f in os.listdir(indir):
    filename = f.replace("password_", "").replace(".wav", "")
    final_test_dir = os.path.join(outdir, filename)

    os.makedirs(final_test_dir, exist_ok=True)

# %%

def process_password_dataset(source_dir, dest_root):

    files = [f for f in os.listdir(source_dir) if f.startswith("password_") and f.endswith(".wav")]

    print(f"Found {len(files)} password files to process.")

    for file in files:
        password_str = file.replace("password_", "").replace(".wav", "")
        
        password_dir = os.path.join(dest_root, password_str)
        
        full_path = os.path.join(source_dir, file)
        
        print(f"Processing: {password_str}...")

        peak_detection(full_path, w_type, password_dir, name="temp")

        generated_files = [f for f in os.listdir(password_dir) if f.startswith("temp_") and f.endswith(".wav")]
        
        generated_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

        char_counts = {} 

        for i, filename in enumerate(generated_files):
            if i < len(password_str):
                char = password_str[i]
                
                if char not in char_counts:
                    new_name = f"{char}.wav"
                    char_counts[char] = 0
                else:
                    char_counts[char] += 1
                    new_name = f"{char}_{char_counts[char]}.wav"
                
                src = os.path.join(password_dir, filename)
                dst = os.path.join(password_dir, new_name)
                
                os.rename(src, dst)

    print("Processing Complete.")

process_password_dataset(indir, outdir)

# %%
