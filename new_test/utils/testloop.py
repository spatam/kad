import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import dataset as dt
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
from utils.dataset import KeyAudioDataset
import joblib


def prepare_single_test_folder(folder_path, le):
    file_paths = []
    labels = []
    
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".wav")])
    
    for f in files:
        filename_no_ext = os.path.splitext(f)[0]
        
        raw_label = filename_no_ext.split("_")[0]
        
        clean_label = raw_label.replace(".", "").strip()

        if clean_label == 'w\u200b':
            clean_label = 'w'
        
        file_paths.append(os.path.join(folder_path, f))
        labels.append(clean_label)

    try:
        encoded_labels = le.transform(labels)
    except ValueError as e:
        print(f"Error in folder {folder_path}")
        print(f"Cleaned labels extracted: {np.unique(labels)}")
        print(f"Encoder classes: {le.classes_}")
        raise e
        
    return np.array(file_paths), encoded_labels

def evaluate_passwords(
        MODEL_WEIGHTS,
        ENCODER_PATH,
        TEST_BASE_DIR,
        MODEL):

    le = joblib.load(ENCODER_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODEL.to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.eval()

    all_true_chars = []
    all_pred_chars = []
    password_results = {}

    password_folders = [f for f in os.listdir(TEST_BASE_DIR) if os.path.isdir(os.path.join(TEST_BASE_DIR, f))]
    print(f"| {'True Word':<12} | {'Predicted':<12} | {'Accuracy'}")
    print("-" * 45)

    for folder in password_folders:
        folder_path = os.path.join(TEST_BASE_DIR, folder)
        paths, labels = prepare_single_test_folder(folder_path, le)
        test_ds = KeyAudioDataset(paths, labels)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
        preds = []
        correct = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.squeeze(1).to(device) 
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_true_chars.append(targets.item())
                all_pred_chars.append(predicted.item())
                char_pred = le.inverse_transform(predicted.cpu().numpy())[0]
                preds.append(char_pred)
                if predicted.item() == targets.item():
                    correct += 1

        accuracy = (correct / len(labels)) * 100
        predicted_word = "".join(preds)
        true_word = ''.join(filter(str.isalnum, folder)) 
        password_results[true_word] = accuracy
        print(f"| {true_word:<12} | {predicted_word:<12} | {accuracy:>6.1f}%")
    pass

    print(f'Average accuracy : {sum(password_results.values())/len(password_results.values())}')
    print(f'F1 Score : {f1_score(all_true_chars, all_pred_chars, average='macro')}')

    plot_results(all_true_chars, all_pred_chars, password_results, le.classes_)

def plot_results(y_true, y_pred, password_accs, class_names):
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    names = list(password_accs.keys())
    values = list(password_accs.values())
    colors = ['green' if v > 80 else 'orange' if v > 50 else 'red' for v in values]
    plt.bar(names, values, color=colors)
    plt.title("Accuracy per Password")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=45)
    plt.ylim(0, 105)

    plt.subplot(1, 2, 2)
    
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.yticks(rotation=0) 
    
    plt.title("Character Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()
    plt.show()

