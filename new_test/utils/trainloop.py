import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import dataset as dt
import matplotlib.pyplot as plt
import joblib

def train_model(TRAIN_DATA_DIR,
                VAL_DATA_DIR,
                BATCH_SIZE,
                EPOCHS,
                LEARNING_RATE,
                MODEL_NAME,
                LABEL_ENC,
                MODEL,
                PATIENCE
                ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_files, test_files, train_labels, test_labels, num_classes, le = dt.prepare_data(
        TRAIN_DATA_DIR, VAL_DATA_DIR
    )
    joblib.dump(le, LABEL_ENC)

    train_ds = dt.KeyAudioDataset(train_files, train_labels, is_train=True)
    test_ds = dt.KeyAudioDataset(test_files, test_labels)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = MODEL.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,)

    history = {
        "train_loss": [], "val_loss": [], 
        "train_acc": [], "val_acc": []
    }
    # Tracking variables
    best_val_loss = float('inf')
    patience = PATIENCE
    trigger_times = 0

    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        correct_train = 0
        total_train = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        avg_train_acc = correct_train / total_train

        model.eval()
        val_loss = 0.0
        
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()

        avg_val_loss = val_loss / len(test_loader)
        avg_val_acc = correct_val / total_val

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(avg_train_acc)
        history["val_acc"].append(avg_val_acc)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Loss: {avg_train_loss:.4f} | Acc: {avg_train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}"
        )

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_NAME)
            print(f"--> Best model saved (Loss: {best_val_loss:.4f})")
            trigger_times = 0 # Reset early stopping
        else:
            trigger_times += 1
            print(f"--> No improvement ({trigger_times}/{patience})")
            if trigger_times >= patience:
                print("Early stopping triggered!")
                break

    print("Training finished. Loading best model weights...")
    model.load_state_dict(torch.load(MODEL_NAME)) 

    return history




def plot_results(history):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Augmentation Pipeline Evaluation: Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid('True')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Training Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.title("Augmentation Pipeline Evaluation: Accuracy Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.grid('True')
    plt.legend()
    plt.show()
