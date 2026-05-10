import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from classifier import DeepfakeDataset, DeepfakeClassifier, transform
import matplotlib.pyplot as plt
import numpy as np
import os

# ── Paths ─────────────────────────────────────────────
BASE_PATH = "/kaggle/input/datasets/xhlulu/140k-real-and-fake-faces/real_vs_fake/real-vs-fake"
TEST_PATH = f"{BASE_PATH}/test"
MODEL_PATH = "models/deepfake_classifier.pth"

# ── Device ────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Load Model ────────────────────────────────────────
def load_model(path):
    model = DeepfakeClassifier().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Model loaded from {path} ✅")
    return model

# ── Load Dataset ──────────────────────────────────────
def load_test_data(test_path, batch_size=32):
    test_dataset = DeepfakeDataset(test_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Test samples: {len(test_dataset)}")
    return test_loader

# ── Evaluate ──────────────────────────────────────────
def evaluate_model(model, test_loader):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Accuracy
    test_acc = accuracy_score(all_labels, all_preds) * 100
    print(f"\nTest Accuracy: {test_acc:.2f}%")

    # Classification report
    print("\nDetailed Report:")
    print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))

    return all_labels, all_preds

# ── Confusion Matrix ──────────────────────────────────
def plot_confusion_matrix(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=['Real', 'Fake'],
           yticklabels=['Real', 'Fake'],
           xlabel='Predicted',
           ylabel='True',
           title='Confusion Matrix')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/confusion_matrix.png")
    plt.show()
    print("Confusion matrix saved! ✅")

# ── Visualize Predictions ─────────────────────────────
def visualize_predictions(model, test_loader, n=8):
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, predicted = outputs.max(1)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
        ax.imshow(img)
        pred = "Fake" if predicted[i] == 1 else "Real"
        true = "Fake" if labels[i] == 1 else "Real"
        color = "green" if pred == true else "red"
        ax.set_title(f"Pred: {pred}\nTrue: {true}", color=color)
        ax.axis('off')

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/predictions.png")
    plt.show()
    print("Predictions visualization saved! ✅")

# ── Run ───────────────────────────────────────────────
if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    test_loader = load_test_data(TEST_PATH)
    all_labels, all_preds = evaluate_model(model, test_loader)
    plot_confusion_matrix(all_labels, all_preds)
    visualize_predictions(model, test_loader)