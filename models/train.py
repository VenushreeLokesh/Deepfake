import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from classifier import DeepfakeDataset, DeepfakeClassifier, transform, get_sampler
import os

BASE_PATH = "/kaggle/input/datasets/xhlulu/140k-real-and-fake-faces/real_vs_fake/real-vs-fake"
TRAIN_PATH = f"{BASE_PATH}/train"
VALID_PATH = f"{BASE_PATH}/valid"
MODEL_SAVE_PATH = "models/deepfake_classifier.pth"

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_dataset = DeepfakeDataset(TRAIN_PATH, transform=transform)
valid_dataset = DeepfakeDataset(VALID_PATH, transform=transform)

sampler = get_sampler(train_dataset)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(valid_dataset)}")

model = DeepfakeClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_model(model, train_loader, valid_loader, epochs):
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss/len(train_loader):.4f} "
              f"Train Acc: {train_acc:.2f}% "
              f"Val Loss: {val_loss/len(valid_loader):.4f} "
              f"Val Acc: {val_acc:.2f}%")

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

if __name__ == "__main__":
    train_model(model, train_loader, valid_loader, EPOCHS)
    save_model(model, MODEL_SAVE_PATH)