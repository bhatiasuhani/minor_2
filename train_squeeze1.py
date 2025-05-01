import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

# Paths to dataset
TRAIN_DIR = './dataset/train'
TEST_DIR = './dataset/test'

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def evaluate(model, loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    print("\n📈 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("❌ Dataset is empty or incorrectly structured. Please check the 'train' and 'test' folders.")
        return

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Load pre-trained SqueezeNet model
    model = models.squeezenet1_1(pretrained=True)

    # Modify classifier for 2 classes
    model.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = 2
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    EPOCHS = 10
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total
        print(f"\nEpoch [{epoch + 1}/{EPOCHS}] Loss: {running_loss / len(train_loader):.4f} | Accuracy: {accuracy:.2f}%")

    # Final Evaluation
    print("\n✅ Final Evaluation on Test Set:")
    evaluate(model, test_loader, device, train_dataset.classes)

    # Save model
    torch.save(model.state_dict(), 'squeezenet.pth')
    print("\n💾 Model saved as squeezenet.pth")

if __name__ == '__main__':
    main()
