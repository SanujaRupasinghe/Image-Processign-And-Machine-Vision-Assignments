import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import os

def calculate_accuracy(loader, model):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get the class with the highest score
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Data Preparation
data_dir = './hymenoptera_data'
os.makedirs(data_dir, exist_ok=True)

# Define transforms for the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Pretrained model normalization
])

# Download and prepare datasets from torchvision's example data
trainset = datasets.FakeData(transform=transform, size=500, image_size=(3, 224, 224), num_classes=2)
testset = datasets.FakeData(transform=transform, size=100, image_size=(3, 224, 224), num_classes=2)

trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

# Load Pretrained ResNet18
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Adapting for 2 classes: ants and bees

# Fine-Tuning
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(10):
    running_loss = 0.0
    for inputs, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader)}")

print(f"Training Accuracy: {calculate_accuracy(trainloader, model)}%")
print(f"Test Accuracy: {calculate_accuracy(testloader, model)}%")
