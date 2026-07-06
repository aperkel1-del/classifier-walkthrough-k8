import os
import time
import habana_frameworks.torch.core as htcore
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

#configure device type
device = torch.device("hpu")

#set input and output directories, and configure training parameters
DATA_DIR = '/data/animal_data'
OUTPUT_DIR = '/workspace/checkpoint'
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4
NUM_WORKERS = 4

#save full database in directory and transform images to compatible size and type
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
full_dataset = datasets.ImageFolder(DATA_DIR, transform = transform)

#--- Split Data into Val and Train ---
val_fraction = 0.2
val_size = int(len(full_dataset) * val_fraction)
train_size = len(full_dataset) - val_size

train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle = True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle = False, num_workers=NUM_WORKERS)

class_names = full_dataset.classes
num_classes = len(class_names)
print(f"Classes: {class_names}")
print(f"Train images: {len(train_dataset)}")
print(f"Val images:   {len(val_dataset)}\n")

os.makedirs(OUTPUT_DIR, exist_ok=True)

#--- Load Pretrained Model ---
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
htcore.mark_step()

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

best_val_acc=0.0

#training and validation loops through 10 epochs.
for epoch in range(1, EPOCHS+1):
    print(f"═══ Epoch {epoch}/{EPOCHS} ═══")
    
    #--- Training Phase ---
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type="hpu", dtype=torch.bfloat16):
            outputs = model(images)
            loss = criterion(outputs, labels)
        loss.backward()

        htcore.mark_step()

        optimizer.step()

        htcore.mark_step()

        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        acc = correct / images.size(0) * 100

        print(f"  Iteration {i+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {acc:.1f}%")

    #--- Validation Phase ---
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type="hpu", dtype=torch.bfloat16):
                outputs = model(images)
            _,  predicted = outputs.max(1)
            total_correct += predicted.eq(labels).sum().item()
    htcore.mark_step()
    val_acc = total_correct / len(val_dataset) * 100
    print(f"\n  ✓ Validation Accuracy: {val_acc:.2f}%\n")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.cpu().state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
        model = model.to(device)
        print(f"  ★ New best model saved ({val_acc:.2f}%)\n")

print(f"Done! Best validation accuracy: {best_val_acc:.2f}%")
