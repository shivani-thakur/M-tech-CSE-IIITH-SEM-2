import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import KFold
import torch
from torch import nn, optim
import torchvision
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class AgeDataset(Dataset):
    def __init__(self, df, data_path, transform=None):
        self.df = df
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_name = os.path.join(self.data_path, self.df.iloc[index, 0])
        image = Image.open(img_name).convert('RGB')
        age = self.df.iloc[index, 1]
        if self.transform:
            image = self.transform(image)
        return image, age

data_path = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/train'
annotations_path = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/train.csv'
annotations = pd.read_csv(annotations_path)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up K-Fold cross-validation
num_epochs = 15
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_performance = {}

for fold, (train_index, val_index) in enumerate(kf.split(annotations)):
    print(f'Fold {fold + 1}')

    train_df = annotations.iloc[train_index]
    val_df = annotations.iloc[val_index]
    
    train_dataset = AgeDataset(train_df, data_path, transform=transform)
    val_dataset = AgeDataset(val_df, data_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

    # Initialize the model
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)

    # Loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, verbose=True)

    # Training loop with early stopping
    best_val_loss = float('inf')
    best_model_state = None
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, ages in tqdm(train_loader, desc=f'Training Epoch {epoch+1}', leave=False):
            images, ages = images.to(device), ages.to(device).float().view(-1, 1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, ages)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation step
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for val_images, val_ages in tqdm(val_loader, desc=f'Validating Epoch {epoch+1}', leave=False):
                val_images, val_ages = val_images.to(device), val_ages.to(device).float().view(-1, 1)
                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_ages)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, f'best_model_fold_{fold}.pth')
            print(f"Fold {fold+1}, Epoch {epoch+1}, Validation Loss Improved to {best_val_loss:.4f}")

        scheduler.step(avg_val_loss)

    fold_performance[fold] = best_val_loss

# Find and load the best fold
best_fold = min(fold_performance, key=fold_performance.get)
model.load_state_dict(torch.load(f'best_model_fold_{best_fold}.pth'))


# Prepare the test dataset
test_data_path = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/test'
test_annotations_path = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/submission.csv'
test_dataset = AgeDataset(pd.read_csv(test_annotations_path), test_data_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# Define the prediction function
def predict(loader, model):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc='Predicting', leave=False):
            images = images.to(device)
            outputs = model(images)
            predictions.extend(outputs.view(-1).cpu().numpy())
    return predictions

# Get predictions
predictions = predict(test_loader, model)

# Create the submission file
submission = pd.read_csv(test_annotations_path)
submission['age'] = predictions
submission.to_csv('/kaggle/working/submission.csv', index=False)
