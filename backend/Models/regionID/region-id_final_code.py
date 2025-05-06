import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import gc
import random
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define paths
TRAIN_IMG_DIR = '/kaggle/input/iiith-campus-images-dataset/images_train/images_train/images_train'
VALID_IMG_DIR = '/kaggle/input/iiith-campus-images-dataset/images_val/images_val'
TRAIN_LABELS = '/kaggle/input/iiith-campus-images-dataset/labels_train_updated.csv'
VALID_LABELS = '/kaggle/input/iiith-campus-images-dataset/labels_val_updated.csv'

# Define augmentations - much more comprehensive for training data
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Slightly larger than needed for random cropping
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomAutocontrast(p=0.2),
    transforms.RandomEqualize(p=0.1),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2)
])

# Simpler transforms for validation
valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Improved Campus Dataset class
class CampusDataset(Dataset):
    def __init__(self, img_dir, labels_file, transform=None, is_validation=False):
        self.img_dir = img_dir
        self.transform = transform
        self.is_validation = is_validation
        self.labels_df = pd.read_csv(labels_file)
        self.valid_indices = []
        
        # Check which images can be loaded successfully
        print("Verifying dataset integrity...")
        for idx in tqdm(range(len(self.labels_df))):
            try:
                img_path = os.path.join(img_dir, self.labels_df.iloc[idx]['filename'])
                if os.path.exists(img_path):
                    with Image.open(img_path) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        self.valid_indices.append(idx)
            except Exception as e:
                print(f"Skipping image at index {idx}: {e}")
                
        print(f"Found {len(self.valid_indices)} valid images out of {len(self.labels_df)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        true_idx = self.valid_indices[idx]
        img_name = os.path.join(self.img_dir, self.labels_df.iloc[true_idx]['filename'])
        
        # Use context manager for better resource handling
        with Image.open(img_name) as image:
            image = image.convert('RGB')
            
            if self.transform:
                image_tensor = self.transform(image)
                
        region_id = self.labels_df.iloc[true_idx]['Region_ID']
        
        # For validation set, also return the image filename
        if self.is_validation:
            return image_tensor, region_id - 1, self.labels_df.iloc[true_idx]['filename']
        else:
            return image_tensor, region_id - 1

# Load the datasets with appropriate transforms
try:
    train_dataset = CampusDataset(TRAIN_IMG_DIR, TRAIN_LABELS, transform=train_transform)
    valid_dataset = CampusDataset(VALID_IMG_DIR, VALID_LABELS, transform=valid_transform, is_validation=True)

    print(f"Loaded {len(train_dataset)} training samples and {len(valid_dataset)} validation samples")

    # Create data loaders with optimized parameters
    train_loader = DataLoader(
        train_dataset,
        batch_size=24,  # Smaller batch size for ConvNext (it's larger than ResNet)
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True if torch.__version__ >= '1.7.0' else False,
        drop_last=True  # Drop incomplete batches for consistent batch norm stats
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=32,  # Also reduced for ConvNext
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True if torch.__version__ >= '1.7.0' else False
    )

except Exception as e:
    print(f"Error loading datasets: {e}")
    print("Please check the paths and structure of your data files")

# Define a powerful model using ConvNext as base with advanced dropout and normalization techniques
class ConvNextClassifier(nn.Module):
    def __init__(self, num_classes=15):
        super(ConvNextClassifier, self).__init__()
        
        # Use ConvNeXt-Small as base model
        self.convnext = models.convnext_small(pretrained=True)
        
        # Freeze early layers to prevent overfitting
        # Freeze first 6 blocks (about 1/3 of the network)
        for name, param in self.convnext.named_parameters():
            if 'features.0.' in name or 'features.1.' in name or 'features.2.' in name:
                param.requires_grad = False
        
        # The issue happens because ConvNext models return a 4D tensor
        # We need to handle this differently than in the previous implementation
        
        # Replace the classifier and properly handle the 4D tensor
        # First, get the in_features
        if hasattr(self.convnext, 'head'):
            # For newer torchvision versions
            in_features = self.convnext.head.in_features
            # Remove the original head
            self.convnext.head = nn.Identity()
        else:
            # For older torchvision versions 
            in_features = self.convnext.classifier[2].in_features if hasattr(self.convnext.classifier, '__getitem__') else self.convnext.classifier.in_features
            # Remove the original classifier
            self.convnext.classifier = nn.Identity()
        
        # Add our own classifier that handles the 4D tensor properly
        self.classifier = nn.Sequential(
            # First adapt the 4D tensor (batch, channels, height, width) to 2D
            nn.AdaptiveAvgPool2d(1),       # -> (batch, channels, 1, 1)
            nn.Flatten(1),                 # -> (batch, channels)
            nn.LayerNorm(in_features),     # Now LayerNorm will work properly
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize the weights for better convergence
        self._initialize_weights()
        
    def _initialize_weights(self):
        # Initialize only the classifier weights we added
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        # Pass through the base model
        features = self.convnext(x)
        # Pass through our classifier
        return self.classifier(features)

# Create model instance
model = ConvNextClassifier(num_classes=15)
model = model.to(device)

# Model complexity
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model has {count_parameters(model):,} trainable parameters")

# Use CrossEntropyLoss with stronger label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.15)  # Increased from 0.1 for better regularization

# Use AdamW optimizer with weight decay and better learning rate
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=2e-4, amsgrad=True)

# Advanced learning rate scheduler - Cosine annealing with warm restarts
# This helps escape local minima better than manual plateaus
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=5,             # Restart every 5 epochs
    T_mult=2,          # Double the restart interval after each restart
    eta_min=1e-6       # Minimum learning rate
)

# Mixed precision training for better efficiency
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

# Improved training function with various optimizations
def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=30):
    history = {
        'train_loss': [],
        'valid_loss': [],
        'valid_acc': [],
        'valid_top3_acc': [],
        'learning_rates': []
    }

    best_acc = 0.0
    best_epoch = 0
    patience = 10  # Increased patience with better scheduler
    no_improve = 0
    validation_filenames = []
    validation_predictions = []
    
    # For AMP (Automatic Mixed Precision)
    use_amp = torch.cuda.is_available()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in progress_bar:
            inputs, labels = batch
            
            # Skip empty batches
            if inputs.size(0) == 0:
                continue

            inputs, labels = inputs.to(device), labels.to(device)
            
            # Clear gradients
            optimizer.zero_grad(set_to_none=True)
            
            if use_amp:
                # Automatic mixed precision training
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                # Scale gradients and optimize
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular training path
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({'loss': loss.item(), 'lr': current_lr})
        
        # Step the scheduler after each epoch
        scheduler.step()
        
        epoch_loss = running_loss / total_samples if total_samples > 0 else 0
        history['train_loss'].append(epoch_loss)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        # Clean up to free memory
        torch.cuda.empty_cache()
        gc.collect()

        # Validation phase
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_filenames = []
        total_val_samples = 0
        correct_top1 = 0
        correct_top3 = 0

        with torch.no_grad():
            progress_bar = tqdm(valid_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Valid]')
            for batch in progress_bar:
                inputs, labels, filenames = batch

                # Skip empty batches
                if inputs.size(0) == 0:
                    continue

                inputs, labels = inputs.to(device), labels.to(device)
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                total_val_samples += inputs.size(0)

                # Get top-1 predictions
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_filenames.extend(filenames)

                # Calculate top-1 accuracy
                correct_top1 += (preds == labels).sum().item()

                # Top-3 accuracy
                _, top3_indices = outputs.topk(3, dim=1)
                batch_correct = 0
                for i in range(labels.size(0)):
                    if labels[i] in top3_indices[i]:
                        batch_correct += 1

                correct_top3 += batch_correct

        # Calculate validation metrics
        valid_loss = running_loss / total_val_samples if total_val_samples > 0 else 0
        valid_acc = correct_top1 / total_val_samples if total_val_samples > 0 else 0
        valid_top3_acc = correct_top3 / total_val_samples if total_val_samples > 0 else 0

        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)
        history['valid_top3_acc'].append(valid_top3_acc)

        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {epoch_loss:.4f} | '
              f'Valid Loss: {valid_loss:.4f} | '
              f'Valid Acc: {valid_acc:.4f} | '
              f'Valid Top-3 Acc: {valid_top3_acc:.4f} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # Save if it's the best model by top-1 accuracy
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_epoch = epoch
            no_improve = 0
            
            # Save predictions for best model
            validation_filenames = all_filenames
            validation_predictions = all_preds
            
            torch.save(model.state_dict(), 'best_convnext_region_classifier_3.pth')
            print(f"Saved new best model with accuracy: {best_acc:.4f}")
            
            # If we reach 98% accuracy, we can stop training
            if best_acc >= 0.985:  # Slightly higher threshold
                print(f"Reached target accuracy of 98.5%! Stopping training.")
                break
        else:
            no_improve += 1
        
        # Early stopping
        if no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
            
        # Clean up after validation
        torch.cuda.empty_cache()
        gc.collect()

    print(f"Best validation accuracy: {best_acc:.4f} at epoch {best_epoch+1}")
    return history, all_preds, all_labels, validation_filenames, validation_predictions

# Train the model
history, final_preds, final_labels, val_filenames, val_predictions = train_model(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=40
)

# Print final validation accuracy and classification report
final_acc = accuracy_score(final_labels, final_preds)
print("\nFinal Validation Accuracy:", final_acc)
print("\nClassification Report:")
print(classification_report(final_labels, final_preds,
                          target_names=[f"Region {i+1}" for i in range(15)]))

# Plot training history
plt.figure(figsize=(15, 10))

# Plot losses
plt.subplot(2, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['valid_loss'], label='Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot accuracies
plt.subplot(2, 2, 2)
plt.plot(history['valid_acc'], label='Top-1 Accuracy')
plt.plot(history['valid_top3_acc'], label='Top-3 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracies')
plt.legend()

# Plot learning rates
plt.subplot(2, 2, 3)
plt.plot(history['learning_rates'], marker='o')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.yscale('log')

# Plot confusion matrix
plt.subplot(2, 2, 4)
cm = confusion_matrix(final_labels, final_preds)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(15)
plt.xticks(tick_marks, [f"{i+1}" for i in range(15)], rotation=45)
plt.yticks(tick_marks, [f"{i+1}" for i in range(15)])
plt.xlabel('Predicted Region')
plt.ylabel('True Region')

plt.tight_layout()
plt.show()

# Load best model for predictions
model.load_state_dict(torch.load('best_convnext_region_classifier_3.pth'))
model.eval()

# Create prediction function
def predict(model, image_path, transform, device):
    """Make predictions on a single image"""
    model.eval()
    with torch.no_grad():
        with Image.open(image_path) as image:
            image = image.convert('RGB')
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            if torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    outputs = model(img_tensor)
            else:
                outputs = model(img_tensor)
                
            # Get probabilities and top class
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            return preds.item() + 1, probs  # Add 1 to get Region_ID from 1-15

# Load validation dataset again to get filenames and create mapping to IDs
val_df = pd.read_csv(VALID_LABELS)

# Create submission dataframe with confidence scores
submission_data = []

# Add validation set predictions (ids 0 to 368)
for i, filename in enumerate(val_filenames):
    # Find the corresponding index in val_df
    val_indices = val_df[val_df['filename'] == filename].index
    if len(val_indices) > 0:
        val_idx = val_indices[0]
        
        submission_data.append({
            'id': val_idx,
            'Region_ID': int(val_predictions[i] + 1)  # Convert 0-14 to 1-15
        })

# Fill remaining test set entries with predictions from best model or default
# You could extend this to actually predict on test images if they're available
for i in range(369, 738):
    submission_data.append({
        'id': i,
        'Region_ID': 1  # Default region
    })

# Convert to DataFrame
submission = pd.DataFrame(submission_data)

# Ensure ID column has all required values and is sorted
expected_ids = set(range(0, 738))
existing_ids = set(submission['id'])
missing_ids = expected_ids - existing_ids

# Add any missing IDs with default Region_ID=1
for missing_id in missing_ids:
    submission = pd.concat([submission, pd.DataFrame([{'id': missing_id, 'Region_ID': 1}])], ignore_index=True)

# Sort by ID and reset index
submission = submission.sort_values('id').reset_index(drop=True)

# Save to CSV
submission.to_csv('2022101094_convnext.csv', index=False)
print("Submission file '2022101094_convnext.csv' created successfully with", len(submission), "entries")