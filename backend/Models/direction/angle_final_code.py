import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
import timm
import math
import random
from torch.cuda.amp import GradScaler, autocast

# -----------------------
# Paths and Globals
# -----------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 96     # increased batch size
EPOCHS = 60        # more epochs
BASE_LR = 5e-4      # higher learning rate
WEIGHT_DECAY = 1e-4 # added weight decay
SEED = 42           # added seed for reproducibility

TRAIN_CSV = '/kaggle/input/iiith-images-latlong-smai/cleaned_data_train.csv'
TRAIN_IMG_DIR = '/kaggle/input/iiith-images-latlong-smai/images_train/images_train/images_train/'
VAL_CSV = '/kaggle/input/iiith-images-latlong-smai/labels_val_updated.csv'
VAL_IMG_DIR = '/kaggle/input/iiith-images-latlong-smai/images_val/images_val/'
MODEL_PATH = '/kaggle/working/efficientnet_angle_regressor.pt'

# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)

# -----------------------
# Dataset - REMOVED Region_ID dependency
# -----------------------
class CampusDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, is_val=False):
        df = pd.read_csv(csv_file)
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_val = is_val
        if is_val:
            self.df['idx'] = self.df.index
        # Convert angles to radians for smoother learning
        self.df['angle_rad'] = self.df['angle'] * (math.pi / 180.0)
        # Create sin and cos components for circular regression - ensure float32
        self.df['sin_angle'] = np.sin(self.df['angle_rad']).astype(np.float32)
        self.df['cos_angle'] = np.cos(self.df['angle_rad']).astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        angle = float(row['angle'])
        sin_angle = float(row['sin_angle'])
        cos_angle = float(row['cos_angle'])
        
        if self.is_val:
            return img, torch.tensor(angle, dtype=torch.float32), torch.tensor(sin_angle, dtype=torch.float32), torch.tensor(cos_angle, dtype=torch.float32), torch.tensor(int(row['idx']), dtype=torch.long)
        return img, torch.tensor(angle, dtype=torch.float32), torch.tensor(sin_angle, dtype=torch.float32), torch.tensor(cos_angle, dtype=torch.float32)

# -----------------------
# Augmentations - Same as original
# -----------------------
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2)
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -----------------------
# Model: EfficientNet WITHOUT Region Conditioning
# -----------------------
class AngleRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        # Use EfficientNet B0 as backbone
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, features_only=True)
        
        # Extract feature dimensions from the backbone
        dummy_input = torch.zeros(1, 3, 224, 224)
        features = self.backbone(dummy_input)
        feature_dim = features[-1].shape[1]  # Last feature map channels
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature processing
        self.features = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.2)
        )
        
        # Sin/Cos prediction - circular regression approach
        self.head = nn.Linear(256, 2)
        
        # Enable gradient checkpointing if available
        if hasattr(self.backbone, 'gradient_checkpointing_enable'):
            self.backbone.gradient_checkpointing_enable()

    def forward(self, x):
        features = self.backbone(x)
        x = self.global_pool(features[-1]).squeeze(-1).squeeze(-1)
        
        # Process features
        x = self.features(x)
        
        # Predict sin and cos components
        sin_cos = self.head(x)
        sin_pred, cos_pred = sin_cos.split(1, dim=1)
        
        # Normalize the output to ensure it falls on the unit circle - ensure float32
        norm = torch.sqrt(sin_pred**2 + cos_pred**2) + 1e-8
        sin_norm = sin_pred / norm
        cos_norm = cos_pred / norm
        
        # Convert to angle in degrees
        angle = torch.atan2(sin_norm, cos_norm) * (180.0 / torch.tensor(math.pi, dtype=torch.float32, device=sin_pred.device))
        # Ensure angle is in [0, 360)
        angle = (angle + 360) % 360
        
        return angle.squeeze(1), sin_norm.squeeze(1), cos_norm.squeeze(1)

# -----------------------
# Loss Functions - Same as original
# -----------------------
def circle_loss(sin_pred, cos_pred, sin_true, cos_true):
    # MSE loss between the normalized sin and cos components
    return nn.MSELoss()(sin_pred, sin_true) + nn.MSELoss()(cos_pred, cos_true)

def maae_loss(pred, true):
    diff = torch.abs(pred - true)
    return torch.mean(torch.min(diff, 360 - diff))

# -----------------------
# Mixup Implementation - Modified to remove region_id dependency
# -----------------------
def mixup_data(x, y_angle, y_sin, y_cos, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    lam = float(lam)  # Ensure lam is a float32 compatible value
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    
    # We mix angles in the sin/cos space to handle the circular nature
    mixed_sin = lam * y_sin + (1 - lam) * y_sin[index]
    mixed_cos = lam * y_cos + (1 - lam) * y_cos[index]
    
    # Reconstruct angle from sin/cos - ensure torch.float32
    mixed_angle = torch.atan2(mixed_sin, mixed_cos) * (180.0 / math.pi)
    mixed_angle = (mixed_angle + 360) % 360
    
    return mixed_x, mixed_angle, mixed_sin, mixed_cos, index, lam

# -----------------------
# DataLoaders
# -----------------------
train_ds = CampusDataset(TRAIN_CSV, TRAIN_IMG_DIR, transform=train_tf)
val_ds = CampusDataset(VAL_CSV, VAL_IMG_DIR, transform=val_tf, is_val=True)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# -----------------------
# Model, Optimizer, AMP, Scheduler
# -----------------------
model = AngleRegressor().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
scaler = GradScaler()

# OneCycleLR for faster convergence
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=BASE_LR,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS,
    pct_start=0.1,
    div_factor=10.0,
    final_div_factor=1000.0
)

# -----------------------
# EMA Model (Exponential Moving Average for better stability) - Same as original
# -----------------------
class EMA():
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()
                
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
                
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# Initialize EMA
ema = EMA(model, decay=0.998)
ema.register()

# -----------------------
# Training & Validation - Modified to remove region_id dependency
# -----------------------
best_maae = float('inf')
val_maae_history = []
train_loss_history = []

print(f"Training on {DEVICE} with {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Model: EfficientNet B0 Image-Only")
print(f"Batch Size: {BATCH_SIZE}, Epochs: {EPOCHS}, Base LR: {BASE_LR}")

for epoch in range(1, EPOCHS+1):
    # Train
    model.train()
    train_loss = 0
    train_angle_loss = 0
    train_circle_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{EPOCHS} [Train]')
    for imgs, angles, sin_angles, cos_angles in progress_bar:
        imgs = imgs.to(DEVICE)
        angles = angles.to(DEVICE)
        sin_angles = sin_angles.to(DEVICE)
        cos_angles = cos_angles.to(DEVICE)

        # Apply mixup with 50% probability
        if random.random() < 0.5:
            imgs, mixed_angles, mixed_sin, mixed_cos, _, _ = mixup_data(
                imgs, angles, sin_angles, cos_angles
            )
            sin_angles, cos_angles = mixed_sin, mixed_cos
            angles = mixed_angles

        optimizer.zero_grad()
        
        with autocast():
            pred_angles, pred_sin, pred_cos = model(imgs)
            
            # Combined loss: angle MAAE + sin-cos circle loss
            angle_loss = maae_loss(pred_angles, angles)
            circ_loss = circle_loss(pred_sin, pred_cos, sin_angles, cos_angles)
            loss = angle_loss * 0.5 + circ_loss * 0.5
            
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Update EMA model
        ema.update()
        
        train_loss += loss.item() * imgs.size(0)
        train_angle_loss += angle_loss.item() * imgs.size(0)
        train_circle_loss += circ_loss.item() * imgs.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}", 
            'angle_loss': f"{angle_loss.item():.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
        })
    
    train_loss /= len(train_loader.dataset)
    train_angle_loss /= len(train_loader.dataset)
    train_circle_loss /= len(train_loader.dataset)
    train_loss_history.append(train_loss)
    
    # Validate with EMA model
    ema.apply_shadow()
    model.eval()
    val_loss = 0
    all_preds, all_trues, all_indices = [], [], []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f'Epoch {epoch}/{EPOCHS} [Val]')
        for imgs, angles, sin_angles, cos_angles, indices in progress_bar:
            imgs = imgs.to(DEVICE)
            angles = angles.to(DEVICE)
            sin_angles = sin_angles.to(DEVICE)
            cos_angles = cos_angles.to(DEVICE)
            
            pred_angles, pred_sin, pred_cos = model(imgs)
            angle_loss = maae_loss(pred_angles, angles)
            val_loss += angle_loss.item() * imgs.size(0)
            
            all_preds.append(pred_angles.cpu().numpy())
            all_trues.append(angles.cpu().numpy())
            all_indices.append(indices.numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'val_loss': f"{angle_loss.item():.4f}"})
    
    # Restore original model
    ema.restore()
            
    val_loss /= len(val_loader.dataset)
    preds = np.concatenate(all_preds)
    trues = np.concatenate(all_trues)
    indices = np.concatenate(all_indices)
    
    # Calculate MAAE (Mean Absolute Angular Error)
    val_maae = np.mean(np.minimum(np.abs(preds-trues), 360-np.abs(preds-trues)))
    val_maae_history.append(val_maae)
    
    print(f"Epoch {epoch}/{EPOCHS}")
    print(f"  Train Loss: {train_loss:.4f} (Angle: {train_angle_loss:.4f}, Circle: {train_circle_loss:.4f})")
    print(f"  Val MAAE: {val_maae:.4f}")
    
    # Save best model
    if val_maae < best_maae:
        best_maae = val_maae
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'ema_shadow': ema.shadow,
            'optimizer_state_dict': optimizer.state_dict(),
            'val_maae': val_maae,
            'best_maae': best_maae,
        }, MODEL_PATH)
        print(f"  Saved best model (MAAE: {best_maae:.4f})")

print(f"\nTraining completed!")
print(f"Best validation MAAE: {best_maae:.4f}")

# -----------------------
# Plot training history - Same as original
# -----------------------
try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS+1), train_loss_history, label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS+1), val_maae_history, label='Val MAAE')
    plt.axhline(y=best_maae, color='r', linestyle='--', label=f'Best MAAE: {best_maae:.4f}')
    plt.title('Validation MAAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAAE (degrees)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.close()
except:
    print("Could not generate training history plot.")
    
    
    
    
import os
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# ---- Paths ----
VAL_CSV       = '/kaggle/input/iiith-images-latlong-smai/labels_val_updated.csv'
VAL_IMG_DIR   = '/kaggle/input/iiith-images-latlong-smai/images_val/images_val'
TEST_IMG_DIR  = '/kaggle/input/iiith-images-latlong-smai/images_test/images_test'
ANGLE_MODEL_PATH = '/kaggle/working/efficientnet_angle_regressor.pt'
OUTPUT_CSV    = '/kaggle/working/final_predictions.csv'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- Transforms ----
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---- TestImageDataset (same) ----
class TestImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.files = sorted(os.listdir(img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.files[idx])).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, idx

# ---- Load Angle Regressor ----
angle_model = AngleRegressor()
ckpt = torch.load(ANGLE_MODEL_PATH, map_location=DEVICE)
angle_model.load_state_dict(ckpt['model_state_dict'])
angle_model.to(DEVICE).eval()
# apply EMA if present
if 'ema_shadow' in ckpt:
    for n, p in angle_model.named_parameters():
        if n in ckpt['ema_shadow']:
            p.data = ckpt['ema_shadow'][n].clone()

# ---- Dataloaders ----
val_ds  = CampusDataset(VAL_CSV, VAL_IMG_DIR, transform=val_tf, is_val=True)
test_ds = TestImageDataset(TEST_IMG_DIR, transform=val_tf)
val_loader  = DataLoader(val_ds,  batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

# ---- Predict Function ----
def predict_angles(angle_model, dataloader):
    preds = np.zeros(len(dataloader.dataset), dtype=np.float32)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            # Unpack img batch and index batch regardless of val vs test
            if len(batch) == 2:
                imgs, indices = batch               # test_loader: (img, idx)
            else:
                imgs, *_, indices = batch           # val_loader: (img, angle, sin, cos, idx)
            
            imgs = imgs.to(DEVICE)

            # Predict angles
            ang, _, _ = angle_model(imgs)

            # Store into preds array
            for i, idx in enumerate(indices):
                preds[idx] = ang[i].item()
    return preds

# ---- Run Prediction & Save ----
val_preds  = predict_angles(angle_model, val_loader)
test_preds = predict_angles(angle_model, test_loader)

# combine with correct indexing
all_preds = np.concatenate([val_preds, test_preds], axis=0)
df = pd.DataFrame({'id': np.arange(len(all_preds)), 'angle': all_preds})
df.to_csv(OUTPUT_CSV, index=False)

print(f" Saved predictions to {OUTPUT_CSV}")
print(f" Val (0–{len(val_preds)-1}): mean={val_preds.mean():.2f}, min={val_preds.min():.2f}, max={val_preds.max():.2f}")
print(f" Test ({len(val_preds)}–{len(all_preds)-1}): mean={test_preds.mean():.2f}, min={test_preds.min():.2f}, max={test_preds.max():.2f}")
