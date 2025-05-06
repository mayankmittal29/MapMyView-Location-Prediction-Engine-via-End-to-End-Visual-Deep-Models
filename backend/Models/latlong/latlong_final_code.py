import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import timm
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import gc
from collections import Counter

# --------------------- Utility Functions ---------------------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------- Paths ---------------------
TRAIN_IMG_DIR  = '/kaggle/input/iiith-images-latlong-smai/images_train/images_train/images_train'
VALID_IMG_DIR  = '/kaggle/input/iiith-images-latlong-smai/images_val/images_val'
TEST_IMG_DIR   = '/kaggle/input/iiith-images-latlong-smai/images_test/images_test'
TRAIN_LABELS   = '/kaggle/input/iiith-images-latlong-smai/cleaned_data_train.csv'
VALID_LABELS   = '/kaggle/input/iiith-images-latlong-smai/labels_val_updated.csv'
OUTPUT_CSV     = 'predictions_predict2.csv'
ANOMALIES      = [95,145,146,158,159,160,161]

# --------------------- Read DataFrames ---------------------
train_df = pd.read_csv(TRAIN_LABELS)
valid_df = pd.read_csv(VALID_LABELS)
valid_df['image_id'] = valid_df['filename'].apply(lambda x: int(x.split('_')[1].split('.')[0].lstrip('0') or '0'))
valid_df = valid_df[~valid_df['image_id'].isin(ANOMALIES)].reset_index(drop=True)

# --------------------- Debug: Check Dataset Sizes ---------------------
print(f"Original valid_df size: {len(valid_df)}")

# --------------------- Debug: Check file extensions ---------------------
def get_file_extensions(directory):
    extensions = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            ext = os.path.splitext(filename)[1].lower()
            extensions.append(ext)
    return Counter(extensions)

# Test if directories exist
print(f"VALID_IMG_DIR exists: {os.path.exists(VALID_IMG_DIR)}")
print(f"TEST_IMG_DIR exists: {os.path.exists(TEST_IMG_DIR)}")

# Only check extensions if directories exist
if os.path.exists(VALID_IMG_DIR):
    print(f"Valid image extensions: {get_file_extensions(VALID_IMG_DIR)}")
if os.path.exists(TEST_IMG_DIR):
    print(f"Test image extensions: {get_file_extensions(TEST_IMG_DIR)}")
    
def count_files_in_folder(folder_path):
    return sum(1 for entry in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, entry)))

# Example usage
print(f"Number of files in '{TEST_IMG_DIR}': {count_files_in_folder(TEST_IMG_DIR)}")

# --------------------- Scaling ---------------------
lat_scaler = StandardScaler().fit(train_df[['latitude']])
long_scaler = StandardScaler().fit(train_df[['longitude']])
for df in [train_df, valid_df]:
    df['scaled_lat'] = lat_scaler.transform(df[['latitude']])
    df['scaled_lon'] = long_scaler.transform(df[['longitude']])

# --------------------- Transforms ---------------------
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.3,0.3,0.3,0.05),
    transforms.RandomAffine(20, translate=(0.15,0.15), scale=(0.85,1.15)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    transforms.RandomErasing(0.2, scale=(0.02,0.15), ratio=(0.3,3.3))
])
val_test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# --------------------- Dataset ---------------------
class GeoDataset(Dataset):
    def __init__(self, img_dir, df, transform=None, is_test=False):
        self.img_dir = img_dir
        self.df = df.copy()
        self.transform = transform
        self.is_test = is_test
        self.df = self.df[self.df['filename'].apply(lambda fn: os.path.exists(os.path.join(img_dir, fn)))].reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, row['filename'])).convert('RGB')
        x = self.transform(img)
        if self.is_test:
            return {'image': x, 'image_id': int(row['filename'].split('_')[1].split('.')[0].lstrip('0') or '0')}
        return {
            'image': x,
            'scaled_lat': torch.tensor(row['scaled_lat'], dtype=torch.float32),
            'scaled_lon': torch.tensor(row['scaled_lon'], dtype=torch.float32),
            'latitude': torch.tensor(row['latitude'], dtype=torch.float32),
            'longitude': torch.tensor(row['longitude'], dtype=torch.float32)
        }

# --------------------- DataLoaders ---------------------
batch_size = 16
train_loader = DataLoader(GeoDataset(TRAIN_IMG_DIR, train_df, train_transform), batch_size, shuffle=True, num_workers=4, pin_memory=True)
valid_loader = DataLoader(GeoDataset(VALID_IMG_DIR, valid_df, val_test_transform), 32, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(GeoDataset(TEST_IMG_DIR, valid_df, val_test_transform, is_test=True), 32, shuffle=False, num_workers=4, pin_memory=True)

# --------------------- Model ---------------------
class SwinGeoWithoutRegion(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0, global_pool='avg')
        feat = self.backbone.num_features
        self.fuse = nn.Sequential(
            nn.Linear(feat, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.2)
        )
        self.lat_head = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))
        self.lon_head = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))
    def forward(self, x):
        feats = self.backbone(x)
        h = self.fuse(feats)
        return self.lat_head(h).squeeze(-1), self.lon_head(h).squeeze(-1)

model = SwinGeoWithoutRegion().to(device)

# --------------------- Loss, Optimizer & Scheduler ---------------------
class GeoLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, pred_lat, pred_lon, true_lat, true_lon):
        return self.mse(pred_lat, true_lat) + self.mse(pred_lon, true_lon)

criterion = GeoLoss()
params_backbone, params_new = [], []
for name, param in model.named_parameters():
    if 'backbone' in name:
        params_backbone.append(param)
    else:
        params_new.append(param)

optimizer = optim.AdamW([
    {'params': params_backbone, 'lr': 1e-5},
    {'params': params_new,      'lr': 2e-4}
], weight_decay=1e-2)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[3e-5,5e-4], steps_per_epoch=len(train_loader), epochs=30)

# --------------------- Training & Evaluation ---------------------
def train_eval():
    best_mse = float('inf')
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    history = {'train_loss': [], 'train_lat_loss': [], 'train_lon_loss': [],
               'val_loss': [], 'val_lat_loss': [], 'val_lon_loss': [],
               'val_unscaled_mse': [], 'lr': []}

    for epoch in range(1, 31):
        model.train()
        running_loss = running_lat = running_lon = 0.0
        count = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train")
        for batch in pbar:
            imgs = batch['image'].to(device)
            lat_s = batch['scaled_lat'].to(device)
            lon_s = batch['scaled_lon'].to(device)
            
            # Forward pass without region ID
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                p_lat, p_lon = model(imgs)
                loss = criterion(p_lat, p_lon, lat_s, lon_s)
                lat_loss = nn.MSELoss()(p_lat, lat_s)
                lon_loss = nn.MSELoss()(p_lon, lon_s)
            
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()

            bs = imgs.size(0)
            running_loss += loss.item() * bs
            running_lat += lat_loss.item() * bs
            running_lon += lon_loss.item() * bs
            count += bs
            lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': loss.item(), 'lat': lat_loss.item(), 'lon': lon_loss.item(), 'lr': lr})

        epoch_loss = running_loss / count
        epoch_lat = running_lat / count
        epoch_lon = running_lon / count
        history['train_loss'].append(epoch_loss)
        history['train_lat_loss'].append(epoch_lat)
        history['train_lon_loss'].append(epoch_lon)
        history['lr'].append(lr)
        print(f"Epoch {epoch} TRAIN -> Loss: {epoch_loss:.6f}, Lat: {epoch_lat:.6f}, Lon: {epoch_lon:.6f}, LR: {lr:.6e}")

        model.eval()
        v_loss = v_lat = v_lon = 0.0
        v_count = 0
        all_preds, all_true = [], []
        vbar = tqdm(valid_loader, desc=f"Epoch {epoch} Val")
        with torch.no_grad():
            for batch in vbar:
                imgs = batch['image'].to(device)
                
                # Forward pass without region ID
                p_lat, p_lon = model(imgs)
                lat_s = batch['scaled_lat'].to(device)
                lon_s = batch['scaled_lon'].to(device)

                loss = criterion(p_lat, p_lon, lat_s, lon_s)
                lat_loss = nn.MSELoss()(p_lat, lat_s)
                lon_loss = nn.MSELoss()(p_lon, lon_s)

                ulat = lat_scaler.inverse_transform(p_lat.cpu().numpy().reshape(-1,1)).flatten()
                ulon = long_scaler.inverse_transform(p_lon.cpu().numpy().reshape(-1,1)).flatten()
                all_preds.append(np.vstack([ulat, ulon]).T)
                tr_lat = batch['latitude'].numpy()
                tr_lon = batch['longitude'].numpy()
                all_true.append(np.vstack([tr_lat, tr_lon]).T)

                bs = imgs.size(0)
                v_loss += loss.item() * bs
                v_lat += lat_loss.item() * bs
                v_lon += lon_loss.item() * bs
                v_count += bs
                vbar.set_postfix({'v_loss': loss.item(), 'v_lat': lat_loss.item(), 'v_lon': lon_loss.item()})

        val_loss = v_loss / v_count
        val_lat = v_lat / v_count
        val_lon = v_lon / v_count
        preds = np.concatenate(all_preds)
        true  = np.concatenate(all_true)
        unscaled_mse = ((preds - true)**2).mean()

        history['val_loss'].append(val_loss)
        history['val_lat_loss'].append(val_lat)
        history['val_lon_loss'].append(val_lon)
        history['val_unscaled_mse'].append(unscaled_mse)

        print(f"Epoch {epoch} VAL   -> Loss: {val_loss:.6f}, Lat: {val_lat:.6f}, Lon: {val_lon:.6f}, Unscaled MSE: {unscaled_mse:.6f}")

        if unscaled_mse < best_mse:
            best_mse = unscaled_mse
            torch.save(model.state_dict(), 'best_geo2.pth')

        torch.cuda.empty_cache(); gc.collect()

    model.load_state_dict(torch.load('best_geo2.pth'))
    return history

def generate_csv(best_mse):
    rows = []
    model.eval()
    
    # Create a list to track processed IDs to avoid duplicates
    processed_ids = set()
    
    print("\n--- Starting CSV Generation ---")
    print(f"Valid loader dataset size: {len(valid_loader.dataset)}")
    
    # Process validation data with more detailed tracking
    valid_count = 0
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Processing validation data"):
            imgs = batch['image'].to(device)
            
            # Forward pass without region ID
            p_lat, p_lon = model(imgs)
            ulat = lat_scaler.inverse_transform(p_lat.cpu().numpy().reshape(-1,1)).flatten()
            ulon = long_scaler.inverse_transform(p_lon.cpu().numpy().reshape(-1,1)).flatten()
            
            # Process each image in the batch
            batch_size = imgs.size(0)
            for i in range(batch_size):
                if i + valid_count >= len(valid_loader.dataset):
                    continue
                    
                # Get the image ID
                image_id = valid_loader.dataset.df.iloc[valid_count + i]['image_id']
                
                # Check if we've already processed this ID
                if image_id in processed_ids:
                    print(f"Warning: Duplicate ID {image_id} in validation data")
                    continue
                    
                processed_ids.add(image_id)
                rows.append({
                    'id': image_id, 
                    'Latitude': ulat[i], 
                    'Longitude': ulon[i]
                })
            
            valid_count += batch_size
    
    print(f"Processed {valid_count} validation images, added {len(processed_ids)} unique IDs")
    
    # Process test data (IDs 369–737)
    test_start_id = 369
    test_end_id = 737
    test_count = 0
    
    # Check if test directory exists and list files
    if os.path.exists(TEST_IMG_DIR):
        test_files = sorted(os.listdir(TEST_IMG_DIR))
        print(f"Found {len(test_files)} files in test directory")
        
        # Create a list to hold test file information
        test_file_list = []
        
        # Check if files have 'img_' prefix
        if test_files and any(fn.startswith('img_') for fn in test_files):
            print("Detected 'img_' prefix in test files")
            
            # Map test files to test IDs sequentially
            for i, filename in enumerate(test_files):
                if i < (test_end_id - test_start_id + 1):  # Ensure we don't exceed the test ID range
                    test_id = test_start_id + i
                    test_file_list.append({
                        'filename': filename,
                        'image_id': test_id
                    })
            
            print(f"Created mapping for {len(test_file_list)} test files")
            if len(test_file_list) > 0:
                print(f"Sample mapping: {test_file_list[:3]}")
    else:
        print(f"Test directory {TEST_IMG_DIR} not found")
        test_file_list = []
    
    # Process test images if we have any
    if test_file_list:
        # Define a dataset for test images
        class TestImageDataset(Dataset):
            def __init__(self, img_dir, file_list, transform):
                self.img_dir = img_dir
                self.file_list = file_list
                self.transform = transform
                
            def __len__(self):
                return len(self.file_list)
                
            def __getitem__(self, idx):
                file_info = self.file_list[idx]
                img_path = os.path.join(self.img_dir, file_info['filename'])
                img = Image.open(img_path).convert('RGB')
                return {
                    'image': self.transform(img),
                    'image_id': file_info['image_id']
                }
        
        # Create test dataset and dataloader
        test_dataset = TestImageDataset(TEST_IMG_DIR, test_file_list, val_test_transform)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        print(f"Created test dataloader with {len(test_dataset)} images")
        
        # Process test data
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Processing test data"):
                imgs = batch['image'].to(device)
                image_ids = batch['image_id'].tolist()  # Convert tensor to list
                
                # Forward pass without region ID
                p_lat, p_lon = model(imgs)
                ulat = lat_scaler.inverse_transform(p_lat.cpu().numpy().reshape(-1,1)).flatten()
                ulon = long_scaler.inverse_transform(p_lon.cpu().numpy().reshape(-1,1)).flatten()
                
                for i in range(len(image_ids)):
                    img_id = image_ids[i]
                    
                    # Check if we've already processed this ID
                    if img_id in processed_ids:
                        print(f"Warning: Duplicate ID {img_id} in test data")
                        continue
                        
                    processed_ids.add(img_id)
                    rows.append({
                        'id': img_id, 
                        'Latitude': ulat[i], 
                        'Longitude': ulon[i]
                    })
                    test_count += 1
        
        print(f"Processed {test_count} test images")
    
    # Handle any missing test IDs (fill with zeros or nearest neighbor)
    missing_test_ids = set(range(test_start_id, test_end_id + 1)) - processed_ids
    if missing_test_ids:
        print(f"Warning: {len(missing_test_ids)} test IDs missing. Adding placeholders.")
        for img_id in missing_test_ids:
            rows.append({
                'id': img_id,
                'Latitude': 0.0,  # Use a default or interpolate from nearest neighbors
                'Longitude': 0.0
            })
    
    # Save CSV
    result_df = pd.DataFrame(rows).sort_values('id')
    result_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\nCSV saved with {len(result_df)} rows:")
    print(f"- {len(processed_ids.intersection(set(range(0, test_start_id))))} validation entries")
    print(f"- {len(processed_ids.intersection(set(range(test_start_id, test_end_id + 1))))} test entries")
    print(f"- {len(missing_test_ids)} placeholder entries for missing test IDs")
    
    # Report best MSE
    if isinstance(best_mse, dict):
        best_mse_value = best_mse.get('val_unscaled_mse', [-1])[-1] if 'val_unscaled_mse' in best_mse else "N/A"
        print(f"Best MSE: {best_mse_value}")
    else:
        print(f"Best MSE: {best_mse:.6f}")
            
# --------------------- Main ---------------------
best_mse = train_eval()
generate_csv(best_mse)

print("Training complete.")


## Fine-tuning the model






# Improved retraining code with better LR scheduling
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

# Load the previously trained model
model.load_state_dict(torch.load('/kaggle/working/best_geo2.pth'))

# Reset optimizer with lower learning rates for fine-tuning
optimizer = optim.AdamW([
    {'params': params_backbone, 'lr': 5e-6},  # Lower LR for backbone
    {'params': params_new, 'lr': 1e-4}        # Lower LR for new layers
], weight_decay=1e-3)  # Slightly reduced weight decay

# Better LR scheduler - Cosine annealing with warm restarts
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-7)
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
# Alternative: ReduceLROnPlateau (uncomment to use instead)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-7, verbose=True)

# Number of additional training epochs
additional_epochs = 20
best_mse = float('inf')

# Training loop
for epoch in range(1, additional_epochs+1):
    model.train()
    running_loss = running_lat = running_lon = 0.0
    count = 0
    
    for batch in tqdm(train_loader, desc=f"Retrain Epoch {epoch}"):
        imgs = batch['image'].to(device)
        lat_s = batch['scaled_lat'].to(device)
        lon_s = batch['scaled_lon'].to(device)
        
        # Forward pass
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            p_lat, p_lon = model(imgs)
            loss = criterion(p_lat, p_lon, lat_s, lon_s)
        
        optimizer.zero_grad()
        if torch.cuda.is_available():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        bs = imgs.size(0)
        running_loss += loss.item() * bs
        count += bs
    
    epoch_loss = running_loss / count
    print(f"Retrain Epoch {epoch} Train Loss: {epoch_loss:.6f}")
    
    # Evaluate
    model.eval()
    v_loss = 0.0
    v_count = 0
    all_preds, all_true = [], []
    
    with torch.no_grad():
        for batch in valid_loader:
            imgs = batch['image'].to(device)
            lat_s = batch['scaled_lat'].to(device)
            lon_s = batch['scaled_lon'].to(device)
            
            p_lat, p_lon = model(imgs)
            loss = criterion(p_lat, p_lon, lat_s, lon_s)
            
            # Calculate unscaled predictions for MSE
            ulat = lat_scaler.inverse_transform(p_lat.cpu().numpy().reshape(-1,1)).flatten()
            ulon = long_scaler.inverse_transform(p_lon.cpu().numpy().reshape(-1,1)).flatten()
            all_preds.append(np.vstack([ulat, ulon]).T)
            tr_lat = batch['latitude'].numpy()
            tr_lon = batch['longitude'].numpy()
            all_true.append(np.vstack([tr_lat, tr_lon]).T)
            
            bs = imgs.size(0)
            v_loss += loss.item() * bs
            v_count += bs
    
    val_loss = v_loss / v_count
    preds = np.concatenate(all_preds)
    true = np.concatenate(all_true)
    unscaled_mse = ((preds - true)**2).mean()
    
    print(f"Retrain Epoch {epoch} Val Loss: {val_loss:.6f}, Unscaled MSE: {unscaled_mse:.6f}")
    
    # Update scheduler (use this for ReduceLROnPlateau)
    # scheduler.step(unscaled_mse)
    
    # Update scheduler (use this for CosineAnnealingWarmRestarts)
    scheduler.step()
    
    # Save best model
    if unscaled_mse < best_mse:
        best_mse = unscaled_mse
        torch.save(model.state_dict(), 'best_retrained_geo2.pth')
        print(f"New best model saved with MSE: {best_mse:.6f}")
    
    torch.cuda.empty_cache()
    gc.collect()

# Load best retrained model and generate predictions
model.load_state_dict(torch.load('best_retrained_geo2.pth'))
generate_csv(best_mse)

print(f"Retraining complete. Best MSE: {best_mse:.6f}")