from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from torchvision import transforms, models
import torch
import torch.nn as nn
from PIL import Image
import io
import timm  # For Swin Transformer model
import pandas as pd
import math
# ---------- ConvNextClassifier Definition ----------
class ConvNextClassifier(nn.Module):
    def __init__(self, num_classes=15):
        super(ConvNextClassifier, self).__init__()
        self.convnext = models.convnext_small(pretrained=True)

        for name, param in self.convnext.named_parameters():
            if 'features.0.' in name or 'features.1.' in name or 'features.2.' in name:
                param.requires_grad = False

        if hasattr(self.convnext, 'head'):
            in_features = self.convnext.head.in_features
            self.convnext.head = nn.Identity()
        else:
            in_features = self.convnext.classifier[2].in_features if hasattr(self.convnext.classifier, '__getitem__') else self.convnext.classifier.in_features
            self.convnext.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.LayerNorm(in_features),
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
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.convnext(x)
        return self.classifier(features)

# ---------- Load Model ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
region_model = ConvNextClassifier(num_classes=15)
region_model.load_state_dict(torch.load("./Models/region_final.pth", map_location=device))
print("Model loaded successfully")
region_model = region_model.to(device)
region_model.eval()

# For lat/lon prediction
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

latlong_model = SwinGeoWithoutRegion()
latlong_model.load_state_dict(torch.load("./Models/latlong_only_images_final_22.pth", map_location=device))
latlong_model = latlong_model.to(device)
latlong_model.eval()
print("LatLong model loaded successfully")

# Load training CSV
df = pd.read_csv("./Models/cleaned_data_train.csv")  # Replace with the path to your actual CSV

# Compute mean and std for latitude and longitude
LAT_MEAN = df["latitude"].mean()
LAT_STD = df["latitude"].std()
LON_MEAN = df["longitude"].mean()
LON_STD = df["longitude"].std()


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

# Load Angle Model
angle_model = AngleRegressor()
ckpt = torch.load("./Models/angle_final_only_images_27.pt", map_location=device, weights_only=False)
angle_model.load_state_dict(ckpt['model_state_dict'])
angle_model.to(device).eval()
print("Angle model loaded successfully")
# ---------- FastAPI Setup ----------
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Image Transform ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------- Prediction Route ----------
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Region prediction
        region_output = region_model(input_tensor)
        pred_region_id = torch.argmax(region_output, dim=1).item() + 1

        # Lat-Long prediction
        pred_lat, pred_lon = latlong_model(input_tensor)
        latitude = pred_lat.item() * LAT_STD + LAT_MEAN
        longitude = pred_lon.item() * LON_STD + LON_MEAN

        # Angle prediction
        angle_deg, sin_val, cos_val = angle_model(input_tensor)
        angle = angle_deg.item()

    return {
        "Region_ID": pred_region_id,
        "latitude": round(latitude, 6),
        "longitude": round(longitude, 6),
        "angle": round(angle, 2)
    }
