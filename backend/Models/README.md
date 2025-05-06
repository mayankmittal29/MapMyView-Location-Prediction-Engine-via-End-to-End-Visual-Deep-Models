# 🌟 Statistical Methods in AI - Project 🌟

## Region Classification for IIITH Campus Images
### Mayank Mittal (2022101094)

---

## 🎯 Project Overview

This project tackles the challenge of classifying regions within the IIITH campus using machine and deep learning techniques. The model is trained to identify 15 distinct regions from images, achieving high accuracy through advanced deep learning approaches.

### Note:- 
## To see the logs for Region_ID go in notebook :- region_id_final_extra_attempts.ipynb in Version-1
## and for latitude-longitude and angle - in latlong-and-angle-final.ipynb with various attempts on all models 

## Link to model files:- https://iiithydstudents-my.sharepoint.com/:f:/g/personal/mayank_mittal_students_iiit_ac_in/EuLcrInRsAxLhqFEs3vr7bEBPvXN7PYUIu6dwqhmW5Gzaw?e=DsC0hX
# Part-1 :- Region_ID
### 🧠 Model Architecture & Approach

I implemented a **fine-tuned ConvNeXt Small** architecture with custom classification layers. This transformer-inspired CNN model was chosen for its superior feature extraction capabilities compared to traditional CNNs. Key implementation aspects include:

- **Transfer Learning**: Pre-trained weights on ImageNet were leveraged as a foundation, with early layers frozen to prevent overfitting
- **Advanced Classification Head**: Custom multi-layer classifier with LayerNorm, GELU activations, and strategic dropout for regularization
- **Mixed Precision Training**: Utilized automatic mixed precision to improve training speed and memory efficiency
- **Progressive Learning Rate Scheduling**: Implemented CosineAnnealingWarmRestarts scheduler to escape local minima

## 📊 Data Pre-processing & Augmentation

Extensive data augmentation pipelines were created to improve model generalization:

- **Training Augmentations**: RandomResizedCrop, flips (horizontal/vertical), rotations, color jittering, affine transforms, and random erasing
- **Validation and Test Pipeline**: Simple resize and normalization to ensure consistent evaluation
- **Dataset Integrity**: Implemented robust error handling to identify and exclude corrupt or invalid images

## 💡 Innovative Approaches

- **Adaptive Learning**: Implemented early stopping with increased patience for better convergence
- **Gradient Clipping**: Applied to prevent exploding gradients and stabilize training
- **Label Smoothing**: Enhanced cross-entropy loss with label smoothing (0.15) for better generalization
- **Thorough Validation**: Tracked both top-1 and top-3 accuracy metrics to better understand model performance
- **Memory Optimization**: Strategic garbage collection and CUDA cache clearing to handle large model training

## 🚀 Results

The model achieved excellent classification performance on the validation set with 96.75% accuracy, demonstrating its ability to distinguish between visually similar campus regions even under varying lighting and weather conditions.

---

# Part-2: Latitude and Longitude Prediction
### 🧠 Model Architecture & Approach

My solution employs a vision transformer architecture leveraging transfer learning to accurately predict geographic coordinates:

1. **Swin Transformer Backbone**: Using `swin_base_patch4_window7_224` pretrained model as the backbone, which offers excellent feature extraction capabilities with efficient attention mechanisms
2. **Location Regression**: Two parallel prediction heads for latitude and longitude prediction, allowing specialized focus on each geographic dimension

### 🔄 Pre-processing Techniques
- **Image Normalization**: Standard ImageNet mean/std normalization (0.485, 0.456, 0.406)/(0.229, 0.224, 0.225)
- **Resize**: All images standardized to 224×224 resolution
- **Coordinate Standardization**: Latitude and longitude values normalized using StandardScaler for improved training stability
- **Anomaly Filtering**: Specific image IDs (95, 145, 146, 158-161) identified as anomalies and excluded from validation

### 💡 Innovative Ideas
- **Hierarchical Feature Fusion**: Multi-stage feature processing with dimension reduction (1024→512) using LayerNorm, GELU activation and dropout for robust feature extraction
- **Specialized Prediction Heads**: Separate regression pathways for latitude and longitude coordinates with shared feature backbone
- **Fine-tuning Strategy**: Different learning rates for backbone (1e-5) and new layers (2e-4) to balance knowledge transfer and adaptation

## 🔍 Technical Implementation Details
### 🛠️ Training Optimizations
- **OneCycleLR Scheduler**: Advanced learning rate scheduling with separate max LRs (3e-5, 5e-4) for backbone and new layers
- **Mixed Precision Training**: FP16 computation via Automatic Mixed Precision (AMP) for faster training
- **AdamW Optimizer**: Weight decay (1e-2) for improved generalization
- **Checkpoint Saving**: Best model saved based on validation unscaled MSE

### 🔄 Data Augmentation Pipeline
- **Geometric Transforms**: Random horizontal flips (p=0.5), random affine transformations with rotation (±20°), translations (±15%), and scaling (0.85-1.15)
- **Color Augmentation**: ColorJitter with brightness, contrast, saturation adjustments (0.3)
- **Random Erasing**: Occlusion simulation (p=0.2) for robustness to partial obstruction

### 📊 Loss Function Design
- **Composite MSE Loss**: Combined Mean Squared Error for both latitude and longitude predictions
- **Unscaled Evaluation**: Final evaluation performed on original coordinate scale for real-world accuracy assessment

### 🔄 Fine-tuning Approach
- **Lower Learning Rates**: Reduced rates (5e-6 for backbone, 1e-4 for new layers) during fine-tuning
- **Cosine Annealing**: LR scheduler with warm restarts (T_0=5) for continued optimization
- **Reduced Weight Decay**: Slightly reduced L2 regularization (1e-3) for fine-tuning phase

## 📊 Results
The model achieves excellent geographic coordinate prediction with comprehensive coverage of both validation and test sets. The final unscaled MSE of around 22,000 demonstrates the effectiveness of this Swin Transformer-based approach for the complex task of image-based geolocation.

I'll create a technical summary for Part-3 based on the code you provided, following the same style and format as before.

# Part-3: Angle Prediction
### 🧠 Model Architecture & Approach

My solution employs a robust CNN-based circular regression approach leveraging transfer learning:

1. **EfficientNet Backbone**: Using `efficientnet_b0` pretrained model as the feature extractor, offering an excellent balance between performance and efficiency
2. **Circular Regression**: Predicting sine and cosine components of angles rather than direct angle values to handle the cyclic nature of orientations (0°-360°)

### 🔄 Pre-processing Techniques
- **Image Normalization**: Standard ImageNet mean/std normalization (0.485, 0.456, 0.406)/(0.229, 0.224, 0.225)
- **Resize**: All images standardized to 224×224 resolution
- **Angular Transformation**: Converting angles to radians, then to sine/cosine components for continuous learning across the 0°/360° boundary
- **Floating-Point Precision**: Ensuring all angular components use float32 precision for numerical stability

### 💡 Innovative Ideas
- **Circular Regression**: Predicting sin/cos components instead of direct angles to address the cyclic nature of orientation data
- **Unit Circle Normalization**: Enforcing the predicted sin/cos components to lie on the unit circle for consistent angle reconstruction
- **Mixup Augmentation with Angular Awareness**: Custom implementation handling angle mixing in sin/cos space to preserve circular properties
- **Exponential Moving Average**: Maintaining a temporal ensemble of model weights (decay=0.998) for improved stability and generalization

## 🔍 Technical Implementation Details
### 🛠️ Training Optimizations
- **OneCycleLR Scheduler**: Advanced learning rate scheduling with warm-up (10%) and cool-down phases for faster convergence
- **Mixed Precision Training**: FP16 computation via Automatic Mixed Precision (AMP) for faster training
- **Gradient Checkpointing**: Memory-efficient backpropagation enabling larger batch sizes (96) on limited VRAM
- **Gradient Clipping**: Norm-based gradient clipping (1.0) to prevent exploding gradients
- **AdamW Optimizer**: Weight decay (1e-4) for improved generalization

### 🔄 Data Augmentation Pipeline
- **Geometric Transforms**: Random horizontal flips, rotations (±15°), and affine translations (±10%)
- **Color Augmentation**: ColorJitter with brightness, contrast, saturation, and hue adjustments
- **Random Erasing**: Occlusion simulation (p=0.2) for robustness to partial obstruction
- **Mixup**: Input and target blending with alpha=0.2 applying specifically in the sin/cos domain for angles

### 📊 Loss Function Design
- **Composite Loss**: Balanced combination (0.5:0.5) of Mean Absolute Angular Error (MAAE) and circular MSE loss
- **MAAE Implementation**: Special handling to compute the minimum angular distance considering the circular nature of angles
- **Circular MSE**: Mean squared error between normalized sin/cos components ensuring proper learning of circular quantities

## 📊 Results & Inference
- **Model Performance**: Achieves excellent angle prediction of MAAE - 27 and 1/1+MAAE as 0.036 with extensive validation and testing
- **Inference Pipeline**: Efficient batch processing for both validation and test sets
- **EMA Application**: Using exponential moving average parameters during inference for better stability and generalization
- **Comprehensive Output**: Generates predictions across the full 0-360° spectrum with well-distributed values

The model effectively solves the complex task of orientation prediction from images without relying on region classification, demonstrating the power of circular regression techniques combined with modern CNN architectures.