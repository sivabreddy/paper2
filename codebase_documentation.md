# Prostate MRI Analysis System - Codebase Documentation

This is a comprehensive medical image analysis system designed for prostate cancer detection using MRI scans. The system implements multiple deep learning models and compares their performance for classification tasks.

## Project Structure

The codebase is organized into several modules, each implementing a different deep learning approach:

1. **Main/** - Core application components
   - GUI.py: Tkinter-based graphical user interface
   - Run.py: Main execution coordinator that runs all models
   - Pre_processing.py: Image preprocessing pipeline including filtering and segmentation
   - read.py: Data loading functionality
   - Various utility files for feature extraction

2. **Model Implementations**:
   - **DCNN/** - Deep Convolutional Neural Network
   - **ResNet/** - Residual Network implementation
   - **Focal_Net/** - Focal Network with CNN architecture
   - **Panoptic_model/** - Panoptic segmentation approach using ResNet50
   - **HFGSO_DRN/** - Hybrid Flower Germination Search Algorithm optimized Deep Residual Network
   - **prop_DMO/** - Proposed Deep Maxout network with LHFGSO optimization

## Workflow

The system follows this processing pipeline:

1. **Data Preprocessing**:
   - ROI (Region of Interest) extraction from MRI images
   - Adaptive Median Filtering (AMF) for noise reduction
   - Segmentation using SegNet architecture
   - Data augmentation (rotation, cropping, flipping, random erasing)

2. **Feature Extraction**:
   - Statistical features (mean, variance, kurtosis, skewness)
   - Texture features (Local Binary Pattern, SLBT)
   - Entropy-based features

3. **Classification**:
   Six different models are implemented and compared:
   - Standard DCNN
   - ResNet
   - Focal-Net
   - Panoptic model
   - HFGSO-DRN (Hybrid Flower Germination Search Algorithm optimized Deep Residual Network)
   - LHFGSO-DMO (Large-scale Hybrid Flower Germination Search Algorithm optimized Deep Maxout)

4. **Evaluation**:
   - Performance metrics: Accuracy, Sensitivity, Specificity
   - Comparative visualization of all models
   - Results displayed in the GUI

## Key Technical Components

- **Optimization Algorithms**: The system implements custom optimization algorithms (HFGSO, LHFGSO) to improve model weights
- **Custom Layers**: Includes a custom Maxout layer implementation
- **Ensemble Approach**: Combines multiple deep learning architectures for robust classification
- **Medical Focus**: Specifically designed for prostate MRI analysis with appropriate preprocessing techniques

## Usage

To run the system:
1. Set up Python environment with dependencies from requirements.txt
2. Execute Main/GUI.py to launch the interface
3. Select dataset and training parameters
4. View comparative performance metrics across all models

The system is designed for binary classification of prostate MRI scans (benign vs. malignant) and provides a comprehensive comparison of different deep learning approaches for this medical imaging task.