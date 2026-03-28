# Hybrid Deep Learning Model for Respiratory Disease Classification

## 📋 Overview

This project implements a **hybrid deep learning model** for automated classification of chest X-ray images into four disease categories: **COVID-19, Pneumonia, Tuberculosis (TB), and Normal**. The model combines the strengths of two state-of-the-art architectures—**EfficientNetB3** and **DenseNet121**—to achieve robust and accurate disease detection.

## 🎯 Problem Statement

Rapid and accurate diagnosis of respiratory diseases from chest X-ray images is critical in clinical settings. Manual interpretation by radiologists is time-consuming and prone to human error. This project addresses the need for an automated, reliable system that can assist in early disease detection and diagnosis.

## 🏗️ Model Architecture

### Hybrid Fusion Approach

The model leverages a **dual-backbone architecture** with feature fusion:

- **EfficientNetB3**: Provides lightweight, scalable feature extraction with excellent generalization
- **DenseNet121**: Ensures dense connectivity and improved gradient flow for better feature reuse

### Architecture Components

1. **Base Networks**: Pre-trained EfficientNetB3 and DenseNet121 extracting features independently
2. **Feature Fusion**: Embeddings from both networks are concatenated
3. **Classification Head**: Fully connected dense layers with:
   - Dropout regularization for preventing overfitting
   - Batch normalization for stable training
   - Softmax activation for 4-class classification

### Why Hybrid?

- **EfficientNet** brings parameter efficiency and strong generalization
- **DenseNet** ensures feature reuse and better gradient flow
- **Fusion** improves robustness compared to single-backbone models (e.g., ResNet50 alone)
- Combined approach leads to better class separability as shown in t-SNE visualizations

## 📊 Dataset

- **Total Samples**: 13,209 chest X-ray images
- **Data Split**:
  - Training: 8,791 samples
  - Validation: 2,434 samples
  - Test: 1,984 samples
  
- **Class Distribution**: Balanced across TB, COVID-19, Pneumonia, and Normal with optimized class weights
- **Data Source**: Preprocessed and split into balanced train/validation/test sets

## ⚙️ Training Details

### Hyperparameters
- **Optimizer**: AdamW with tuned learning rate
- **Loss Function**: Categorical cross-entropy
- **Batch Size**: Optimized for GPU memory
- **Callbacks**: 
  - EarlyStopping (prevent overfitting)
  - ModelCheckpoint (save best weights)
  - ReduceLROnPlateau (adaptive learning rate)

### Regularization
- Dropout layers to prevent overfitting
- Batch normalization for stable training
- L2 regularization on dense layers
- Class weights to handle residual class imbalance

## 📈 Performance Metrics

| Metric | Score |
|--------|-------|
| Training Accuracy | ~96% |
| Validation Accuracy | ~95% |
| **Test Accuracy** | **~95.5%** |
| Test Loss | ~0.83 |

## 🔍 Key Observations

### Class-wise Performance
- **COVID-19 vs. Normal**: Very clean separation with minimal overlap
- **Pneumonia**: Widest spread in t-SNE visualization, reflecting heterogeneous radiographic features
- **TB**: Partial overlap with Pneumonia, consistent with known clinical similarities
- **Normal**: Clearly distinct from disease classes

### Model Strengths
✅ High accuracy across all disease categories  
✅ Good generalization on unseen test data  
✅ Balanced performance across classes  
✅ Improved separability compared to single-backbone models  

## 🛠️ Dependencies

```
tensorflow
numpy
matplotlib
seaborn
scikit-learn
```

## 🚀 Usage

### Installation
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

### Running the Notebook

1. Open `Hybrid model.ipynb` in Jupyter or Google Colab
2. Mount your Google Drive (if using Colab):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Ensure your dataset is organized in the correct directory structure
4. Run all cells sequentially to:
   - Load and preprocess data
   - Build the hybrid model
   - Train on the dataset
   - Evaluate on test data
   - Generate visualizations

## 📁 Project Structure

```
├── Hybrid model.ipynb          # Main notebook with complete pipeline
├── README.md                   # This file
└── summary.txt                 # Detailed project summary
```

## 🎓 Technical Highlights

- **Feature Extraction**: Pre-trained backbone networks leverage transfer learning
- **Embedding Concatenation**: Combines complementary features from both architectures
- **Robust Training**: Multiple callbacks ensure optimal model generalization
- **Visualization**: t-SNE plots show class separability and model performance
- **Class Weighting**: Handles imbalanced datasets effectively

## 🚩 Future Improvements & Remaining Gaps

### Current Limitations
❌ Clinical validation: Needs external hospital datasets to confirm real-world generalization  
❌ Explainability: Would benefit from Grad-CAM or saliency maps for trust and interpretability  
❌ Multi-center validation: Required to prove robustness across different imaging conditions and equipment  

### Recommended Enhancements
- Integration of explainability techniques (Grad-CAM, attention maps)
- External validation on multi-center datasets
- Deployment as a REST API for clinical use
- Mobile app for on-site diagnosis assistance
- Ensemble methods to further improve accuracy

## 💡 Clinical Significance

This hybrid approach demonstrates the potential of deep learning in:
- **Early Detection**: Assisting radiologists in rapid disease identification
- **Triage**: Prioritizing patients who need immediate attention
- **Decision Support**: Providing a second opinion on radiological findings
- **Resource Optimization**: Reducing diagnostic time and improving efficiency

## 📝 Notes

- The model is trained on preprocessed chest X-ray images
- Data augmentation techniques are applied to improve model robustness
- GPU acceleration is recommended for training (supports both CPU and GPU)
- The notebook is compatible with Google Colab and local Jupyter environments

## 📧 Link For the trained model

https://drive.google.com/file/d/1WGAnx7-0QvOwXcWOUDgMGNk7wsQL6IPZ/view?usp=drive_link

---

**Project Type**: Deep Learning / Computer Vision / Medical Image Analysis  
**Framework**: TensorFlow/Keras  
**Status**: Completed with excellent performance metrics
