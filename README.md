# Eye Disease Classifier using Vision Transformer (ViT)

A deep learning pipeline for multi-class classification of eye diseases using Vision Transformer (`google/vit-large-patch16-224`) and TensorFlow/Keras. The model is trained on retinal images to predict 10 different eye conditions.

## Objective

Develop an AI model that classifies retinal fundus images into common eye diseases, aiding early detection and diagnosis.

---

## Dataset Overview

- Loaded via: `tf.keras.utils.image_dataset_from_directory`
- Input image size: **224x224x3**
- Total classes: **10**
- Classes:
  - Central Serous Chorioretinopathy
  - Diabetic Retinopathy
  - Disc Edema
  - Glaucoma
  - Healthy
  - Macular Scar
  - Myopia
  - Pterygium
  - Retinal Detachment
  - Retinitis Pigmentosa

---

## Model Architecture

- **Backbone**: Pre-trained Vision Transformer (`TFViTModel`)
- **Preprocessing**:
  - Resize → Rescale → Channel Permute
- **Head**:
  - `Lambda` layer for ViT embedding extraction
  - `Dense` layer with `softmax` activation
- **Frozen ViT** during initial training

---

## Training Configuration

- Optimizer: `Adam`
- Loss: `SparseCategoricalCrossentropy`
- Epochs: `100`
- EarlyStopping: Enabled (patience=10)
- ReduceLROnPlateau: Enabled
- Batch Size: `16`
- Split:
  - Train: 70%
  - Validation: 15%
  - Test: 15%

---

## Performance Summary

| Metric         | Value        |
|----------------|--------------|
| Val Accuracy   | ~87.0%       |
| Test Accuracy | **86.7%**     |
| Top F1 Classes | Retinitis Pigmentosa, Disc Edema, Diabetic Retinopathy |
| Harder Classes | Glaucoma, Healthy, Macular Scar |

> **Note**: Excellent performance with scope for further improvement via fine-tuning, data balancing, or augmentation.

---

## Evaluation Metrics

```bash
Test Loss: 0.3512
Test Accuracy: 0.8668