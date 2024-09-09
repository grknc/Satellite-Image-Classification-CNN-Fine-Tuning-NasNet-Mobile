# Satellite Image Classification | CNN | Fine-Tuning NasNet Mobile

## Project Overview

![image](https://github.com/user-attachments/assets/5209afde-0934-4527-b8fb-dd317a8794d5)

This project focuses on the classification of satellite images into categories such as "Cloudy", "Desert", "Green_Area", and "Water". The dataset is a collection of satellite images, each sized 256x256 pixels, sourced from Kaggle. The primary goal is to classify these images using deep learning techniques, achieving a high accuracy through fine-tuning pre-trained models.

### Key Highlights:
- **Dataset**: Satellite images in various categories, prepared for image classification tasks.
- **Model Architecture**: The model utilizes the **NasNet Mobile** architecture, fine-tuned to achieve **95% accuracy**.
- **Image Preprocessing**: Data augmentation and preprocessing techniques are applied to improve model robustness.
- **Libraries**: The project is implemented using **Keras** and **TensorFlow**.

## Libraries Used

```python
import keras
import tensorflow as tf
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, metrics
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models  import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

## Key Components

1. **Image Loading and Preprocessing**: 
   - Satellite images are loaded from the dataset directory and preprocessed using various augmentation techniques to ensure model generalization.

2. **Model Architecture**:
   - The **NasNet Mobile** model is fine-tuned for satellite image classification.
   - The model is optimized using early stopping and model checkpointing for efficient training.

3. **Training**:
   - The model is trained with the use of **ImageDataGenerator** for augmenting the training data, making it more robust against overfitting.

4. **Evaluation**:
   - The final model achieves **95% accuracy** on the validation dataset, showing its effectiveness in satellite image classification tasks.

## Usage

To run the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare the dataset:
   - Download the satellite image classification dataset from Kaggle and place it in the appropriate folder.

4. Train the model:
   - Run the Jupyter notebook to train and fine-tune the model.

## Results

The project successfully classifies satellite images into their respective categories with high accuracy. Fine-tuning **NasNet Mobile** significantly improved performance, achieving **95% accuracy**.
