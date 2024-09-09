# Satellite Image Classification | CNN | Fine-Tuning NasNet Mobile

## Project Overview

![image](https://github.com/user-attachments/assets/5209afde-0934-4527-b8fb-dd317a8794d5)

This project focuses on the classification of satellite images into categories such as "Cloudy", "Desert", "Green_Area", and "Water". The dataset is a collection of satellite images, each sized 256x256 pixels, sourced from Kaggle. The primary goal is to classify these images using deep learning techniques, achieving a high accuracy through fine-tuning pre-trained models.

The core of the project is located in the src folder, which contains the main files, including:
- satellite_image_classification.ipynb: This Jupyter notebook contains the entire workflow, from data preprocessing to model training and evaluation.

By utilizing convolutional neural networks (CNN) and transfer learning with NasNet Mobile, the model achieves a classification accuracy of 95%. This setup demonstrates the effectiveness of deep learning techniques for satellite image classification tasks

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

To run this project on your local machine, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   ```

2. **Navigate to the `src` folder**:
   The `src` folder contains the Jupyter notebook (`satellite_image_classification.ipynb`) and the necessary scripts for running the project.
   ```bash
   cd src
   ```

3. **Install the required dependencies**:
   Before running the notebook, install all required packages listed in `requirements.txt`.
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the dataset**:
   - Download the satellite image classification dataset from Kaggle.
   - Place the dataset in the appropriate folder as referenced in the notebook.

5. **Run the Jupyter notebook**:
   Execute the notebook to preprocess the data, train the model, and evaluate its performance.
   ```bash
   jupyter notebook satellite_image_classification.ipynb
   ```

## Results

The project successfully classifies satellite images into their respective categories with high accuracy. Fine-tuning **NasNet Mobile** significantly improved performance, achieving **95% accuracy**.
