# Image Classification using CNN and Pre-trained Models

## Overview
This project implements image classification for six emotions using Convolutional Neural Networks (CNNs) and a pre-trained EfficientNetB0 model. The dataset is preprocessed, and multiple models are trained and evaluated to compare their performance.

## Project Structure
```plaintext
├── 6 Emotions for image classification/  # Dataset directory
│   ├── emotion_1/
│   ├── emotion_2/
│   ├── ...
├── test images/  # Test images for prediction
├── cnn1_model.keras  # First CNN model
├── cnn2_model.keras  # Second CNN model
├── pre_trained_model.keras  # Pre-trained model
├── model_architecture.png  # CNN-1 model architecture
├── model_architecture2.png  # CNN-2 model architecture
├── model_architecture3.png  # Pre-trained model architecture
├── train.py  # Python script to train and evaluate models
├── README.md  # Project documentation
```

## Dependencies
This project requires the following dependencies:
```python
pip install tensorflow numpy pandas matplotlib scikit-learn
```

## Dataset Preparation
Ensure the dataset follows this directory structure:
```plaintext
6 Emotions for image classification/
├── emotion_1/
│   ├── image1.jpeg
│   ├── image2.jpeg
│   └── ...
├── emotion_2/
│   ├── image1.jpeg
│   ├── image2.jpeg
│   └── ...
...
```
Corrupt files are removed before training by checking for valid JPEG headers.

## Model Architectures
Three different models are trained and compared:
1. **CNN-1**: A simple CNN model with two convolutional layers and max pooling.
2. **CNN-2**: A deeper CNN model with larger filters and pooling layers.
3. **Pre-trained Model**: EfficientNetB0 with a custom classification head.

## Training
Run the following command to train the models:
```sh
python train.py
```
The models are trained for 20 epochs each, with categorical crossentropy loss and accuracy metrics.

## Evaluation
Each model's test accuracy is computed and displayed at the end of training. The results are summarized in a table.

## Visualization
Loss and accuracy trends over epochs are visualized using Matplotlib.

## Predictions
The trained models are used to predict the class of new images. The predictions are displayed alongside the true labels.

## Model Architecture Diagrams
The architectures of all three models are visualized using `plot_model` from TensorFlow.

## Results
The test accuracy of different models is compared:
```plaintext
| Model        | Test Accuracy |
|-------------|--------------|
| CNN-1       | 0.432        |
| CNN-2       | 0.467        |
| Pre-trained | 0.678        |
```

## Conclusion
- CNN-2 performed better than CNN-1 due to deeper layers and larger filter sizes.
- The pre-trained EfficientNetB0 model provided the best accuracy by leveraging transfer learning.

## Future Work
- Fine-tuning the pre-trained model for better accuracy.
- Exploring data augmentation techniques for improved generalization.
- Implementing other deep learning models like ResNet or MobileNet.

## Author
Souvik Roy

