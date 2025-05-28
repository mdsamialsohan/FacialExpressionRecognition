# Facial Expression Recognition Using CNN

This project implements a Convolutional Neural Network (CNN) model to recognize facial expressions from grayscale images. The model is trained on a dataset of facial images categorized into 7 emotion classes.

---

## Features

- Preprocesses image data using Keras' `ImageDataGenerator` for training and validation.
- Builds a CNN with multiple convolutional, max pooling, dropout, and dense layers.
- Trains the model for 25 epochs.
- Saves the model architecture as a JSON file and model weights as an H5 file.

---

## Dependencies

- Python 3.x
- OpenCV (`cv2`)
- TensorFlow / Keras
- numpy (usually installed with Keras/TensorFlow)

Install dependencies with:

```bash
pip install tensorflow keras opencv-python
```

## Model Architecture

  - Input: 48x48 grayscale images

  - Convolutional layers: 32, 64, 128 filters with ReLU activation

  - MaxPooling layers to reduce spatial dimensions

  - Dropout layers for regularization

  - Fully connected dense layer with 1024 units

  - Output layer with 7 units (for 7 emotion classes) and softmax activation

## Training Details

 - Optimizer: Adam with learning rate 0.0001 and decay 1e-6

 - Loss function: Categorical cross-entropy

 - Batch size: 64

 - Epochs: 25

 - Steps per epoch: 28709 // 64 (adjust based on your dataset)

 - Validation steps: 7178 // 64 (adjust based on your dataset)

## Notes

 - Ensure your dataset is properly labeled and structured.

 - You can adjust hyperparameters (epochs, batch size, learning rate) to improve performance.

 - The model currently uses grayscale images; for color images, modify the input shape and preprocessing.

## License

This project is open source and free to use.

If you want me to help you with adding inference code or deployment instructions, just ask!
