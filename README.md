# COVID Image Classifier using Convolutional Neural Networks (CNN)

## Project Overview
This project utilizes a Convolutional Neural Network (CNN) to classify images into two categories: "COVID Positive" and "COVID Negative." The model is trained using a dataset of images stored in the `Images` directory. Using the TensorFlow and Keras libraries, the model was trained on labeled images, and a validation process was incorporated to ensure that the model generalizes well to unseen data.

The following steps were involved in the project:

- **Loading and Preprocessing Data:** Image data was loaded, resized, and normalized for training.
- **Model Creation:** A CNN model was built, trained, and evaluated for binary classification.
- **Model Evaluation:** Accuracy and loss curves were plotted to visualize the model's performance.
- **Model Saving:** After training, the model was saved for future use.
- **Prediction on New Images:** A function was created to classify new images based on the trained model.

## Libraries Used
- **NumPy**: Numerical operations
- **Matplotlib**: Data visualization (for plotting accuracy and loss graphs)
- **TensorFlow/Keras**: Deep learning framework used for creating, training, and evaluating the CNN model
- **os**: File system operations
- **ImageDataGenerator**: For real-time data augmentation and splitting the data into training and validation sets.

## Data Directory Structure
The dataset should be organized into directories within the `Images` folder. Each class (e.g., COVID Positive, COVID Negative) should have its own sub-directory containing images. An example structure looks like this:


## Steps in the Project

### 1. Data Loading, Preprocessing, and Splitting
We used the `ImageDataGenerator` class from Keras for data augmentation and rescaling the pixel values. The data was split into training (80%) and validation (20%) sets. The images were resized to 150x150 pixels to ensure consistent input dimensions for the CNN model.

### 2. Building the CNN Model
The model is built using the following architecture:
- **Conv2D Layer**: 3 convolutional layers to extract spatial features.
- **MaxPooling2D Layer**: For reducing spatial dimensions after each convolution.
- **Flatten Layer**: To flatten the 2D feature maps into a 1D vector.
- **Dense Layer**: Fully connected layer for classification.
- **Dropout**: To avoid overfitting by randomly deactivating some neurons during training.
- **Output Layer**: A sigmoid activation for binary classification (COVID Positive/Negative).

### 3. Compiling and Training the Model
The model is compiled with the Adam optimizer and binary cross-entropy loss function. Early stopping is implemented to prevent overfitting by monitoring the validation loss. The model was trained for up to 30 epochs.

### 4. Evaluation: Accuracy and Loss Graphs
After training, the model's performance is visualized by plotting the training and validation accuracy, as well as the training and validation loss over the epochs. This helps to assess how well the model is learning and whether it’s overfitting.

### 5. Saving the Model
The trained model was saved for later use using the `model.save()` function, so it can be easily loaded for future predictions without needing to retrain.

### 6. Classifying a New Image
A function `classify_image` was created to classify new images based on the trained model. The image is loaded, preprocessed, and then passed through the model to predict its class (COVID Positive or Negative).

## Conclusion
This project demonstrates the use of deep learning, particularly CNNs, to solve a binary image classification problem. The model was able to classify images into two categories: COVID Positive and COVID Negative. The model's performance was monitored using accuracy and loss plots, and early stopping was implemented to avoid overfitting.

Future work could involve further fine-tuning the model, improving the dataset (for example, by including more diverse images), or exploring advanced architectures for improved accuracy.

## Requirements
- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib

To install the required libraries, you can use:

```bash
pip install tensorflow numpy matplotlib
