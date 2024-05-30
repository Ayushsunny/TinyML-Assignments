#-----------------Assignment 1-----------------#

Multiclass Classification using Dense Neural Network. Please use the dataset available at the given link below

## Training a Neural Network on the Fashion MNIST Dataset

### Dataset from GitHub

To start, we clone the Fashion MNIST dataset from GitHub. This dataset contains 70,000 grayscale images of 28x28 pixels each, belonging to 10 different categories of clothing items.

```python
!git clone https://github.com/zalandoresearch/fashion-mnist.git
```

### Importing Necessary Libraries

We use TensorFlow for building and training the neural network, NumPy for data manipulation, and Matplotlib for visualization.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
```

### Loading the Fashion MNIST Dataset

We load the dataset using TensorFlow's built-in function and split it into training and test sets.

```python
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
```

### Normalizing the Input Data

The pixel values of the images are scaled to the range [0, 1] to help the model train faster and improve performance.

```python
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
```

### Reshaping the Input Data

We flatten each 28x28 image into a single 784-element vector, as the neural network's input layer expects a 1D array.

```python
X_train = X_train.reshape(-1, 28 * 28)
X_test = X_test.reshape(-1, 28 * 28)
```

### Defining the Model

We define a neural network with several dense layers and dropout layers to prevent overfitting. The output layer uses the softmax activation function to produce probability distributions for the 10 categories.

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(28 * 28,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### Compiling the Model

We compile the model using the Adam optimizer, sparse categorical cross-entropy loss function, and accuracy as a metric.

```python
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

### Training the Model

The model is trained for 50 epochs with a batch size of 64. We also validate the model using the test data.

```python
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), verbose=1)
```

### Evaluating the Model

After training, we evaluate the model's performance on the test set to get the accuracy and loss.

```python
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f'Test accuracy: {test_acc:.2f}')
print(f'Test loss: {test_loss:.2f}')
```

### Plotting Training and Validation Accuracy

We visualize the training and validation accuracy to understand how well the model is performing and to detect any overfitting.

```python
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

### Plotting Training and Validation Loss

Similarly, we plot the training and validation loss to monitor the model's learning process.

```python
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

This code trains a neural network on the Fashion MNIST dataset, evaluates its performance, and visualizes the training process to help understand the model's behavior.