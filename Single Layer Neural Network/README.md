#-----------------Assignment 1-----------------#

Implement a Regression using Single-Layer Neural Network. Take two set x and y and trained the model to find out the correct y=function of x. Already we discussed these in our previous lectures.


## Overview

This repository contains two Jupyter notebooks that demonstrate the development and improvement of a single-layer neural network for regression tasks using TensorFlow and Python. The first notebook, `Single-Layer Neural Network.ipynb`, introduces a simple single-layer model, while the second notebook, `Fine Tuned Single-Layer Neural Network.ipynb`, fine-tunes the model for better performance.

## Single-Layer Neural Network

### Description

In the `Single-Layer Neural Network.ipynb` notebook, a basic neural network model with a single dense layer is created to perform linear regression on user-provided data. The model is trained using the stochastic gradient descent (SGD) optimizer and the mean squared error loss function. After training, the model's predictions are compared to the actual values, and the training loss is visualized.

### Key Steps

1. **User Input**: Users provide x and y values as input.
2. **Data Conversion**: The input strings are converted to NumPy arrays and reshaped for TensorFlow.
3. **Model Definition**: A simple neural network with one dense layer is defined.
4. **Model Compilation**: The model is compiled with the SGD optimizer and mean squared error loss function.
5. **Model Training**: The model is trained for 1000 epochs.
6. **Predictions**: The model makes predictions on the input data.
7. **Visualization**: Training loss and regression line are plotted.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Taking User input for x and y
x_input = input("Enter x values (separated by spaces): ")
y_input = input("Enter y values (separated by spaces): ")

# Converting input strings to numpy arrays
x = np.array(x_input.split(), dtype=np.float32)
y = np.array(y_input.split(), dtype=np.float32)

# Reshaping x and y for TensorFlow
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

# Defined the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compiled the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Training the model and storing it to the training history
history = model.fit(x, y, epochs=1000, verbose=0)

# Making predictions
y_pred = model.predict(x)

# Print the predictions
for i in range(len(x)):
    print(f"For x = {x[i][0]}, predicted y = {y_pred[i][0]:.2f}, actual y = {y[i][0]}")

# Taking user input for prediction
user_input = float(input("Enter a value of x to predict y: "))
user_pred = model.predict([[user_input]])
print(f"For x = {user_input}, predicted y = {user_pred[0][0]:.2f}")

# Ploting training loss graph
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Ploting the regression line and data points graph
plt.subplot(1, 2, 2)
plt.scatter(x, y, label='Actual Data')
plt.plot(x, y_pred, color='red', label='Regression Line')
plt.title('Regression Line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.tight_layout()
plt.show()
```

## Fine-Tuned Single-Layer Neural Network

### Description

The `Fine Tuned Single-Layer Neural Network.ipynb` notebook builds upon the first model by introducing a more complex neural network with additional layers and a different optimizer and loss function. The fine-tuned model uses the Adam optimizer and the Huber loss function, with the aim of improving the model's performance on the regression task.

### Key Steps

1. **User Input**: Users provide x and y values as input.
2. **Data Conversion**: The input strings are converted to NumPy arrays and reshaped for TensorFlow.
3. **Model Definition**: A neural network with multiple dense layers and ReLU activations is defined.
4. **Model Compilation**: The model is compiled with the Adam optimizer and Huber loss function.
5. **Model Training**: The model is trained for 1000 epochs.
6. **Predictions**: The model makes predictions on the input data.
7. **Visualization**: Training loss and regression line are plotted.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Taking User input for x and y
x_input = input("Enter x values (separated by spaces): ")
y_input = input("Enter y values (separated by spaces): ")

# Converting input strings to numpy arrays
x = np.array(x_input.split(), dtype=np.float32)
y = np.array(y_input.split(), dtype=np.float32)

# Reshaping x and y for TensorFlow
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

# Defined the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=32, activation='relu', input_shape=[1]),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

# Compiled the model
model.compile(optimizer='adam', loss='huber')

# Training the model and storing it to the training history
history = model.fit(x, y, epochs=1000, verbose=0)

# Making predictions
y_pred = model.predict(x)

# Print the predictions
for i in range(len(x)):
    print(f"For x = {x[i][0]}, predicted y = {y_pred[i][0]:.2f}, actual y = {y[i][0]}")

# Taking user input for prediction
user_input = float(input("Enter a value of x to predict y: "))
user_pred = model.predict([[user_input]])
print(f"For x = {user_input}, predicted y = {user_pred[0][0]:.2f}")

# Ploting training loss graph
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Ploting the regression line and data points graph
plt.subplot(1, 2, 2)
plt.scatter(x, y, label='Actual Data')
plt.plot(x, y_pred, color='red', label='Regression Line')
plt.title('Regression Line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.tight_layout()
plt.show()
```

## Conclusion

These notebooks provide a hands-on approach to understanding and improving neural network models for regression tasks. By experimenting with different model architectures and training parameters, users can gain insights into the impact of these changes on model performance.