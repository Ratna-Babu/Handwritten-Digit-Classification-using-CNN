# Handwritten-Digit-Classification-using-CNN


This project demonstrates how to build and train a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset using TensorFlow and Keras.

## Project Structure

The project consists of three main parts:

1. **Data Loading and Preprocessing:**
   ```python
   import tensorflow as tf

   # MNIST dataset
   (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

   # Invert the colors of the images
   x_train = 255 - x_train
   x_test = 255 - x_test
   # Preprocessing the Images
   x_train = x_train.astype('float32') / 255
   x_test = x_test.astype('float32') / 255

   # Reshape the images to include the channel dimension
   x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
   x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

   # One-hot encode the labels
   y_train = tf.keras.utils.to_categorical(y_train, 10)
   y_test = tf.keras.utils.to_categorical(y_test, 10)
   ```
   - Imports the MNIST dataset using `tf.keras.datasets.mnist.load_data()`.
   - Inverts the colors of the images for better model performance.
   - Normalizes pixel values to the range [0, 1].
   - Reshapes images to include the channel dimension.
   - One-hot encodes the labels for multi-class classification.

2. **CNN Model Building and Training:**
   - Builds a CNN model using `Sequential` with layers like `Conv2D`, `MaxPooling2D`, `Flatten`, and `Dense`.
     ```python
       # CNN model
       model = Sequential([
       Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
       MaxPooling2D(pool_size=(2, 2)),
       Conv2D(64, kernel_size=(3, 3), activation='relu'),
       MaxPooling2D(pool_size=(2, 2)),
       Flatten(),
       Dense(128, activation='relu'),
       Dense(10, activation='softmax')
       ])
      ```
   - Compiles the model with the Adam optimizer, categorical cross-entropy loss, and accuracy metric.
     ```python
     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
     ```
   - Trains the model using `fit()` with specified epochs, batch size, and validation split.
     ```python
     model.fit(x_train, y_train, epochs=10, batch_size=200, validation_split=0.2)
     ```
   - Saves the trained model to a file named 'mnist_cnn.h5'.
      ```python
      model.save('mnist_cnn.h5') 
      ```

3. **Prediction:**
   - Loads the trained model using `load_model()`.
   - Defines a `predict_digit()` function to preprocess an input image and make a prediction.
   - Defines a `display_prediction()` function to show the image with the prediction
   - Calls `predict_digit()` with an image path to predict the digit.
   - Prints the predicted digit to the console.

## Usage

1. **Install Dependencies:**
   `pip install tensorflow==2.12.0`
   `pip install keras==2.12.0`
   `pip install Pillow==9.5.0`
   `pip install numpy==1.24.3`
   `pip install opencv-python==4.8.0`
   `pip install matplotlib==3.7.2`
2. **Run the Code:**
   Execute the code in a Google Colab environment or a Jupyter Notebook.

3. **Prediction:**
   - Replace `'2.png'` in the `image_path` variable with the path to your image.
   - Run the code to see the prediction result printed to the console.

## Model Performance

The model achieves high accuracy on the MNIST test set. To see the output, run the code.


## Note

- The model is trained on inverted images (black digits on a white background).
- Make sure to have the necessary libraries installed before running the code.
