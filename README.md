# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
An autoencoder is an unsupervised neural network that compresses input images into lower-dimensional representations and then reconstructs them, aiming to produce outputs identical to the inputs. For training, we use the MNIST dataset, which contains 60,000 images of handwritten digits (each 28x28 pixels). The objective is to train a convolutional neural network on this dataset to classify each digit accurately into one of the 10 classes, ranging from 0 to 9.
## Convolution Autoencoder Network Model
![image](https://github.com/user-attachments/assets/d564b5d3-b587-48d4-871c-4dd54df81d29)


## DESIGN STEPS

## STEP 1:
Import the necessary libraries and dataset.

## STEP 2:
Load the dataset and scale the values for easier computation.

## STEP 3:
Add noise to the images randomly for both the train and test sets.

## STEP 4:
Build the Neural Model using,Convolutional,Pooling and Upsampling layers.

## STEP 5:
Pass test data for validating manually.

## STEP 6:
Plot the predictions for visualization.
## PROGRAM
### Name:BASKARAN V
### Register Number:212222230020
```
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
(x_train, _), (x_test, _) = mnist.load_data()

x_train_scaled = x_train.astype('float32') / 255.
x_test_scaled = x_test.astype('float32') / 255.
x_train_scaled = np.reshape(x_train_scaled, (len(x_train_scaled), 28, 28, 1))
x_test_scaled = np.reshape(x_test_scaled, (len(x_test_scaled), 28, 28, 1))

noise_factor = 0.5
x_train_noisy = x_train_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_scaled.shape)
x_test_noisy = x_test_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_scaled.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

inp_img=keras.Input(shape=(28,28,1))
x=layers.Conv2D(16,(3,3),activation='relu',padding='same')(inp_img)
x=layers.MaxPooling2D((2,2),padding='same')(x)
x=layers.Conv2D(8,(3,3),activation='relu',padding='same')(x)
x=layers.MaxPooling2D((2,2),padding='same')(x)
x=layers.Conv2D(8,(3,3),activation='relu',padding='same')(x)
encoder=layers.MaxPooling2D((2,2),padding='same')(x)
x=layers.Conv2D(8,(3,3),activation='relu',padding='same')(encoder)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(8,(3,3),activation='relu',padding='same')(x)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(16,(3,3),activation='relu')(x)
x=layers.UpSampling2D((2,2))(x)
decoder=layers.Conv2D(1,(3,3),activation='sigmoid',padding='same')(x)
model=keras.Model(inp_img,decoder)
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(x_train_noisy,x_train_scaled,epochs=50,batch_size=128,shuffle=True,validation_data=(x_test_noisy,x_test_scaled))

import pandas as pd
metrics=pd.DataFrame(model.history.history)
plt.figure(figsize=(7,2.5))
plt.plot(metrics['loss'], label='Training Loss')
plt.plot(metrics['val_loss'], label='Validation Loss')
plt.title('Training Loss vs. Validation Loss\n BASKARAN V(212222230020)')

decodeimg=model.predict(x_test_noisy)

def display_images(x_test_scaled, x_test_noisy, decodeimg, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        for j, img in enumerate([x_test_noisy,x_test_scaled,decodeimg]):
            ax = plt.subplot(3, n, i + 1 + j * n)
            plt.imshow(img[i].reshape(28, 28), cmap='gray')
            ax.axis('off')
    plt.show()
display_images(x_test_noisy,x_test_scaled, decodeimg)
```



## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![download](https://github.com/user-attachments/assets/0de447a1-0ce4-4427-bf42-4159a8e8fcbc)


### Original vs Noisy Vs Reconstructed Image

![download](https://github.com/user-attachments/assets/62936791-f417-4698-b4d2-b5457bbbb05a)




## RESULT
Thus we have successfully developed a convolutional autoencoder for image denoising application.
