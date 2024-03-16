import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 
from tensorflow import keras

#prepare data
(training_images, training_labels), (testing_images, testing_labels) = keras.datasets.cifar10.load_data();
training_images, testing_images = training_images/255, testing_images/255

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

for i in range(16):
    #4x4 grid
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap = plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

plt.show()

training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

model = keras.models.Sequential()
# filters for features
# 2D convolutional layer; 32 filters, 3x3, shape of input data: 32x32 RGB images
model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
# reduces image to essential information
model.add(keras.layers.MaxPooling2D((2,2)))
# learn more features (64)
model.add(keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Conv2D(64, (3,3), activation='relu'))
# converts from 2D to 1D
model.add(keras.layers.Flatten())
# a fully connected layer, with 64 units 
model.add(keras.layers.Dense(64, activation='relu'))
# classify input in 1 to 10 categories
model.add(keras.layers.Dense(10, activation='softmax'))

# adjusts weights to minimize the loss
# prepare model for training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

model.save("image_classifier.model")
