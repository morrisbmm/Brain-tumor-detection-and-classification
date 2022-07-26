
import os
import numpy as np # linear algebra
import pandas as pd # data processing
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

import zipfile
local_zip = '/content/brain_tumor.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/content/data')
zip_ref.close()

base_dir = '/content/data/brain_tumor'
training_dir = os.path.join(base_dir, 'Training')
testing_dir = os.path.join(base_dir, 'Testing')
pituitary_train = os.path.join(training_dir,'pituitary')
notumor_train = os.path.join(training_dir,'notumor')
meningioma_train = os.path.join(training_dir,'meningioma')
glioma_train = os.path.join(training_dir,'glioma')

print('total training pituitary images:', len(os.listdir(pituitary_train)))
print('total training notumor images:', len(os.listdir(notumor_train)))
print('total training meningioma images:', len(os.listdir(meningioma_train)))
print('total training glioma images', len(os.listdir(glioma_train)))

# creating a custom relu function

model = tf.keras.Sequential([
    #first convolution layer
    tf.keras.layers.Conv2D(32,(3,3), activation = 'relu',input_shape = (64,64,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    #second convolution layer
    tf.keras.layers.Conv2D(32,(3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #third convolution layer
    tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #fourth convolution layer
    tf.keras.layers.Conv2D(128,(3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #Flatten the results to feed to a DNN
    tf.keras.layers.Flatten(),
    #512 neuron hidden layer
    tf.keras.layers.Dense(128, activation = 'relu'),
    # add a dropout rate 0.2
    tf.keras.layers.Dropout(0.2),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation = 'relu'),
    #add a dropout rate 0.2
    tf.keras.layers.Dropout(0.2),
    #add a final softmax layer for classification
    tf.keras.layers.Dense(4, activation = 'softmax')
    
])

TRAINING_DIR = training_dir
train_datagen = ImageDataGenerator(rescale = 1.0/255,
                                  rotation_range = 40,
                                  width_shift_range = 0.2,
                                  height_shift_range = 0.2,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True,
                                  fill_mode = 'nearest')
VALIDATION_DIR = testing_dir
validation_datagen = ImageDataGenerator(rescale = 1.0/255)
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                   target_size = (64,64),
                                                   batch_size = 32,
                                                   class_mode = 'categorical',
                                                   color_mode = 'grayscale')
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                             target_size = (64,64),
                                                             batch_size = 32,
                                                             class_mode = 'categorical',
                                                             color_mode = 'grayscale')

# checking the parameters of the model

model.summary()

# compiling the model

from tensorflow.keras.optimizers import Adam
model.compile(optimizer = Adam(learning_rate = 0.001),
             loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

# training the  model

history = model.fit(train_generator,
                             steps_per_epoch = 5712//32,
                             epochs = 200,
                             validation_data = validation_generator,
                             validation_steps = 1311//32,
                             verbose = 2 )

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label = 'Training accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation accuracy')
plt.title('Training and Validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'r', label = 'Training Loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.legend(loc= 0)

plt.show()