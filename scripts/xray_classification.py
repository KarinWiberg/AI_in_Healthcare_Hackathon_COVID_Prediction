#1.0 validation accuracy and recall (was on a slower computer so worked with a small validation set, n=18)

import numpy as np 
import pandas as pd 
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers

images_dir = 'dataset/'

datagen = ImageDataGenerator (
            rescale = 1./255, 
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
            )

train_generator  =    datagen.flow_from_directory(
                             images_dir,
                             seed=42,
                             target_size = (200,200),
                             batch_size =32 ,               
                             class_mode = 'binary',
                             subset = 'training'
                            )

#Xtrain, ytrain = train_generator.next()

Validation_generator = datagen.flow_from_directory(
                             images_dir ,
                             seed=42, 
                             target_size = (200,200),
                             batch_size = 32 ,               
                             class_mode = 'binary',
                             subset = 'validation'
                            )

#Xtest, ytest = Validation_generator.next()

from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint

model=Sequential()

model.add(Conv2D(32,(3,3),input_shape=(200,200,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128,activation='relu'))
model.add(Dense(2,activation='softmax'))

import keras.backend as K

def precision(y_true, y_pred): #taken from old keras source code
     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
     precision = true_positives / (predicted_positives + K.epsilon())
     return precision
def recall(y_true, y_pred): #taken from old keras source code
     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
     recall = true_positives / (possible_positives + K.epsilon())
     return recall

def fbeta(y_true, y_pred, threshold_shift=0.9):
    beta = 2
    y_pred = K.clip(y_pred, 0, 1)
    y_pred_bin = K.round(y_pred + threshold_shift)
    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy', recall, fbeta])


H = model.fit_generator(
                    train_generator,
                    epochs = 20,
                    validation_data = Validation_generator)


from matplotlib import pyplot as plt
N = 20
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0,N) , H.history['loss'] , label ='train_loss')
plt.plot(np.arange(0,N) , H.history['val_loss'] , label ='val_loss')
plt.title("Training Loss and Accuracy")
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.legend(loc="lower left")
plt.show()
