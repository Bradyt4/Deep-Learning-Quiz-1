'''

Q3 Inception Module for CIFAR-10 dataset

'''
import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Flatten, Dense
from keras.layers import Input
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

epochs = 50

# Get the data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

datagen = ImageDataGenerator(
    rotation_range=5,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
    )

# Get the data ready
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Create imput
input_img = Input(shape = (32, 32, 3))

first = Conv2D(64, (1,1),padding='same', activation='relu')(input_img)
first = AveragePooling2D((2,2), strides=None, padding='same')(first)
first = Conv2D(32, (1,1),padding='same', activation='relu')(first)
# Create Volumes for the Inception module
volume_1 = Conv2D(64, (1,1), padding='same', activation='relu')(first)

volume_2 = Conv2D(96, (1,1), padding='same', activation='relu')(first)
volume_2 = Conv2D(128, (3,3), padding='same', activation='relu')(volume_2)

volume_3 = Conv2D(16, (1,1), padding='same', activation='relu')(first)
volume_3 = Conv2D(32, (5,5), padding='same', activation='relu')(volume_3)

volume_4 = MaxPooling2D((3,3), strides=(1,1), padding='same')(first)
volume_4 = Conv2D(32, (1,1), padding='same', activation='relu')(volume_4)

# Concatenate all volumes of the Inception module


inception_module = keras.layers.concatenate([volume_1, volume_2, volume_3,
                                              volume_4], axis=3)
post_inception = Conv2D(128, (1,1), padding='same', activation='relu')(inception_module)
output = Flatten()(post_inception)

out = Dense(10, activation='softmax')(output)


model = Model(inputs = input_img, outputs = out)
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=128)
hist = model.fit(datagen.flow(X_train, y_train, batch_size=128),
                    steps_per_epoch=len(X_train) / 128, epochs=epochs, validation_data=(X_test, y_test))


scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


