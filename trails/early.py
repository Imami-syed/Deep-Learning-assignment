import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
train_data1=np.load('data0.npy')
lab_data1=np.load('lab0.npy')
x_train, x_test, y_train, y_test = train_test_split(train_data1, lab_data1, test_size=0.33)
x_train = x_train.reshape(-1, 40, 168, 1)
x_test = x_test.reshape(-1, 40, 168, 1)
x_train = x_train / 255.0
x_test = x_test / 255.0
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(40, 168, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    # keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(37, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=3)
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), callbacks=[early_stop])

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)
with open('early.txt', 'w') as f:
    f.write(str(test_acc))
    f.write("\n")

