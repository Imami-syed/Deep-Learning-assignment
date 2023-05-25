#%pylab
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#%matplotlib inline
from sklearn.model_selection import train_test_split
train_data1=np.load('data0.npy')
lab_data1=np.load('lab0.npy')
i=9997
#plt.plot(train_data1[i])
#imshow(train_data1[i])
#print(lab_data1[i])
#print(train_data1[0].shape)
X_train, X_test, y_train, y_test = train_test_split(train_data1, lab_data1, test_size=0.33)
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255
model=tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(2,2),strides=(1, 1), padding='same', activation='relu', input_shape=(40,168,1))) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(2,2),strides=(1, 1), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(2,2),strides=(1, 1), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(37, activation='softmax'))
# Take a look at the model summary
model.summary()
model.compile(loss='sparse_categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
X_train = X_train.reshape(-1,40,168,1)
X_test = X_test.reshape(-1,40, 168, 1)
model_log=model.fit(X_train, y_train,
          batch_size=40,
          epochs=168,
          verbose=1,
          validation_split=.3)
score = model.evaluate(X_test, y_test, verbose=0)
print("score[1]")
	
