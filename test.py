from keras.models import Sequential
from keras.datasets import mnist

from blurpool import BlurPool2D


X_train, y_train, X_test, y_test = mnist.load_data()

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=5, stride=1, activation='relu'))
model.add(BlurPool2D(kernel_size=5, stride=2))
model.add(Flatten())
model.add(Dense(units=10, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=50, batch_size=32, metrics=['accuracy'])