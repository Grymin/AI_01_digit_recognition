import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

np.set_printoptions(suppress=True, linewidth=120)

# Loading data
(X_train, y_train), (X_test, y_test) = load_data()
print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of X_test: {X_test.shape}')
print(f'Shape of y_train: {y_train.shape}')
print(f'Shape of y_test: {y_test.shape}')

X_train = X_train/255
X_test = X_test/255

# Example of numbers
fig, axs = plt.subplots(2,5)
for i in range(0, 10):
   row = i // 5
   col = i % 5
   axs[row, col].axis('off')
   axs[row, col].imshow(X_train[i], cmap='gray')
   axs[row, col].set_title(y_train[i])
plt.show()

# Generation of model
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=10, activation="softmax"))
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# Fitting the model
epochs = 20
history = model.fit(X_train, y_train, epochs=epochs)

# Evaluating the model
model.evaluate(X_test, y_test, verbose=2)
metrics = pd.DataFrame(history.history)
print('Evaluation of the model')
print(metrics)

# Process of our fitting
fig, axs = plt.subplots(2, 1)
plt.subplots_adjust(hspace=1)
axs[0].scatter(x=range(0, epochs, 1), y=metrics['loss'])
axs[0].set_title('loss')
axs[1].scatter(x=range(0, epochs, 1), y=metrics['accuracy'])
axs[1].set_title('accuracy')
plt.title('loss and accuracy in the model')
plt.show()

np.argmax(model.predict(X_test), axis=-1)

y_predict = np.argmax(model.predict(X_test), axis=-1)
print('values predicted for the X_test:')
print(y_predict)

comparison = pd.concat([pd.DataFrame(y_test, columns=['y_test']), pd.DataFrame(y_predict, columns=['y_predict'])], axis=1)
print('Comparison of the y_test and predicted values')
comparison.head(n=5)

wrong = comparison[comparison.loc[:, 'y_test'] != comparison.loc[:, 'y_predict']]
wrong_indexes = wrong.index

fig, axs = plt.subplots(4, 5)
plt.subplots_adjust(hspace=1)
wrong_indexes_first_20 = wrong_indexes[:20]
for i, ind in enumerate(wrong_indexes_first_20):
   row = i // 5
   col = i % 5
   axs[row, col].axis('off')
   axs[row, col].imshow(X_test[ind], cmap='gray')
   axs[row, col].set_title(f't:{y_test[ind]}, p:{y_predict[ind]}')
plt.show()