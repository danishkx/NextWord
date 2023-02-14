import tensorflow
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, Dense, GRU
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

file = open("/content/drive/MyDrive/Colab Notebooks/dsnew.txt", "r", encoding = "utf8")
lines = []
for i in file:
    lines.append(i)
    
print("The First Line: ", lines[0])
print("The Last Line: ", lines[-1])

data = ""
for i in lines:
    data = ' '. join(lines)
    
data = data.replace('\n', '').replace('\r', '').replace('\ufeff', '').replace('"','')
data[:360]

data = data.split()  
data = ' '.join(data)
data[:500]

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
tokenizer = pickle.dump(tokenizer, open('/content/drive/MyDrive/Colab Notebooks/tokenizer36.pkl', 'wb'))

sequence_data = tokenizer.texts_to_sequences([data])[0]
sequence_data[:15]

len(sequence_data)
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)

sequences = []
for i in range(3, len(sequence_data)):
    words = sequence_data[i-3:i+1]
    sequences.append(words)

print("The Length of sequences are: ", len(sequences))

sequences = np.array(sequences)
sequences[:10]

X = []
y = []

for i in sequences:
    X.append(i[0:3])
    y.append(i[3])
 
X = np.array(X)
y = np.array(y)
print("The Data is: ", X[:10])
print("The responses are: ", y[:10])

y = to_categorical(y, num_classes=vocab_size)
y[:5]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=3))
model.add(LSTM(1000, return_sequences=True))
model.add(LSTM(1000))
model.add(Dense(1000, activation="relu"))
model.add(Dense(vocab_size, activation='softmax'))
model.summary()

from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

checkpoint = ModelCheckpoint("nextword37.h5", monitor='loss', verbose=1,save_best_only=True)
reduce = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.0001, verbose = 1)

model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001),metrics=['accuracy'])
history =  model.fit(X, y, epochs=30, batch_size=64, validation_data=(X_val, y_val),callbacks=[checkpoint,reduce])
model.save('/content/drive/MyDrive/Colab Notebooks/nextword37.h5',overwrite=True)

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
