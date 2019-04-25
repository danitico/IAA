from keras import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import numpy as np
import arff

data = arff.load(open('competition-iaa-2018-2019/train.arff', 'r'))
data1 = arff.load(open('competition-iaa-2018-2019/test.arff', 'r'))

array = np.array(data['data'])
test = np.array(data1['data'])

for i in range(0, array.shape[0]):
    if array[i][13] == 'Baja':
        array[i][13] = 0
    elif array[i][13] == 'Media':
        array[i][13] = 1
    elif array[i][13] == 'Moderada':
        array[i][13] = 2
    else:
        array[i][13] = 3


train_labels = array[:, 13]
train_labels = to_categorical(train_labels)
train = array[:, 0:13]

model = Sequential()

model.add(Dense(80, activation='relu', input_shape=(13,)))
model.add(Dropout(0.2))

model.add(Dense(85, activation='relu'))
model.add(Dropout(0.01))

model.add(Dense(4, activation='sigmoid'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = []
callbacks.append(EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, mode='min', restore_best_weights=True))

history = model.fit(train, train_labels, epochs=1000, batch_size=128, validation_split=0.15, verbose=1,
                    callbacks=callbacks)

print(np.mean(np.array(history.history['acc'])))

predictions = model.predict_classes(test)

print(predictions)

