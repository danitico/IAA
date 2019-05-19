from keras import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import keras_metrics as km
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import arff
import csv

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


labels = array[:, 13]
array = array[:, 0:13]
# labels = to_categorical(labels)

scaler = MinMaxScaler()

scaler.fit(array)
array = scaler.transform(array)
test = scaler.transform(test)

train_x = array[0:2100, :]
val_x = array[2100:, :]
train_y = labels[0:2100]
val_y = labels[2100:]
train_y = to_categorical(train_y)
val_y = to_categorical(val_y)

model = Sequential()

model.add(Dense(100, activation='relu', input_shape=(13,)))

model.add(Dense(50, activation='relu'))

model.add(Dropout(0.05))

model.add(Dense(25, activation='relu'))

model.add(Dense(12, activation='relu'))

# model.add(Dense(20, activation='relu', input_shape=(13,)))
#
# model.add(Dense(10, activation='relu'))
#
model.add(Dense(4, activation='sigmoid'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

callbacks = []
callbacks.append(EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, mode='min', restore_best_weights=True,
                               verbose=1))

history = model.fit(train_x, train_y, epochs=1000, batch_size=64, verbose=1, validation_data=(val_x, val_y),
                    callbacks=callbacks)

# history = model.fit(array, labels, epochs=150, batch_size=64, verbose=1)


print(np.max(np.array(history.history['acc'])))

# predictions = model.predict_classes(test)
#
# file = csv.reader(open('competition-iaa-2018-2019/sampleSubmission.csv'))
# data_file = list(file)
# data_file_array = np.array(data_file)
#
# data_file_array[1:, 1] = predictions
#
# writer = csv.writer(open('/tmp/output.csv', 'w'))
# writer.writerows(data_file_array)
#
# print(data_file_array)
