from keras import Sequential
from keras.layers import Dense
from checkingInstances import getData, getDataofMissingOnes, getPositionMissingOnes
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import arff


def myModel():
    model = Sequential()

    model.add(Dense(10, activation='relu', kernel_initializer='normal', input_shape=(10,)))
    model.add(Dense(5, activation='relu', kernel_initializer='normal'))
    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

def getMissingDataAttr1():
    data, label = getData(0)

    scaler = MinMaxScaler()
    scaler.fit_transform(data)

    model1.fit(data, label, epochs=100, batch_size=64, verbose=0)

    return model1.predict(getDataofMissingOnes(0))


def getMissingDataAttr2():
    data, label = getData(1)

    scaler = MinMaxScaler()
    scaler.fit_transform(data)

    model2.fit(data, label, epochs=100, batch_size=64, verbose=0)

    return model2.predict(getDataofMissingOnes(1))


def getMissingDataAttr3():
    data, label = getData(2)

    scaler = MinMaxScaler()
    scaler.fit_transform(data)

    model3.fit(data, label, epochs=100, batch_size=64, verbose=0)

    return model3.predict(getDataofMissingOnes(2))


def replacingMissingValues():
    myArff = arff.load(open('competition-iaa-2018-2019/train.arff', 'r'))
    data = np.array(myArff['data'])

    positions1 = getPositionMissingOnes(0)
    data1 = getMissingDataAttr1()

    j = 0
    for i in positions1:
        data[i][8] = data1[j]
        j += 1

    positions2 = getPositionMissingOnes(1)
    data2 = getMissingDataAttr2()

    j = 0
    for i in positions2:
        data[i][9] = data2[j]
        j += 1

    positions3 = getPositionMissingOnes(2)
    data3 = getMissingDataAttr3()

    j = 0
    for i in positions3:
        data[i][10] = data3[j]
        j += 1

    myArff['data'] = data

    f = open('pruea.arff', 'w')
    arff.dump(myArff, f)


def replacingMissingValuesTest():
    myArff = arff.load(open('competition-iaa-2018-2019/test.arff', 'r'))
    data = np.array(myArff['data'])

    posicion1 = []
    posicion2 = []
    posicion3 = []

    for i in range(0, data.shape[0]):
        if data[i][8] is None:
            posicion1.append(i)

        if data[i][9] is None:
            posicion2.append(i)

        if data[i][10] is None:
            posicion3.append(i)

    posicion1 = np.array(posicion1)
    posicion2 = np.array(posicion2)
    posicion3 = np.array(posicion3)

    predecir1 = []
    predecir2 = []
    predecir3 = []

    for i in range(0, data.shape[0]):
        if data[i][8] is None:
            predecir1.append(data[i])

        if data[i][9] is None:
            predecir2.append(data[i])

        if data[i][10] is None:
            predecir3.append(data[i])

    predecir1 = np.array(predecir1)
    predecir1 = predecir1[:, [0, 1, 2, 3, 4, 5, 6, 7, 11, 12]]

    predecir2 = np.array(predecir2)
    predecir2 = predecir2[:, [0, 1, 2, 3, 4, 5, 6, 7, 11, 12]]

    predecir3 = np.array(predecir3)
    predecir3 = predecir3[:, [0, 1, 2, 3, 4, 5, 6, 7, 11, 12]]

    predicted1 = model1.predict(predecir1)
    predicted2 = model2.predict(predecir2)
    predicted3 = model3.predict(predecir3)

    j = 0
    for i in posicion1:
        data[i][8] = predicted1[j]
        j += 1

    j = 0
    for i in posicion2:
        data[i][9] = predicted2[j]
        j += 1

    j = 0
    for i in posicion3:
        data[i][10] = predicted3[j]
        j += 1

    myArff['data'] = data

    f = open('test.arff', 'w')
    arff.dump(myArff, f)


model1 = myModel()
model2 = myModel()
model3 = myModel()
replacingMissingValues()
replacingMissingValuesTest()
