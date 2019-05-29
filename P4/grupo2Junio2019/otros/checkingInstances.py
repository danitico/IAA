import arff
import numpy as np

def info():
    myArff = arff.load(open('competition-iaa-2018-2019/train.arff', 'r'))
    data = np.array(myArff['data'])

    perdidos1 = 0
    perdidos2 = 0
    perdidos3 = 0

    for i in range(0, 643):
        if data[i][8] is None:
            perdidos1 += 1

        if data[i][9] is None:
            perdidos2 += 1

        if data[i][10] is None:
            perdidos3 += 1

    print("Datos perdidos del primer atributo en el primer año -> ", perdidos1)
    print("Datos perdidos del segundo atributo en el primer año -> ", perdidos2)
    print("Datos perdidos del tercer atributo en el primer año -> ", perdidos3)

    perdidos1 = 0
    perdidos2 = 0
    perdidos3 = 0

    for i in range(643, 2100):
        if data[i][8] is None:
            perdidos1 += 1

        if data[i][9] is None:
            perdidos2 += 1

        if data[i][10] is None:
            perdidos3 += 1

    print("Datos perdidos del primer atributo en el segundo año -> ", perdidos1)
    print("Datos perdidos del segundo atributo en el segundo año -> ", perdidos2)
    print("Datos perdidos del tercer atributo en el segundo año -> ", perdidos3)

    perdidos1 = 0
    perdidos2 = 0
    perdidos3 = 0

    for i in range(2100, 3559):
        if data[i][8] is None:
            perdidos1 += 1

        if data[i][9] is None:
            perdidos2 += 1

        if data[i][10] is None:
            perdidos3 += 1

    print("Datos perdidos del primer atributo en el tercer año -> ", perdidos1)
    print("Datos perdidos del segundo atributo en el tercer año -> ", perdidos2)
    print("Datos perdidos del tercer atributo en el tercer año -> ", perdidos3)


def getData(opcion):

    myArff = arff.load(open('competition-iaa-2018-2019/train.arff', 'r'))
    dataarff = np.array(myArff['data'])

    label1 = []
    label2 = []
    label3 = []
    data1 = []
    data2 = []
    data3 = []

    for i in range(0, 3559):
        if dataarff[i][8] is not None and opcion == 0:
            label1.append(dataarff[i][8])
            data1.append(dataarff[i])

        if dataarff[i][9] is not None and opcion == 1:
            label2.append(dataarff[i][9])
            data2.append(dataarff[i])

        if dataarff[i][10] is not None and opcion == 2:
            label3.append(dataarff[i][10])
            data3.append(dataarff[i])

    if opcion == 0:
        label1 = np.array(label1)
        data1 = np.array(data1)
        data1 = data1[:, [0, 1, 2, 3, 4, 5, 6, 7, 11, 12]]

        return data1, label1

    elif opcion == 1:
        label2 = np.array(label2)
        data2 = np.array(data2)
        data2 = data2[:, [0, 1, 2, 3, 4, 5, 6, 7, 11, 12]]

        return data2, label2

    elif opcion == 2:
        label3 = np.array(label3)
        data3 = np.array(data3)
        data3 = data3[:, [0, 1, 2, 3, 4, 5, 6, 7, 11, 12]]

        return data3, label3


def getPositionMissingOnes(opcion):
    myArff = arff.load(open('competition-iaa-2018-2019/train.arff', 'r'))
    data = np.array(myArff['data'])

    posicion1 = []
    posicion2 = []
    posicion3 = []

    for i in range(0, 3559):
        if data[i][8] is None and opcion == 0:
            posicion1.append(i)

        if data[i][9] is None and opcion == 1:
            posicion2.append(i)

        if data[i][10] is None and opcion == 2:
            posicion3.append(i)

    if opcion == 0:
        return np.array(posicion1)

    elif opcion == 1:
        return np.array(posicion2)

    elif opcion == 2:
        return np.array(posicion3)


def getDataofMissingOnes(opcion):
    myArff = arff.load(open('competition-iaa-2018-2019/train.arff', 'r'))
    data = np.array(myArff['data'])

    predecir1 = []
    predecir2 = []
    predecir3 = []

    for i in range(0, 3559):
        if data[i][8] is None and opcion == 0:
            predecir1.append(data[i])

        if data[i][9] is None and opcion == 1:
            predecir2.append(data[i])

        if data[i][10] is None and opcion == 2:
            predecir3.append(data[i])

    if opcion == 0:
        predecir1 = np.array(predecir1)
        predecir1 = predecir1[:, [0, 1, 2, 3, 4, 5, 6, 7, 11, 12]]

        return predecir1

    elif opcion == 1:
        predecir2 = np.array(predecir2)
        predecir2 = predecir2[:, [0, 1, 2, 3, 4, 5, 6, 7, 11, 12]]

        return predecir2

    elif opcion == 2:
        predecir3 = np.array(predecir3)
        predecir3 = predecir3[:, [0, 1, 2, 3, 4, 5, 6, 7, 11, 12]]

        return predecir3
