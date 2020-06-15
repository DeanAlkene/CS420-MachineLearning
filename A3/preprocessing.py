import numpy as np
import pickle

def unpickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='iso-8859-1')
    return data['data'], np.array(data['labels'])

def getTrainingData():
    X, y = [], []
    for i in range(1, 6):
        data, labels = unpickle('data_batch_%d' % (i))
        X.append(data)
        y.append(labels)
    X = np.vstack(X)
    y = np.hstack(y)
    np.save('cifar_X_train', X)
    np.save('cifar_y_train', y)

def getTestData():
    data, labels = unpickle('test_batch')
    np.save('cifar_X_test', data)
    np.save('cifar_y_test', labels)

getTrainingData()
getTestData()