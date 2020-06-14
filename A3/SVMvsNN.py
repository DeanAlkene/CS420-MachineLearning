import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import time

cancer = load_svmlight_file('data/breast-cancer')
cancer_X = cancer[0].toarray()
cancer_y = cancer[1]
dna = load_svmlight_file('data/dna')
dna_X = dna[0].toarray()
dna_y = dna[1]

params_svm = {
    'scale': [True, False],
    'test_size': [0.1, 0.2, 0.3, 0.4, 0.5],
    'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'cancer_dim': [2, 3, 5, 8, 10],
    'dna_dim': [2, 5, 10, 20, 50, 100, 150, 180]
}

params_mlp = {
    'scale': [True, False],
    'layers': [(10,), (100,), (10, 10), (100, 100), (200, 200), (100, 200, 100)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'alpha': [0.001, 0.0001],
    'lr': ['constant', 'invscaling', 'adaptive'],
    'cancer_dim': [2, 3, 5, 8, 10],
    'dna_dim': [2, 5, 10, 20, 50, 100, 150, 180]
}

def runSVM():
    # scale
    test_size = 0.4
    C = 1.0
    kernel = 'rbf'
    cancer_dim = 10
    dna_dim = 180
    for scale in params_svm['scale']:
        X_train, X_test, y_train, y_test = train_test_split(cancer_X, cancer_y, test_size=test_size)
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
        start = time.time()
        model = SVC(C=C, kernel=kernel)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        end = time.time()
        with open('cancer_scale.txt', 'a') as f:
            f.write("scale=%s, test_size=%f, C=%f, kernel=%s, cancer_dim=%d, score=%f, time=%fs\n" % (str(scale), test_size, C, kernel, cancer_dim, score, end - start))
        X_train, X_test, y_train, y_test = train_test_split(dna_X, dna_y, test_size=test_size)
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
        start = time.time()
        model = SVC(C=C, kernel=kernel)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        end = time.time()
        with open('dna_scale.txt', 'a') as f:
            f.write("scale=%s, test_size=%f, C=%f, kernel=%s, dna_dim=%d, score=%f, time=%fs\n" % (str(scale), test_size, C, kernel, dna_dim, score, end - start))
    
    scale = False
    # test_size
    C = 1.0
    kernel = 'rbf'
    cancer_dim = 10
    dna_dim = 180
    for test_size in params_svm['test_size']:
        X_train, X_test, y_train, y_test = train_test_split(cancer_X, cancer_y, test_size=test_size)
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
        start = time.time()
        model = SVC(C=C, kernel=kernel)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        end = time.time()
        with open('cancer_test_size.txt', 'a') as f:
            f.write("scale=%s, test_size=%f, C=%f, kernel=%s, cancer_dim=%d, score=%f, time=%fs\n" % (str(scale), test_size, C, kernel, cancer_dim, score, end - start))
        X_train, X_test, y_train, y_test = train_test_split(dna_X, dna_y, test_size=test_size)
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
        start = time.time()
        model = SVC(C=C, kernel=kernel)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        end = time.time()
        with open('dna_test_size.txt', 'a') as f:
            f.write("scale=%s, test_size=%f, C=%f, kernel=%s, dna_dim=%d, score=%f, time=%fs\n" % (str(scale), test_size, C, kernel, dna_dim, score, end - start))
    
    scale = False
    test_size = 0.4
    # C
    kernel = 'rbf'
    cancer_dim = 10
    dna_dim = 180
    for C in params_svm['C']:
        X_train, X_test, y_train, y_test = train_test_split(cancer_X, cancer_y, test_size=test_size)
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
        start = time.time()
        model = SVC(C=C, kernel=kernel)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        end = time.time()
        with open('cancer_C.txt', 'a') as f:
            f.write("scale=%s, test_size=%f, C=%f, kernel=%s, cancer_dim=%d, score=%f, time=%fs\n" % (str(scale), test_size, C, kernel, cancer_dim, score, end - start))
        X_train, X_test, y_train, y_test = train_test_split(dna_X, dna_y, test_size=test_size)
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
        start = time.time()
        model = SVC(C=C, kernel=kernel)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        end = time.time()
        with open('dna_C.txt', 'a') as f:
            f.write("scale=%s, test_size=%f, C=%f, kernel=%s, dna_dim=%d, score=%f, time=%fs\n" % (str(scale), test_size, C, kernel, dna_dim, score, end - start))
    
    scale = False
    test_size = 0.4
    C = 1.0
    # kernel
    cancer_dim = 10
    dna_dim = 180
    for kernel in params_svm['kernel']:
        X_train, X_test, y_train, y_test = train_test_split(cancer_X, cancer_y, test_size=test_size)
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
        start = time.time()
        model = SVC(C=C, kernel=kernel)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        end = time.time()
        with open('cancer_kernel.txt', 'a') as f:
            f.write("scale=%s, test_size=%f, C=%f, kernel=%s, cancer_dim=%d, score=%f, time=%fs\n" % (str(scale), test_size, C, kernel, cancer_dim, score, end - start))
        X_train, X_test, y_train, y_test = train_test_split(dna_X, dna_y, test_size=test_size)
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
        start = time.time()
        model = SVC(C=C, kernel=kernel)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        end = time.time()
        with open('dna_kernel.txt', 'a') as f:
            f.write("scale=%s, test_size=%f, C=%f, kernel=%s, dna_dim=%d, score=%f, time=%fs\n" % (str(scale), test_size, C, kernel, dna_dim, score, end - start))
    
    scale = False
    test_size = 0.4
    C = 1.0
    kernel = 'rbf'
    # cancer_dim = 10
    # dna_dim = 180
    for cancer_dim in params_svm['cancer_dim']:
        X_train, X_test, y_train, y_test = train_test_split(cancer_X, cancer_y, test_size=test_size)
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
        start = time.time()
        if cancer_dim != 10:
            pca = PCA(n_components=cancer_dim)
            X_train_new = pca.fit_transform(X_train)
            X_test_new = pca.transform(X_test)
        else:
            X_train_new = X_train
            X_test_new = X_test
        if scale:
            scaler = StandardScaler()
            X_train_new = scaler.fit_transform(X_train_new)
            X_test_new = scaler.fit_transform(X_test_new)
        model = SVC(C=C, kernel=kernel)
        model.fit(X_train_new, y_train)
        score = model.score(X_test_new, y_test)
        end = time.time()
        with open('cancer_dim.txt', 'a') as f:
            f.write("scale=%s, test_size=%f, C=%f, kernel=%s, cancer_dim=%d, score=%f, time=%fs\n" % (str(scale), test_size, C, kernel, cancer_dim, score, end - start))
    for dna_dim in params_svm['dna_dim']:
        X_train, X_test, y_train, y_test = train_test_split(dna_X, dna_y, test_size=test_size)
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
        start = time.time()
        if dna_dim != 180:
            pca = PCA(n_components=dna_dim)
            X_train_new = pca.fit_transform(X_train)
            X_test_new = pca.transform(X_test)
        else:
            X_train_new = X_train
            X_test_new = X_test
        if scale:
            scaler = StandardScaler()
            X_train_new = scaler.fit_transform(X_train_new)
            X_test_new = scaler.fit_transform(X_test_new)
        model = SVC(C=C, kernel=kernel)
        model.fit(X_train_new, y_train)
        score = model.score(X_test_new, y_test)
        end = time.time()
        with open('dna_dim.txt', 'a') as f:
            f.write("scale=%s, test_size=%f, C=%f, kernel=%s, dna_dim=%d, score=%f, time=%fs\n" % (str(scale), test_size, C, kernel, dna_dim, score, end - start))
        
def runMLP():
    pass

runSVM()