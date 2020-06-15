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
    'test_size': [0.1, 0.2, 0.3, 0.4, 0.5],
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
    # cancer_dim
    # dna_dim
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
    # scale
    test_size = 0.4
    layers = (100, )
    activation = 'relu'
    alpha = 0.0001
    lr = 'adaptive'
    cancer_dim = 10
    dna_dim = 180
    for scale in params_mlp['scale']:
        X_train, X_test, y_train, y_test = train_test_split(cancer_X, cancer_y, test_size=test_size)
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
        start = time.time()
        model = MLPClassifier(hidden_layer_sizes=layers, activation=activation, alpha=alpha, learning_rate=lr, max_iter=1000)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        end = time.time()
        with open('cancer_scale.txt', 'a') as f:
            f.write("scale=%s, test_size=%f, layers=%s, activation=%s, alpha=%f, lr=%s, cancer_dim=%d, score=%f, time=%fs\n" % (str(scale), test_size, str(layers), activation, alpha, lr, cancer_dim, score, end - start))
        X_train, X_test, y_train, y_test = train_test_split(dna_X, dna_y, test_size=test_size)
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
        start = time.time()
        model = MLPClassifier(hidden_layer_sizes=layers, activation=activation, alpha=alpha, learning_rate=lr, max_iter=1000)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        end = time.time()
        with open('dna_scale.txt', 'a') as f:
            f.write("scale=%s, test_size=%f, layers=%s, activation=%s, alpha=%f, lr=%s, dna_dim=%d, score=%f, time=%fs\n" % (str(scale), test_size, str(layers), activation, alpha, lr, dna_dim, score, end - start))
    
    scale = False
    # test_size
    layers = (100, )
    activation = 'relu'
    alpha = 0.0001
    lr = 'adaptive'
    cancer_dim = 10
    dna_dim = 180
    for test_size in params_mlp['test_size']:
        X_train, X_test, y_train, y_test = train_test_split(cancer_X, cancer_y, test_size=test_size)
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
        start = time.time()
        model = MLPClassifier(hidden_layer_sizes=layers, activation=activation, alpha=alpha, learning_rate=lr, max_iter=1000)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        end = time.time()
        with open('cancer_test_size.txt', 'a') as f:
            f.write("scale=%s, test_size=%f, layers=%s, activation=%s, alpha=%f, lr=%s, cancer_dim=%d, score=%f, time=%fs\n" % (str(scale), test_size, str(layers), activation, alpha, lr, cancer_dim, score, end - start))
        X_train, X_test, y_train, y_test = train_test_split(dna_X, dna_y, test_size=test_size)
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
        start = time.time()
        model = MLPClassifier(hidden_layer_sizes=layers, activation=activation, alpha=alpha, learning_rate=lr, max_iter=1000)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        end = time.time()
        with open('dna_test_size.txt', 'a') as f:
            f.write("scale=%s, test_size=%f, layers=%s, activation=%s, alpha=%f, lr=%s, dna_dim=%d, score=%f, time=%fs\n" % (str(scale), test_size, str(layers), activation, alpha, lr, dna_dim, score, end - start))
    
    scale = False
    test_size = 0.4
    # layers
    activation = 'relu'
    alpha = 0.0001
    lr = 'adaptive'
    cancer_dim = 10
    dna_dim = 180
    for layers in params_mlp['layers']:
        X_train, X_test, y_train, y_test = train_test_split(cancer_X, cancer_y, test_size=test_size)
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
        start = time.time()
        model = MLPClassifier(hidden_layer_sizes=layers, activation=activation, alpha=alpha, learning_rate=lr, max_iter=1000)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        end = time.time()
        with open('cancer_layers.txt', 'a') as f:
            f.write("scale=%s, test_size=%f, layers=%s, activation=%s, alpha=%f, lr=%s, cancer_dim=%d, score=%f, time=%fs\n" % (str(scale), test_size, str(layers), activation, alpha, lr, cancer_dim, score, end - start))
        X_train, X_test, y_train, y_test = train_test_split(dna_X, dna_y, test_size=test_size)
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
        start = time.time()
        model = MLPClassifier(hidden_layer_sizes=layers, activation=activation, alpha=alpha, learning_rate=lr, max_iter=1000)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        end = time.time()
        with open('dna_layers.txt', 'a') as f:
            f.write("scale=%s, test_size=%f, layers=%s, activation=%s, alpha=%f, lr=%s, dna_dim=%d, score=%f, time=%fs\n" % (str(scale), test_size, str(layers), activation, alpha, lr, dna_dim, score, end - start))
    
    scale = False
    test_size = 0.4
    layers = (100, )
    # activation
    alpha = 0.0001
    lr = 'adaptive'
    cancer_dim = 10
    dna_dim = 180
    for activation in params_mlp['activation']:
        X_train, X_test, y_train, y_test = train_test_split(cancer_X, cancer_y, test_size=test_size)
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
        start = time.time()
        model = MLPClassifier(hidden_layer_sizes=layers, activation=activation, alpha=alpha, learning_rate=lr, max_iter=1000)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        end = time.time()
        with open('cancer_activation.txt', 'a') as f:
            f.write("scale=%s, test_size=%f, layers=%s, activation=%s, alpha=%f, lr=%s, cancer_dim=%d, score=%f, time=%fs\n" % (str(scale), test_size, str(layers), activation, alpha, lr, cancer_dim, score, end - start))
        X_train, X_test, y_train, y_test = train_test_split(dna_X, dna_y, test_size=test_size)
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
        start = time.time()
        model = MLPClassifier(hidden_layer_sizes=layers, activation=activation, alpha=alpha, learning_rate=lr, max_iter=1000)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        end = time.time()
        with open('dna_activation.txt', 'a') as f:
            f.write("scale=%s, test_size=%f, layers=%s, activation=%s, alpha=%f, lr=%s, dna_dim=%d, score=%f, time=%fs\n" % (str(scale), test_size, str(layers), activation, alpha, lr, dna_dim, score, end - start))
    
    scale = False
    test_size = 0.4
    layers = (100, )
    activation = 'relu'
    # alpha
    lr = 'adaptive'
    cancer_dim = 10
    dna_dim = 180
    for alpha in params_mlp['alpha']:
        X_train, X_test, y_train, y_test = train_test_split(cancer_X, cancer_y, test_size=test_size)
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
        start = time.time()
        model = MLPClassifier(hidden_layer_sizes=layers, activation=activation, alpha=alpha, learning_rate=lr, max_iter=1000)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        end = time.time()
        with open('cancer_alpha.txt', 'a') as f:
            f.write("scale=%s, test_size=%f, layers=%s, activation=%s, alpha=%f, lr=%s, cancer_dim=%d, score=%f, time=%fs\n" % (str(scale), test_size, str(layers), activation, alpha, lr, cancer_dim, score, end - start))
        X_train, X_test, y_train, y_test = train_test_split(dna_X, dna_y, test_size=test_size)
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
        start = time.time()
        model = MLPClassifier(hidden_layer_sizes=layers, activation=activation, alpha=alpha, learning_rate=lr, max_iter=1000)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        end = time.time()
        with open('dna_alpha.txt', 'a') as f:
            f.write("scale=%s, test_size=%f, layers=%s, activation=%s, alpha=%f, lr=%s, dna_dim=%d, score=%f, time=%fs\n" % (str(scale), test_size, str(layers), activation, alpha, lr, dna_dim, score, end - start))
    
    scale = False
    test_size = 0.4
    layers = (100, )
    activation = 'relu'
    alpha = 0.0001
    # lr
    cancer_dim = 10
    dna_dim = 180
    for lr in params_mlp['lr']:
        X_train, X_test, y_train, y_test = train_test_split(cancer_X, cancer_y, test_size=test_size)
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
        start = time.time()
        model = MLPClassifier(hidden_layer_sizes=layers, activation=activation, alpha=alpha, learning_rate=lr, max_iter=1000)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        end = time.time()
        with open('cancer_lr.txt', 'a') as f:
            f.write("scale=%s, test_size=%f, layers=%s, activation=%s, alpha=%f, lr=%s, cancer_dim=%d, score=%f, time=%fs\n" % (str(scale), test_size, str(layers), activation, alpha, lr, cancer_dim, score, end - start))
        X_train, X_test, y_train, y_test = train_test_split(dna_X, dna_y, test_size=test_size)
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
        start = time.time()
        model = MLPClassifier(hidden_layer_sizes=layers, activation=activation, alpha=alpha, learning_rate=lr, max_iter=1000)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        end = time.time()
        with open('dna_lr.txt', 'a') as f:
            f.write("scale=%s, test_size=%f, layers=%s, activation=%s, alpha=%f, lr=%s, dna_dim=%d, score=%f, time=%fs\n" % (str(scale), test_size, str(layers), activation, alpha, lr, dna_dim, score, end - start))
    
    scale = False
    test_size = 0.4
    layers = (100, )
    activation = 'relu'
    alpha = 0.0001
    lr = 'adaptive'
    # cancer_dim
    # dna_dim
    for cancer_dim in params_mlp['cancer_dim']:
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
        model = MLPClassifier(hidden_layer_sizes=layers, activation=activation, alpha=alpha, learning_rate=lr, max_iter=1000)
        model.fit(X_train_new, y_train)
        score = model.score(X_test_new, y_test)
        end = time.time()
        with open('cancer_dim.txt', 'a') as f:
            f.write("scale=%s, test_size=%f, layers=%s, activation=%s, alpha=%f, lr=%s, cancer_dim=%d, score=%f, time=%fs\n" % (str(scale), test_size, str(layers), activation, alpha, lr, cancer_dim, score, end - start))
    for dna_dim in params_mlp['dna_dim']:
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
        model = MLPClassifier(hidden_layer_sizes=layers, activation=activation, alpha=alpha, learning_rate=lr, max_iter=1000)
        model.fit(X_train_new, y_train)
        score = model.score(X_test_new, y_test)
        end = time.time()
        with open('dna_dim.txt', 'a') as f:
            f.write("scale=%s, test_size=%f, layers=%s, activation=%s, alpha=%f, lr=%s, dna_dim=%d, score=%f, time=%fs\n" % (str(scale), test_size, str(layers), activation, alpha, lr, dna_dim, score, end - start))

def runSVMLarge():
    X_train = np.load('data/cifar_X_train.npy')
    X_test = np.load('data/cifar_X_test.npy')
    y_train = np.load('data/cifar_y_train.npy')
    y_test = np.load('data/cifar_y_test.npy')
    model = SVC(C=5.0, kernel='rbf')
    scaler = StandardScaler()

    # Exp 1: SVM: RBF, C=5.0
    print('Exp-1')
    start = time.time()
    model = SVC(C=5.0, kernel='rbf')
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    end = time.time()
    with open('CIFAR.txt', 'a') as f:
        f.write("SVM C=5.0, RBF kernel, score=%f, time=%f\n" % (score, end - start))
    # Exp 2: Z-score + SVM
    print('Exp-2')
    start = time.time()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    model = SVC(C=5.0, kernel='rbf')
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    end = time.time()
    with open('CIFAR.txt', 'a') as f:
        f.write("SVM C=5.0, RBF kernel, Z-score, score=%f, time=%f\n" % (score, end - start))
    # Exp 3: PCA-50/500/1500 + SVM
    print('Exp-3')
    dim_range = [50, 500, 1500]
    for k in dim_range:
        print("k=%d" % (k))
        start = time.time()
        pca = PCA(n_components=k)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
        model = SVC(C=5.0, kernel='rbf')
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        end = time.time()
        with open('CIFAR.txt', 'a') as f:
            f.write("SVM C=5.0, RBF kernel, PCA-%d, score=%f, time=%f\n" % (k, score, end - start))

# runSVM()
# runMLP()
runSVMLarge()