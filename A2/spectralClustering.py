import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, cluster, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

def drawScatter(data, label, class_num, method_name, dataset_name):
    data_2d = pd.DataFrame(data, columns=['x', 'y'])
    label_2d = pd.DataFrame(label, columns=['label'])
    label_names = [i for i in range(class_num)]
    colors = [plt.cm.tab10(i / float(len(label_names))) for i in range(len(label_names))]
    tmp_2d = pd.concat([data_2d, label_2d], axis=1)

    plt.figure()
    for i, label in enumerate(label_names):
        plt.scatter(tmp_2d.loc[tmp_2d.label==label].x, tmp_2d.loc[tmp_2d.label==label].y, s=5, cmap=colors[i], alpha=0.5)
    plt.title(method_name + ' ' + dataset_name)
    plt.savefig('res/' + method_name + '_' + dataset_name)

def main():
    n_samples = 5000
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                        noise=.05)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=10)
    no_structure = np.random.rand(n_samples, 2), None

    # Anisotropicly distributed data
    random_state = 160
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = datasets.make_blobs(n_samples=n_samples,
                                cluster_std=[1.0, 2.5, 0.5],
                                random_state=random_state)

    data = [
        (noisy_circles, {'name': 'Circle', 'n_clusters': 2}),
        (noisy_moons, {'name': 'Moon', 'n_clusters': 2}),
        (varied, {'name': 'Varied', 'n_clusters': 3}),
        (aniso, {'name': 'Aniso', 'n_clusters': 3}),
        (blobs, {'name': 'Blobs', 'n_clusters': 3}),
        (no_structure, {'name': 'NoStructure', 'n_clusters': 3})]

    for (dataset, param) in data:
        X, y = dataset
        X = StandardScaler().fit_transform(X)
        spectral = cluster.SpectralClustering(
            n_clusters=param['n_clusters'], eigen_solver='arpack',
            affinity="nearest_neighbors")
        y_pred = spectral.fit_predict(X)
        drawScatter(X, y_pred, param['n_clusters'], 'SpectralClustering', param['name'])

        gmm = mixture.GaussianMixture(
            n_components=param['n_clusters'], covariance_type='full')
        y_pred = gmm.fit_predict(X)
        drawScatter(X, y_pred, param['n_clusters'], 'GMM', param['name'])

if __name__ == '__main__':
    main()