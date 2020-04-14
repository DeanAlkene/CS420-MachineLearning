import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

class Dataset(object):
    def __init__(self, class_num=3, data_num=200, center=None, dim=2):
        self.class_num = class_num
        self.data_num = data_num
        self.dim = dim
        self.data = np.zeros((class_num * data_num, dim))
        self.label = np.zeros((class_num * data_num, 1))
        if center != None:
            self.cluster_mean = center
        else:
            means = []
            for i in range(self.class_num):
                mean = []
                for d in range(self.dim):
                    mean.append(np.random.normal(random.randint(0, self.class_num), 0.5))
                means.append(mean)
            self.cluster_mean = np.array(means)
    
    def generate(self):
        for i in range(self.class_num):
            sigma = np.random.uniform(0.2, 0.4)
            for j in range(self.data_num):
                for d in range(self.dim):
                    sample_p = np.random.normal(self.cluster_mean[i][d], sigma)
                    self.data[i*self.data_num+j][d] = sample_p
                self.label[i*self.data_num+j] = i + 1
    
    def show(self):
        if self.dim == 2:
            data_2d = pd.DataFrame(self.data, columns=['x', 'y'])
            label_2d = pd.DataFrame(self.label, columns=['label'])
            label_names = [i+1 for i in range(self.class_num)]
            colors = [plt.cm.tab10(i/float(len(label_names))) for i in range(len(label_names))]
            tmp_2d = pd.concat([data_2d, label_2d], axis=1)

            plt.figure()
            for i, label in enumerate(label_names):
                plt.scatter(tmp_2d.loc[tmp_2d.label==label].x, tmp_2d.loc[tmp_2d.label==label].y, s=5, cmap=colors[i], alpha=0.5)
            plt.title('Data Generation with %d classes, %d samples each class'%(self.class_num, self.data_num))
            # plt.show()
            plt.savefig('res/Data_'+str(self.class_num)+'_'+str(self.data_num)+'.jpg')

class EM(object):
    def __init__(self, n_components=5, dataset=None):
        self.model = None
        self.n_components = n_components
        self.class_num = dataset.class_num
        self.data_num = dataset.data_num
        self.data = dataset.data
        self.label = dataset.label
        self.AIC_score = []
        self.BIC_score = []
        self.bestAIC_k = 0
        self.bestBIC_k = 0

    def AIC_selection(self):
        for n_comp in range(1, self.n_components+1):
            GMM = GaussianMixture(n_components=n_comp, max_iter=10000)
            GMM.fit(self.data)
            self.AIC_score.append(GMM.aic(self.data))
        self.bestAIC_k = np.argmin(self.AIC_score) + 1
        print("Best k by AIC: %d"%(self.bestAIC_k))
        self.model = GaussianMixture(n_components=self.bestAIC_k, max_iter=10000)
        self.model.fit(self.data)

    def BIC_selection(self):
        for n_comp in range(1, self.n_components+1):
            GMM = GaussianMixture(n_components=n_comp, max_iter=10000)
            GMM.fit(self.data)
            self.BIC_score.append(GMM.bic(self.data))
        self.bestBIC_k = np.argmin(self.BIC_score) + 1
        print("Best k by BIC: %d"%(self.bestBIC_k))
        self.model = GaussianMixture(n_components=self.bestBIC_k, max_iter=10000)
        self.model.fit(self.data)

    def draw(self, suffix):
        label = self.model.predict(self.data)
        data_2d = pd.DataFrame(self.data, columns=['x', 'y'])
        label_2d = pd.DataFrame(label, columns=['label'])
        label_names = np.unique(label)
        colors = [plt.cm.tab10(i/float(len(label_names))) for i in range(len(label_names))]
        tmp_2d = pd.concat([data_2d, label_2d], axis=1)

        plt.figure()
        for i, label in enumerate(label_names):
            plt.scatter(tmp_2d.loc[tmp_2d.label==label].x, tmp_2d.loc[tmp_2d.label==label].y, s=5, cmap=colors[i], alpha=0.5)
        plt.title('Best GMM with ' + suffix + '_' + str(self.class_num) + '_' + str(self.data_num))
        plt.savefig('res/GMM_'+ suffix + '_' + str(self.class_num) + '_' + str(self.data_num) + '.jpg')

class VBEM(object):
    def __init__(self, n_components=5, dataset=None):
        self.model = BayesianGaussianMixture(n_components=n_components, max_iter=10000)
        self.n_components = n_components
        self.class_num = dataset.class_num
        self.data_num = dataset.data_num
        self.data = dataset.data
        self.label = dataset.label
        self.bestVBEM_k = 0
        self.model.fit(self.data)
    
    def draw(self):
        label = self.model.predict(self.data)
        self.bestVBEM_k = max(label) + 1
        data_2d = pd.DataFrame(self.data, columns=['x', 'y'])
        label_2d = pd.DataFrame(label, columns=['label'])
        label_names = np.unique(label)
        colors = [plt.cm.tab10(i/float(len(label_names))) for i in range(len(label_names))]
        tmp_2d = pd.concat([data_2d, label_2d], axis=1)

        plt.figure()
        for i, label in enumerate(label_names):
            plt.scatter(tmp_2d.loc[tmp_2d.label==label].x, tmp_2d.loc[tmp_2d.label==label].y, s=5, cmap=colors[i], alpha=0.5)
        plt.title('Best GMM with VBEM_' + str(self.class_num) + '_' + str(self.data_num))
        plt.savefig('res/GMM_VBEM_' + str(self.class_num) + '_' + str(self.data_num) + '.jpg')

def experiment():
    for class_num in [1, 2, 3, 5, 8]:
        for data_num in [200, 500, 1000]:
            dataset = Dataset(class_num=class_num, data_num=data_num)
            dataset.generate()
            dataset.show()
            em = EM(n_components=10, dataset=dataset)
            em.AIC_selection()
            em.draw('AIC')
            em.BIC_selection()
            em.draw('BIC')
            vbem = VBEM(n_components=10, dataset=dataset)
            vbem.draw()
            with open('res/res.txt', 'a') as f:
                f.write("cluster_num: %d, data_num_per_cluster: %d\n"%(class_num, data_num))
                f.write('Best k, AIC: %d\n'%(em.bestAIC_k))
                f.write('Best k, BIC: %d\n'%(em.bestBIC_k))
                f.write('Best k, VBEM: %d\n\n'%(vbem.bestVBEM_k))

def main():
    experiment()

if __name__ == '__main__':
    main()
