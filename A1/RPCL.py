import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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
                    mean.append(np.random.normal(random.randint(0, self.class_num), 2))
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
            plt.savefig('resRPCL/Data_'+str(self.class_num)+'_'+str(self.data_num)+'.jpg')

class RPCL(object):
    def __init__(self, max_k=5, eta=0.05, theta=1.0, gamma=0.1, dataset=None):
        self.max_k = max_k
        self.eta = eta
        self.theta = theta
        self.gamma = gamma
        self.data = dataset.data
        self.label = dataset.label
        self.dim = dataset.dim
        self.bestK = 0
        self.data_num = np.shape(dataset.data)[0]
        self.centers = None
        self.p = None #p[n][k]
        self.disMat = np.zeros((self.data_num, self.max_k)) #dist[n][k]
    
    def distance(self, x, y):
        return np.sqrt(np.sum(np.square(x - y)))

    def fit(self, max_iter=1000):
        #Init
        centers = []
        for i in range(self.max_k):
            center = []
            for d in range(self.dim):
                center.append(np.random.normal(2, 1))
            centers.append(center)
        self.centers = np.array(centers)
        self.draw(centers=self.centers, fname='Init3')

        #Loop
        winnerIdx = np.zeros((self.data_num, 1), dtype=np.int64)
        rivalIdx = np.zeros((self.data_num, 1), dtype=np.int64)
        for i in range(max_iter):
            self.p = np.zeros((self.data_num, self.max_k))
            for n in range(self.data_num):
                for k in range(self.max_k):
                    self.disMat[n][k] = self.distance(self.data[n], self.centers[k])

                tmp = np.argsort(self.disMat[n])
                winnerIdx[n] = tmp[0]
                rivalIdx[n] = tmp[1]
                self.p[n][winnerIdx[n]] = 1.0
                self.p[n][rivalIdx[n]] = -self.gamma

                self.centers[winnerIdx[n]] += self.p[n][winnerIdx[n]] * self.eta * (self.data[n] - self.centers[winnerIdx[n]])
                self.centers[rivalIdx[n]] += self.p[n][rivalIdx[n]] * self.eta * (self.data[n] - self.centers[rivalIdx[n]]) 

        #Select
        repel_list = []
        for k in range(self.max_k):
            if np.min(self.disMat[:,k]) > self.theta:
                repel_list.append(k)
        self.centers = np.delete(self.centers, repel_list, axis=0)
        self.bestK = self.centers.shape[0]
            
    def draw(self, ifShowCenter=True, ifShowLabel=False, label=None, centers=None, fname='', title='RPCL'):
        if ifShowLabel:
            data_2d = pd.DataFrame(self.data, columns=['x', 'y'])
            label_2d = pd.DataFrame(label, columns=['label'])
            label_names = np.unique(label)
            colors = [plt.cm.tab10(i/float(len(label_names)+1)) for i in range(len(label_names)+1)]
            tmp_2d = pd.concat([data_2d, label_2d], axis=1)

            plt.figure()
            for i, label in enumerate(label_names):
                plt.scatter(tmp_2d.loc[tmp_2d.label==label].x, tmp_2d.loc[tmp_2d.label==label].y, s=5, cmap=colors[i], alpha=0.5)
            if ifShowCenter:
                plt.scatter(centers[:,0], centers[:,1], s=50, cmap=colors[len(label_names)], alpha=1.0)
            plt.title(title)
            # plt.show()
            plt.savefig('resRPCL/' + fname + '_labeled.jpg')
        else:
            data_2d = pd.DataFrame(self.data, columns=['x', 'y'])
            colors = [plt.cm.tab10(i/float(self.max_k+1)) for i in range(self.max_k+1)]

            plt.figure()
            plt.scatter(data_2d.x, data_2d.y, s=5, cmap=colors[0], alpha=0.5)
            if ifShowCenter:
                plt.scatter(centers[:,0], centers[:,1], s=50, cmap=colors[self.max_k], alpha=1.0)
            plt.title(title)
            # plt.show()
            plt.savefig('resRPCL/' + fname + '.jpg')

def experiment(dataset, RPCLmodel):
    k = RPCLmodel.bestK
    centers = RPCLmodel.centers
    model = KMeans(n_clusters=k, init=centers)
    model.fit(dataset.data)
    RPCLmodel.draw(ifShowLabel=True, label=model.labels_, centers=model.cluster_centers_, fname='KmeansP3', title='K-means')
    if k - 1 > 0:
        model = KMeans(n_clusters=k-1)
        model.fit(dataset.data)
        RPCLmodel.draw(ifShowLabel=True, label=model.labels_, centers=model.cluster_centers_, fname='KmeansN-1_3', title='K-means, k=k*-1')
    model = KMeans(n_clusters=k+1)
    model.fit(dataset.data)
    RPCLmodel.draw(ifShowLabel=True, label=model.labels_, centers=model.cluster_centers_, fname='KmeansN+1_3', title='K-means, k=k*+1')


def main():
    dataset = Dataset(class_num=3, data_num=500)
    dataset.generate()
    model = RPCL(eta=0.001, theta=0.1, gamma=0.05, dataset=dataset)
    model.fit(max_iter=500)
    model.draw(centers=model.centers, fname='RPCL3')
    model.draw(centers=model.centers, ifShowLabel=True, label=model.label, fname='RPCL3')
    experiment(dataset, model)

if __name__ == '__main__':
    main()
    