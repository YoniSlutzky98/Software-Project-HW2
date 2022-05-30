from turtle import color
from sklearn import cluster
from sklearn import datasets as ds
import matplotlib.pyplot as plt
import numpy as np

def main():
    obs = ds.load_iris().data
    inertions = []
    ks = []
    for k in range(1,11):
        inertia = cluster.KMeans(n_clusters = k, init = "k-means++", 
        random_state = 0).fit(obs).inertia_
        inertions.append(inertia)
        ks.append(k)
    plt.plot(ks, inertions)
    plt.title('Elbow Method for Selection of Optimal "K" Clusters',
     fontdict={'family': 'david',
        'color':  'brown',
        'weight': 'bold',
        'size': 18,
        })
    plt.xlabel('K', fontdict={'size':14})
    plt.ylabel('Inertia', fontdict={'size':14})
    plt.locator_params('x', nbins = 10)
    ax = plt.gca()
    ax.plot(3, 90, 'o', ms=25, mec='black', mfc = 'none', mew = 3)
    ax.annotate('Elbow Point', xy=(3, 80), xytext=(5, 190),
    color='black', size='large',arrowprops=dict(
      arrowstyle='simple,tail_width=0.3,head_width=0.8,head_length=0.8',
      facecolor='black', shrinkB=20)
    )
    plt.savefig('elbow.png')

if __name__ == "__main__":
    main()