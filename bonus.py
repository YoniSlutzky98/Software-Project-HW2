from operator import mod
from pyexpat import model
import sklearn
import matplotlib.pyplot as plt
import numpy as np

def main():
    obs = sklearn.datasets.load_iris()
    inertions = []
    ks = []
    for k in range(1,11):
        inertia = sklearn.cluster.KMeans(k, init="k-means++", random_state=0).fit(obs).inertia
        inertions.append(inertia)
        ks.append(k)
    plt.plot(ks, inertions)
    plt.show()

if __name__ == "__main__":
    main()