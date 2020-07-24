import sys
import csv
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from process_data import process_data

config = {2: {'single': [2, 3, 7, 9], 'average': [2, 4, 6]}, 3: {
    'single': [2, 3, 4, 5], 'average': [2, 3, 4, 7]}}


def plot_2d(data, labels, n, linkage):
    data = np.array(data)
    plt.figure()
    plt.title(f'Clustering - Linkage: {linkage}')
    plt.scatter(data[:, 0], data[:, 1], c=labels, marker='.')
    plt.savefig(f'images/hierarchical/2d_{linkage}_{n}.png')


def plot_3d(data, labels, n, linkage):
    data = np.array(data)
    fig = plt.figure()
    ax = Axes3D(fig)
    plt.title(f'Clustering - Linkage: {linkage}')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, marker='.')
    plt.savefig(f'images/hierarchical/3d_{linkage}_{n}.png')


def main():
    if len(sys.argv) < 2:
        print('usage: python hierarchical.py <data>')
        sys.exit(1)

    data = process_data(sys.argv[1])
    dimension = len(data[0])

    for linkage in ['single', 'average']:
        for n in config[dimension][linkage]:
            clustering = AgglomerativeClustering(
                linkage=linkage, n_clusters=n).fit(data)

            if dimension == 2:
                plot_2d(data, clustering.labels_, n, linkage)
            elif dimension == 3:
                plot_3d(data, clustering.labels_, n, linkage)


if __name__ == '__main__':
    main()
