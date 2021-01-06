import math
import random
import sys
import csv

import scipy.cluster


def main() -> None:
    if len(sys.argv) < 2:
        print('No input file was provided')
        return
    # reading csv
    input_path = sys.argv[1]
    with open(input_path) as file:
        irises = [([float(feature) for feature in iris[:4]], iris[4]) for iris in list(csv.reader(file, delimiter=','))[1:]]


    # shuffling
    random.shuffle(irises)

    vectors = [iris[0] for iris in irises]

    # clustering
    cluster_number = 3
    whitened_vectors = scipy.cluster.vq.whiten(vectors)
    centroids, mean_distance = scipy.cluster.vq.kmeans(whitened_vectors, cluster_number)
    cluster_indexes, distortion = scipy.cluster.vq.vq(whitened_vectors, centroids)

    # classification
    clusters = [[] for i in range(cluster_number)]
    for i in range(len(cluster_indexes)):
        clusters[cluster_indexes[i]].append(irises[i])

    correct = 0
    for cluster in clusters:
        classified_irises = {'setosa': len([iris for iris in cluster if iris[1] == 'setosa']),
                             'virginica': len([iris for iris in cluster if iris[1] == 'virginica']),
                             'versicolor': len([iris for iris in cluster if iris[1] == 'versicolor'])}
        correct += max(classified_irises.values())

    print(f'{round(correct / len(irises) * 100, 2)}%')


if __name__ == '__main__':
    main()
