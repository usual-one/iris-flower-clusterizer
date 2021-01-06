import itertools
import csv

import scipy.cluster
import matplotlib.pyplot as plt


def classify_clustered_irises(clusters: list) -> dict:
    irises = []
    for cluster in clusters:
        irises.extend(cluster)
    iris_types = list(set([iris[1] for iris in irises]))

    clustered_iris_dict = {t: [] for t in iris_types}
    for cluster in clusters:
        clustered_iris_dict[get_iris_types(cluster)[0]] = cluster
    return clustered_iris_dict


def get_iris_types(irises: list) -> list[str]:
    iris_types = list(set([iris[1] for iris in irises]))
    types_count = {t: 0 for t in iris_types}
    for iris in irises:
        types_count[iris[1]] += 1
    return sorted(list(types_count.keys()), key=lambda x: types_count[x], reverse=True)


def cluster_irises(irises: list) -> list:
    vectors = [iris[0] for iris in irises]

    iris_types = list(set([iris[1] for iris in irises]))
    cluster_number = len(iris_types)

    whitened_vectors = scipy.cluster.vq.whiten(vectors)
    centroids, mean_distance = scipy.cluster.vq.kmeans(whitened_vectors, cluster_number)
    cluster_indexes, distortion = scipy.cluster.vq.vq(whitened_vectors, centroids)

    clusters = [[] for i in range(cluster_number)]
    for i in range(len(cluster_indexes)):
        clusters[cluster_indexes[i]].append(irises[i])
    return clusters


def read_irises(path: str) -> list:
    with open(path) as file:
        irises = [([float(feature) for feature in iris[:4]], iris[4]) for iris in list(csv.reader(file, delimiter=','))[1:]]
    return irises


def visualize_irises(iris_dict: dict, clustered_iris_dict: dict) -> None:
    figure, ((actual_sepal, predicted_sepal), (actual_petal, predicted_petal)) = plt.subplots(2, 2, figsize=(9, 9))

    actual_sepal.set_title('Actual Sepal')
    actual_sepal.set_xlabel('Sepal Width')
    actual_sepal.set_ylabel('Sepal Length')
    predicted_sepal.set_title('Predicted Sepal')
    predicted_sepal.set_xlabel('Sepal Width')
    predicted_sepal.set_ylabel('Sepal Length')
    actual_petal.set_title('Actual Petal')
    actual_petal.set_xlabel('Petal Width')
    actual_petal.set_ylabel('Petal Length')
    predicted_petal.set_title('Predicted Petal')
    predicted_petal.set_xlabel('Petal Width')
    predicted_petal.set_ylabel('Petal Length')

    iris_types = list(iris_dict.keys())
    colors = itertools.cycle(['r', 'g', 'b'])
    types_colors = {t: next(colors) for t in iris_types}
    for t in iris_types:
        actual_sepal.scatter([iris[0][1] for iris in iris_dict[t]],
                [iris[0][0] for iris in iris_dict[t]], c=types_colors[t])
        actual_petal.scatter([iris[0][3] for iris in iris_dict[t]],
                [iris[0][2] for iris in iris_dict[t]], c=types_colors[t])
        predicted_sepal.scatter([iris[0][1] for iris in clustered_iris_dict[t]],
                [iris[0][0] for iris in clustered_iris_dict[t]], c=types_colors[t])
        predicted_petal.scatter([iris[0][3] for iris in clustered_iris_dict[t]],
                [iris[0][2] for iris in clustered_iris_dict[t]], c=types_colors[t])

    plt.show()

