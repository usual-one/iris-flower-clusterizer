import random
import sys

from src.irisclustering import *

def main() -> None:
    if len(sys.argv) < 2:
        print('No input file was provided')
        return
    # reading csv
    input_path = sys.argv[1]
    irises = read_irises(input_path)
    iris_types = list(set([iris[1] for iris in irises]))
    iris_dict = {t: [iris for iris in irises if iris[1] == t] for t in iris_types}

    # shuffling
    random.shuffle(irises)

    # clustering
    clusters = cluster_irises(irises)

    # classifying
    clustered_iris_dict = classify_clustered_irises(clusters)

    # calculating accuracy
    correct = 0
    for t in iris_types:
        for iris in clustered_iris_dict[t]:
            if t == iris[1]:
                correct += 1
    print(f'{round(correct / len(irises) * 100, 2)}%')

    # visualizing
    visualize_irises(iris_dict, clustered_iris_dict)


if __name__ == '__main__':
    main()
