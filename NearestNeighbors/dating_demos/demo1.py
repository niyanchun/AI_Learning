"""
kNN demo1
"""

import operator

import matplotlib.pyplot as plt
import numpy as np


def create_dataset():
    features = np.array([[1.0, 4.0], [2.5, 2.7], [4.3, 8.0], [5.2, 7.5]])
    labels = ['A', 'A', 'B', 'B']
    return features, labels


def draw_dataset():
    f, _ = create_dataset()
    x = np.append(f[:, 0], 2.0)
    y = np.append(f[:, 1], 5.0)
    plt.scatter(x, y, c=['b', 'b', 'r', 'r', 'k'])
    plt.show()


def file2matrix(filename):
    fr = open(filename)
    lines = fr.readlines()
    number_of_lines = len(lines)
    features = np.zeros((number_of_lines, 3))
    labels = []
    index = 0
    for line in lines:
        line = line.strip()
        data = line.split('\t')
        features[index, :] = data[0:3]
        labels.append(int(data[-1]))
        index += 1
    return features, labels


def auto_norm(dataset):
    min_val = dataset.min(0)
    max_val = dataset.max(0)
    ranges = max_val - min_val
    m = dataset.shape[0]
    norm_dataset = dataset - np.tile(min_val, (m, 1))
    norm_dataset = norm_dataset / np.tile(ranges, (m, 1))

    return norm_dataset, ranges, min_val


def classify0(x, dataset, labels, k):
    dataset_size = dataset.shape[0]
    diff_mat = np.tile(x, (dataset_size, 1)) - dataset
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sorted_dist_indicies = distances.argsort()
    class_count = {}
    for i in range(k):
        voted_label = labels[sorted_dist_indicies[i]]
        class_count[voted_label] = class_count.get(voted_label, 0) + 1
    # Python 2.x 里面用`class_count.iteritems()`
    sorted_class_count = sorted(class_count.items(),
                                key=operator.itemgetter(1),
                                reverse=True)
    return sorted_class_count[0][0]


def dating_class_test():
    ratio = 0.1
    dataset, labels = file2matrix('dataSet.txt')
    norm_mat, ranges, min_vals = auto_norm(dataset)
    m = norm_mat.shape[0]
    num_test = int(m * ratio)
    err_count = 0.0
    for i in range(num_test):
        classifier_result = classify0(norm_mat[i, :],
                                      norm_mat[num_test:m, :],
                                      labels[num_test:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" %
              (classifier_result, labels[i]))
        if classifier_result != labels[i]:
            err_count += 1.0
    print("the total error rate is: %f" % (err_count / num_test))
    print(err_count)


def classify_person():
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(input('percentage of time spent playing video games?'))
    ff_miles = float(input('frequent flier miles earned per year?'))
    ice_cream = float(input('liters of ice cream consumed per year?'))
    dataset, labels = file2matrix('dataSet.txt')
    norm_mat, ranges, min_vals = auto_norm(dataset)
    in_array = np.array([ff_miles, percent_tats, ice_cream])
    classifier_result = classify0(in_array - min_vals / ranges, norm_mat, labels, 3)
    print("You will probably like this person: ", result_list[classifier_result - 1])


def draw_f1f2(dataset, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = dataset[:, 1]
    y = dataset[:, 2]
    ax.scatter(x, y, 15.0 * np.array(labels), 15.0 * np.array(labels), label='test')
    ax.legend(loc='best')
    plt.show()


def draw_f0f1(dataset, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataset[:, 0], dataset[:, 1],
               15.0 * np.array(labels), 15.0 * np.array(labels))
    plt.show()


def draw_both(dataset, labels):
    fig = plt.figure(figsize=(20, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(dataset[:, 1], dataset[:, 2],
                15.0 * np.array(labels), 15.0 * np.array(labels))
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(dataset[:, 0], dataset[:, 1],
                15.0 * np.array(labels), 15.0 * np.array(labels))
    plt.show()


if __name__ == "__main__":
    # draw_dataset()
    dataset, labels = file2matrix("dataSet.txt")
    # draw_f1f2(dataset, labels)
    # draw_f0f1(dataset, labels)
    draw_both(dataset, labels)
    # dating_class_test()
    # classify_person()
