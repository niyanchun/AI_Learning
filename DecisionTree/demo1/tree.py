import numpy as np
from math import log
import operator


def create_dataset():
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']  # features_name
    return dataset, labels


def calc_shannon_ent(dataset):
    num_entries = len(dataset)
    label_counts = {}
    for line in dataset:
        current_label = line[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)

    return shannon_ent


def split_dataset(dataset, axis, value):
    """
    按照给定特征划分数据集
    :param dataset: 待划分数据集
    :param axis: 划分数据集的特征（比如按照f1，还是f2划分）
    :param value: 需要返回的特征值（比如axis取f1时，是返回f1=0还是f1=1）
    :return:
    """
    ret_dataset = []
    for line in dataset:
        if line[axis] == value:
            reduced_line = line[:axis]
            reduced_line.extend(line[axis + 1:])
            ret_dataset.append(reduced_line)
    return ret_dataset


def choose_best_feature_to_split(dataset):
    num_features = len(dataset[0]) - 1
    base_entropy = calc_shannon_ent(dataset)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_features):
        feat_list = [example[i] for example in dataset]
        unique_vals = set(feat_list)
        new_entropy = 0.0
        for value in unique_vals:
            sub_dataset = split_dataset(dataset, i, value)
            prob = len(sub_dataset) / float(len(dataset))
            new_entropy += prob * calc_shannon_ent(sub_dataset)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
            class_count[vote] += 1
        sorted_class_count = sorted(class_count.items(),
                                    key=operator.itemgetter(1),
                                    reverse=True)
        return sorted_class_count[0][0]


def create_tree(dataset, labels):
    class_list = [example[-1] for example in dataset]
    # 类别完全相同则停止继续划分
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 遍历完所有特征时返回出现次数最多的类别，比如剩('yes'),('yes'),('no')
    if len(dataset[0]) == 1:
        return majority_cnt(class_list)
    best_feature = choose_best_feature_to_split(dataset)
    best_feature_label = labels[best_feature]
    my_tree = {best_feature_label: {}}
    del (labels[best_feature])
    feature_values = [example[best_feature] for example in dataset]
    unique_values = set(feature_values)
    for value in unique_values:
        sub_labels = labels[:]
        my_tree[best_feature_label][value] = create_tree(
            split_dataset(dataset, best_feature, value),
            sub_labels)
    return my_tree


if __name__ == '__main__':
    dataset, labels = create_dataset()

    # calc_shannon_ent test
    # print(calc_shannon_ent(dataset))

    # split_dataset test
    # print(split_dataset(dataset, 0, 1))
    # print(split_dataset(dataset, 0, 0))
    #
    # # choose_best_feature_to_split test
    print(choose_best_feature_to_split(dataset))
    #
    # # create_tree test
    # print(create_tree(dataset, labels))
