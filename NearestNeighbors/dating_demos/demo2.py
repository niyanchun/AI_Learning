# coding=utf-8

import numpy as np
from sklearn import neighbors

from demo1 import file2matrix

data_set, labels = file2matrix("dataSet.txt")
training_set = data_set[200:, :]
training_labels = np.array(labels[200:])
testing_set = data_set[:200, :]
testing_labels = np.array(labels[:200])

clf = neighbors.KNeighborsClassifier(n_neighbors=3)
clf.fit(training_set, training_labels)

predicted_label = clf.predict(testing_set)
print("predicted label:\n", predicted_label)

score = clf.score(testing_set, testing_labels)
print("score: %f" % score)

print(clf.predict_proba(testing_set))
