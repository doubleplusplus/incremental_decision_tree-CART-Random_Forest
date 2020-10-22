"""
Random Forest implementation with CART decision trees
This version is for continuous dataset (feature values)

Author: Jamie
Date: 09/10/2018
"""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from multiprocessing import Pool
from random import sample
import time
np.seterr(divide='ignore', invalid='ignore')  # ignore Runtime Warning about divide


class TreeNode:
    def __init__(self, n_features):
        self.n_features = n_features
        self.left_child = None
        self.right_child = None
        self.split_feature = None
        self.split_value = None
        self.split_gini = 1
        self.label = None

    def is_leaf(self):
        return self.label is not None

    """ use 2d array (matrix) to compute gini index. Numerical feature values only """
    def gini(self, f, y, target):
        trans = f.reshape(len(f), -1)  # transpose 1d np array
        a = np.concatenate((trans, target), axis=1)  # vertical concatenation
        a = a[a[:, 0].argsort()]  # sort by column 0, feature values
        sort = a[:, 0]
        split = (sort[0:-1] + sort[1:]) / 2  # compute possible split values

        left, right = np.array([split]), np.array([split])
        classes, counts = np.unique(y, return_counts=True)
        n_classes = len(classes)
        # count occurrence of labels for each possible split value
        for i in range(n_classes):
            temp = a[:, -n_classes + i].cumsum()[:-1]
            left = np.vstack((left, temp))  # horizontal concatenation
            right = np.vstack((right, counts[i] - temp))

        sum_1 = left[1:, :].sum(axis=0)  # sum occurrence of labels
        sum_2 = right[1:, :].sum(axis=0)
        n = len(split)
        gini_t1, gini_t2 = [1] * n, [1] * n
        # calculate left and right gini
        for i in range(n_classes):
            gini_t1 -= (left[i + 1, :] / sum_1) ** 2
            gini_t2 -= (right[i + 1, :] / sum_2) ** 2
        s = sum(counts)
        g = gini_t1 * sum_1 / s + gini_t2 * sum_2 / s
        g = list(g)
        min_g = min(g)
        split_value = split[g.index(min_g)]
        return split_value, min_g

    def split_feature_value(self, x, y, target):
        # compute gini index of every column
        n = x.shape[1]  # number of x columns
        sub_features = sample(range(n), self.n_features)  # feature sub-space
        # list of (split_value, split_gini) tuples
        value_g = [self.gini(x[:, i], y, target) for i in sub_features]
        result = min(value_g, key=lambda t: t[1])  # (value, gini) tuple with min gini
        feature = sub_features[value_g.index(result)]  # feature with min gini
        return feature, result[0], result[1]  # split feature, value, gini

    # recursively grow the tree
    def attempt_split(self, x, y, target):
        c = Counter(y)
        majority = c.most_common()[0]  # majority class and count
        label, count = majority[0], majority[1]
        if len(y) < 2 or len(c) == 1 or count/len(y) > 0.9:  # stop criterion
            self.label = label  # set leaf
            return
        # split feature, value, gini
        feature, value, split_gini = self.split_feature_value(x, y, target)
        # stop split when gini decrease smaller than some threshold
        if self.split_gini - split_gini < 0.01:  # stop criterion
            self.label = label  # set leaf
            return
        index1 = x[:, feature] <= value
        index2 = x[:, feature] > value
        x1, y1, x2, y2 = x[index1], y[index1], x[index2], y[index2]
        target1, target2 = target[index1], target[index2]
        if len(y2) == 0 or len(y1) == 0:  # stop split
            self.label = label  # set leaf
            return
        # splitting procedure
        self.split_feature = feature
        self.split_value = value
        self.split_gini = split_gini
        self.left_child, self.right_child = TreeNode(self.n_features), TreeNode(self.n_features)
        self.left_child.split_gini, self.right_child.split_gini = split_gini, split_gini
        self.left_child.attempt_split(x1, y1, target1)
        self.right_child.attempt_split(x2, y2, target2)

    # trace down the tree for each data instance, for prediction
    def sort(self, x):  # x is 1d array
        if self.label is not None:
            return self.label
        if x[self.split_feature] <= self.split_value:
            return self.left_child.sort(x)
        else:
            return self.right_child.sort(x)


class ClassifierTree:
    def __init__(self, n_features):
        self.root = TreeNode(n_features)

    def train(self, x, y):
        # one hot encoded target is for gini index calculation
        # categories='auto' silence future warning
        encoder = OneHotEncoder(categories='auto')
        labels = y.reshape(len(y), -1)  # transpose 1d np array
        target = encoder.fit_transform(labels).toarray()
        self.root.attempt_split(x, y, target)

    def classify(self, x):  # x is 2d array
        return [self.root.sort(x[i]) for i in range(x.shape[0])]


class RandomForest:
    def __init__(self, n_classifiers=30):
        self.n_classifiers = n_classifiers
        self.classifiers = []
        self.x = None
        self.y = None

    def build_tree(self, tree):
        n = len(self.y)  # n for bootstrap sampling size
        # n = int(n * 0.5)
        x, y = resample(self.x, self.y, n_samples=n)  # bootstrap sampling
        tree.train(x, y)
        return tree  # return tree for multiprocessing pool

    def fit(self, x, y):
        self.x, self.y = x, y
        n_select_features = int(np.sqrt(x.shape[1]))  # number of features
        for i in range(self.n_classifiers):
            tree = ClassifierTree(n_select_features)
            self.classifiers.append(tree)
        # multiprocessing pool
        pool = Pool()
        self.classifiers = pool.map(self.build_tree, self.classifiers)
        pool.close()
        pool.join()

    def predict(self, x_test):  # ensemble
        pred = [tree.classify(x_test) for tree in self.classifiers]
        pred = np.array(pred)
        result = [Counter(pred[:, i]).most_common()[0][0] for i in range(pred.shape[1])]
        return result


def test():
    start_time = time.time()
    # It's a continuous dataset, only numerical feature values
    df = pd.read_csv('./dataset/waveform.data', header=None, sep=',')
    data = df.values
    x = data[:, :-1]
    y = data[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    rf = RandomForest(n_classifiers=50)  # optimal 100 trees
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print('RF accuracy:', acc)
    print("--- Running time: %.6f seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    test()
