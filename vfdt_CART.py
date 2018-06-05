# Very Fast Decision Tree i.e. Hoeffding Tree, described in
# "Mining High-Speed Data Streams" (Domingos & Hulten, 2000)
#
# this program contains 2 classes: vfdt, vfdt_node
# test in command line window: python3 vfdt.py
# changed to CART: Gini index
#
# Jamie
# 02/06/2018
# ver 0.03


import numpy as np
import pandas as pd
import math
import time
from itertools import combinations


# VFDT node class
class vfdt_node:
    # parameter nijk: statistics of feature i, value j, class k
    def __init__(self, possible_split_features):
        self.parent = None
        self.left_child = None
        self.right_child = None
        self.split_feature = None
        self.split_value = None
        self.new_examples_seen = 0
        self.total_examples_seen = 0
        self.class_frequency = {}
        self.nijk = {i:{} for i in possible_split_features}
        self.possible_split_features = possible_split_features

    def add_children(self, split_feature, left, right):
        self.split_feature = split_feature
        self.left_child = left
        self.right_child = right
        left.parent = self
        right.parent = self
        self.nijk.clear()  # reset stats

    # recursively trace down the tree to distribute data examples to corresponding leaves
    def sort_example(self, example):
        if (not self.is_leaf()):
            index = self.possible_split_features.index(self.split_feature)
            value = example[:-1][index]

            try:  # continous value
                if value <= self.split_value:
                    return self.left_child.sort_example(example)
                else:
                    return self.right_child.sort_example(example)
            except TypeError:  # discrete value
                if value in self.split_value[0]:
                    return self.left_child.sort_example(example)
                else:
                    return self.right_child.sort_example(example)
        else:
            return(self)

    def is_leaf(self):
        return(self.left_child == None and self.right_child == None)

    # the most frequent classification
    def most_frequent(self):
        if (self.class_frequency):
            prediction = max(self.class_frequency, key=self.class_frequency.get)
        else:
            # if self.class_frequency dict is empty, go back to parent
            class_frequency = self.parent.class_frequency
            prediction = max(class_frequency, key=class_frequency.get)
        return(prediction)

    # upadate leaf stats in order to calculate infomation gain
    def update_stats(self, example):
        label = example[-1]
        feats = self.possible_split_features
        for i in feats:
            if (i != None):
                value = example[:-1][feats.index(i)]
                if (value not in self.nijk[i]):
                    d = {label : 1}
                    self.nijk[i][value] = d
                else:
                    if (label not in self.nijk[i][value]):
                        self.nijk[i][value][label] = 1
                    else:
                        self.nijk[i][value][label] += 1
        self.total_examples_seen += 1
        self.new_examples_seen += 1
        if (label not in self.class_frequency):
            self.class_frequency[label] = 1
        else:
            self.class_frequency[label] += 1

    def check_not_splitting(self):
        # compute Gini index for not splitting
        X0 = 1
        class_frequency = self.class_frequency
        n = sum(class_frequency.values())
        for j, k in class_frequency.items():
            X0 -= (k/n)**2
        return X0

    # use hoeffding tree model to test node split, return the split feature
    def splittable(self, delta, nmin, tau):
        if(self.new_examples_seen < nmin):
            return(None)
        else:
            self.new_examples_seen = 0  # reset
        nijk = self.nijk
        min = 1
        second_min = 1
        Xa = ''
        split_value = None
        for feature in self.possible_split_features:
            if (len(nijk[feature]) != 1):
                gini, value = self.Gini(feature)
                if (gini < min):
                    min = gini
                    Xa = feature
                    split_value = value
                elif (min < gini < second_min):
                    second_min = gini
            else:
                return None

        sigma = self.hoeffding_bound(delta)
        X0 = self.check_not_splitting()
        if min < X0:
            if (second_min - min > sigma):
                return [Xa, split_value]
            elif (sigma < tau and second_min - min < tau):
                return [Xa, split_value]
            else:
                return None
        else:
            return None

    def hoeffding_bound(self, delta):
        n = self.total_examples_seen
        R = np.log(len(self.class_frequency))
        return (R * R * np.log(1/delta) / (2 * n))**0.5

    def Gini(self, feature):
        '''
        Gini(D) = 1 - Sum(pi^2)
        Gini(D, F=f) = |D1|/|D|*Gini(D1) + |D2|/|D|*Gini(D2)
        '''
        njk = self.nijk[feature]
        D = self.total_examples_seen
        class_frequency = self.class_frequency

        m1 = 1  # minimum gini
        m2 = 1  # second minimum gini
        Xa_value = None
        test = next(iter(njk))  # test j value
        # if not isinstance(test, np.object):
        try:  # continous feature values
            test += 0
            sort = sorted([j for j in njk.keys()])
            split = []
            for i in range(1, len(sort)):
                temp = (sort[i-1] + sort[i])/2
                split.append(temp)

            D1 = 0
            D1_class_frequency = {j:0 for j in class_frequency.keys()}
            for index in range(len(split)):
                nk = njk[sort[index]]

                for j in nk:
                    D1_class_frequency[j] += nk[j]
                D1 += sum(nk.values())
                D2 = D - D1
                g_d1 = 1
                g_d2 = 1

                D2_class_frequency = {}
                for key, value in class_frequency.items():
                    if key in D1_class_frequency:
                        D2_class_frequency[key] = value - D1_class_frequency[key]
                    else:
                        D2_class_frequency[key] = value

                for key, v in D1_class_frequency.items():
                    g_d1 -= (v/float(D1))**2
                for key, v in D2_class_frequency.items():
                    g_d2 -= (v/float(D2))**2
                g = g_d1*D1/D + g_d2*D2/D
                if g < m1:
                    m1 = g
                    Xa_value = split[index]
                elif m1 < g < m2:
                    m2 = g
            return [m1, Xa_value]

        # discrete feature_values
        except TypeError:
            length = len(njk)
            feature_values = list(njk.keys())
            right = None
            if length > 10:  # too many discrete feature values
                for j, k in njk.items():
                    D1 = sum(k.values())
                    D2 = D - D1
                    g_d1 = 1
                    g_d2 = 1

                    D2_class_frequency = {}
                    for key, value in class_frequency.items():
                        if key in k:
                            D2_class_frequency[key] = value - k[key]
                        else:
                            D2_class_frequency[key] = value

                    for key, v in k.items():
                        g_d1 -= (v/D1)**2

                    if D2 != 0:
                        for key, v in D2_class_frequency.items():
                            g_d2 -= (v/D2)**2
                    g = g_d1*D1/D + g_d2*D2/D
                    if g < m1:
                        m1 = g
                        Xa_value = j
                    elif m1 < g < m2:
                        m2 = g
                right = list(np.setdiff1d(feature_values, Xa_value))

            else:  # fewer discrete feature values
                comb = self.select_combinations(feature_values)
                for i in comb:
                    left = list(i)
                    D1 = 0
                    D2 = 0
                    D1_class_frequency = {key:0 for key in class_frequency.keys()}
                    D2_class_frequency = {key:0 for key in class_frequency.keys()}
                    for j,k in njk.items():
                        for key, value in class_frequency.items():
                            if j in left:
                                if key in k:
                                    D1 += sum(k.values())
                                    D1_class_frequency[key] += k[key]
                            else:
                                if key in k:
                                    D2 += sum(k.values())
                                    D2_class_frequency[key] += k[key]
                    g_d1 = 1
                    g_d2 = 1
                    for key, v in D1_class_frequency.items():
                        g_d1 -= (v/D1)**2
                    for key, v in D2_class_frequency.items():
                        g_d2 -= (v/D1)**2
                    g = g_d1*D1/D + g_d2*D2/D
                    if g < m1:
                        m1 = g
                        Xa_value = left
                    elif m1 < g < m2:
                        m2 = g
                right = list(np.setdiff1d(feature_values, Xa_value))
            return [m1, [Xa_value, right]]

    def select_combinations(self, feature_values):
        combination = []
        e = len(feature_values)
        if e % 2 == 0:
            end = int(e/2)
            for i in range(1, end+1):
                if i == end:
                    cmb = list(combinations(feature_values, i))
                    enough = int(len(cmb)/2)
                    combination.extend(cmb[:enough])
                else:
                    combination.extend(combinations(feature_values, i))
        else:
            end = int((e-1)/2)
            for i in range(1, end+1):
                combination.extend(combinations(feature_values, i))

        return combination

# very fast decision tree class, i.e. hoeffding tree
class vfdt:
    # parameters
    # feature_values  # number of values of each feature # dict
    # feature_values = {feature:[unique values list]}
    # delta   # used for hoeffding bound
    # tau  # used to deal with ties
    # nmin  # used to limit the G computations

    def __init__(self, feature_values, delta=0.05, nmin=50, tau=0.1):
        self.feature_values = feature_values
        self.delta = delta
        self.nmin = nmin
        self.tau = tau
        features = list(feature_values.keys())
        self.root = vfdt_node(features)
        self.n_examples_processed = 0

    # update the tree by adding training example
    def update(self, example):
        self.n_examples_processed += 1
        node = self.root.sort_example(example)
        node.update_stats(example)

        result = node.splittable(self.delta, self.nmin, self.tau)
        if result != None:
            feature = result[0]
            value = result[1]
            #print('SPLIT')
            #print(feature, value)
            self.node_split(node, feature)
            node.split_value = value

    # split node, produce children
    def node_split(self, node, split_feature):
        features = node.possible_split_features
        # replace deleted split feature with None
        # new_features = [None if f == split_feature else f for f in features]
        left = vfdt_node(features)
        right = vfdt_node(features)
        node.add_children(split_feature, left, right)

    # predict test example's classification
    def predict(self, example):
        leaf = self.root.sort_example(example)
        prediction = leaf.most_frequent()
        return(prediction)

    # accuracy of a test set
    def accuracy(self, examples):
        correct = 0
        for ex in examples:
            if (self.predict(ex) == ex[-1]):
                correct +=1
        return(float(correct) / len(examples))

    def print_tree(self, node):
        if node.is_leaf():
            print('Leaf')
        else:
            print(node.split_feature)
            self.print_tree(node.left_child)
            self.print_tree(node.right_child)

def test_run():
    start_time = time.time()

    # bank.csv whole data size: 4521
    # if more than 4521, it revert back to 4521
    rows = 4521
    # n_training = int(0.8 * rows)
    # read_csv has parameter nrows=n that read the first n rows
    '''skiprows=1, index_col=0,'''
    df = pd.read_csv('bank.csv', nrows=rows, header=0, sep=';')
    title = list(df.columns.values)
    features = title[:-1]

    ''' change month string to int '''
    import calendar
    d = dict((v.lower(),k) for k,v in enumerate(calendar.month_abbr))
    df.month = df.month.map(d)
    #print(df.head(10))

    # unique values for each feature to use in VFDT
    feature_values = {f:None for f in features}
    for f in features:
        feature_values[f] = df[f].unique()

    # convert df to data examples
    array = df.head(4000).values
    set1 = []
    set2 = []
    set3 = []
    possible_split_features = title[:-1]
    count = 0
    for i in range(len(array)):
        count += 1
        if (count <= 1000):
            set1.append(array[i])
        elif (count > 1000 and count <= 2000):
            set2.append(array[i])
        else:
            set3.append(array[i])
    # to simulate continous training, modify the tree for each training set
    examples = [set1, set2, set3]

    # test set is different from training set
    n_test = 500
    test_set = df.tail(n_test).values
    test = []
    for i in range(len(test_set)):
        test.append(test_set[i])

    # heoffding bound parameter delta: with 1 - delta probability
    # the true mean is at least r - gamma
    # vfdt parameter nmin: test split if new sample size > nmin

    tree = vfdt(feature_values, delta=0.05, nmin=100, tau=0.05)
    print('Total data size: ', rows)
    print('Test set (tail): ', len(test_set))
    n = 0
    for training_set in examples:
        n += len(training_set)
        for ex in training_set:
            tree.update(ex)
        print('Training set:', n, end=', ')
        print('ACCURACY: %.4f' % tree.accuracy(test))

    #tree.print_tree(tree.root)
    print("--- Running time: %.6f seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    test_run()
