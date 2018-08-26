# Very Fast Decision Tree i.e. Hoeffding Tree, described in
# "Mining High-Speed Data Streams" (Domingos & Hulten, 2000)
#
# this program contains 2 classes: Vfdt, VfdtNode
# test in command line window: python3 Vfdt.py
# changed to CART: gini index
#
# Jamie
# 02/06/2018
# ver 0.03


import numpy as np
import pandas as pd
import time
from itertools import combinations
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# VFDT node class
class VfdtNode:
    # parameter nijk: statistics of feature i, value j, class k
    def __init__(self, possible_split_features):
        self.parent = None
        self.left_child = None
        self.right_child = None
        self.split_feature = None
        self.split_value = None  # both continuous and discrete value
        self.discrete_split_values = {}
        self.new_examples_seen = 0
        self.total_examples_seen = 0
        self.class_frequency = {}
        self.nijk = {i: {} for i in possible_split_features}
        self.possible_split_features = possible_split_features

    def add_children(self, split_feature, split_value, left, right):
        self.split_feature = split_feature
        self.split_value = split_value
        self.left_child = left
        self.right_child = right
        left.parent = self
        right.parent = self
        discrete = self.discrete_split_values

        left.discrete_split_values = discrete
        right.discrete_split_values = discrete
        self.nijk.clear()  # reset stats
        if isinstance(split_value, list):
            left_value = split_value[0]
            right_value = split_value[1]
            if len(left_value) <= 1:
                new_features = [None if f == split_feature else f for f in left.possible_split_features]
                left.possible_split_features = new_features
            if len(right_value) <= 1:
                new_features = [None if f == split_feature else f for f in right.possible_split_features]
                right.possible_split_features = new_features

            left.discrete_split_values[split_feature] = left_value
            right.discrete_split_values[split_feature] = right_value

    def is_leaf(self):
        return self.left_child is None and self.right_child is None

    # recursively trace down the tree to distribute data examples to corresponding leaves
    def sort_example(self, x):
        if self.is_leaf():
            return self
        else:
            index = self.possible_split_features.index(self.split_feature)
            value = x[index]

            try:  # continuous value
                if value <= self.split_value:
                    return self.left_child.sort_example(x)
                else:
                    return self.right_child.sort_example(x)
            except TypeError:  # discrete value
                if value in self.split_value[0]:
                    return self.left_child.sort_example(x)
                else:
                    return self.right_child.sort_example(x)

    # the most frequent class
    def most_frequent(self):
        try:
            prediction = max(self.class_frequency, key=self.class_frequency.get)
        except ValueError:
            # if self.class_frequency dict is empty, go back to parent
            class_frequency = self.parent.class_frequency
            prediction = max(class_frequency, key=class_frequency.get)
        return prediction

    # update leaf stats in order to calculate infomation gain
    def update_stats(self, x, y):
        feats = self.possible_split_features
        iterator = [f for f in feats if f is not None]
        for i in iterator:
            value = x[feats.index(i)]
            if value not in self.nijk[i]:
                self.nijk[i][value] = {y: 1}
            else:
                try:
                    self.nijk[i][value][y] += 1
                except KeyError:
                    self.nijk[i][value][y] = 1

        self.total_examples_seen += 1
        self.new_examples_seen += 1
        try:
            self.class_frequency[y] += 1
        except KeyError:
            self.class_frequency[y] = 1

    def check_not_splitting(self):
        # compute gini index for not splitting
        X0 = 1
        class_frequency = self.class_frequency
        n = sum(class_frequency.values())
        for j, k in class_frequency.items():
            X0 -= (k/n)**2
        return X0

    # use hoeffding tree model to test node split, return the split feature
    def attempt_split(self, delta, nmin, tau):
        if self.new_examples_seen < nmin:
            return None
        if len(self.class_frequency) == 1:
            return None

        self.new_examples_seen = 0  # reset

        nijk = self.nijk
        min = 1
        second_min = 1
        Xa = ''
        split_value = None
        for feature in self.possible_split_features:
            if feature is not None:
                njk = nijk[feature]
                if len(njk) == 1:
                    Xa = feature
                    split_value = next(iter(njk))
                    if isinstance(split_value, str):
                        # print(Xa, [[split_value],[]])
                        return [Xa, [[split_value],[]]]
                    else:
                        return [Xa, split_value]
                gini, value = self.gini(feature)

                if gini < min:
                    min = gini
                    Xa = feature
                    split_value = value
                elif min < gini < second_min:
                    second_min = gini

        sigma = self.hoeffding_bound(delta)
        g_X0 = self.check_not_splitting()
        if min < g_X0:
            if second_min - min > sigma:
                return [Xa, split_value]
            elif sigma < tau and second_min - min < tau:
                return [Xa, split_value]
            else:
                return None
        else:
            return None

    def hoeffding_bound(self, delta):
        n = self.total_examples_seen
        R = np.log(len(self.class_frequency))
        return np.sqrt(R * R * np.log(1/delta) / (2 * n))

    def gini(self, feature):
        # gini(D) = 1 - Sum(pi^2)
        # gini(D, F=f) = |D1|/|D|*gini(D1) + |D2|/|D|*gini(D2)

        njk = self.nijk[feature]
        D = self.total_examples_seen
        class_frequency = self.class_frequency

        m1 = 1  # minimum gini
        m2 = 1  # second minimum gini
        Xa_value = None
        test = next(iter(njk))  # test j feature value
        # if not isinstance(test, np.object):
        try:  # continuous feature values
            test += 0
            sort = np.array(sorted([j for j in njk.keys()]))
            split = (sort[0:-1] + sort[1:])/2   # vectorized computation, like in R

            D1 = 0
            D1_class_frequency = {j: 0 for j in class_frequency.keys()}
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
            if length > 10:  # too many discrete feature values, estimate
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

            else:  # fewer discrete feature values, get combinations
                comb = self.select_combinations(feature_values)
                for i in comb:
                    left = list(i)
                    D1_class_frequency = {key: 0 for key in class_frequency.keys()}
                    D2_class_frequency = {key: 0 for key in class_frequency.keys()}
                    for j,k in njk.items():
                        for key, value in class_frequency.items():
                            if j in left:
                                if key in k:
                                    D1_class_frequency[key] += k[key]
                            else:
                                if key in k:
                                    D2_class_frequency[key] += k[key]
                    g_d1 = 1
                    g_d2 = 1
                    D1 = sum(D1_class_frequency.values())
                    D2 = D - D1
                    for key, v in D1_class_frequency.items():
                        g_d1 -= (v/D1)**2
                    for key, v in D2_class_frequency.items():
                        g_d2 -= (v/D2)**2
                    g = g_d1*D1/D + g_d2*D2/D
                    if g < m1:
                        m1 = g
                        Xa_value = left
                    elif m1 < g < m2:
                        m2 = g
                right = list(np.setdiff1d(feature_values, Xa_value))
            return [m1, [Xa_value, right]]

    # divide values into two groups, return the combination of left groups
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
class Vfdt:
    # parameters
    # feature_values  # number of values of each feature # dict
    # feature_values = {feature:[unique values list]}
    # delta   # used for hoeffding bound
    # tau  # used to deal with ties
    # nmin  # used to limit the G computations

    def __init__(self, feature_values, delta=0.01, nmin=100, tau=0.1):
        self.feature_values = feature_values
        self.delta = delta
        self.nmin = nmin
        self.tau = tau
        features = list(feature_values.keys())
        self.root = VfdtNode(features)
        self.n_examples_processed = 0

    # update the tree by adding training example
    def update(self, x, y):
        self.n_examples_processed += 1
        node = self.root.sort_example(x)
        node.update_stats(x, y)

        result = node.attempt_split(self.delta, self.nmin, self.tau)
        if result is not None:
            feature = result[0]
            value = result[1]
            self.node_split(node, feature, value)

    # split node, produce children
    def node_split(self, node, split_feature, split_value):
        features = node.possible_split_features
        # print('node_split')
        left = VfdtNode(features)
        right = VfdtNode(features)
        node.add_children(split_feature, split_value, left, right)

    # predict test example's classification
    def predict(self, x_test):
        prediction = []
        if isinstance(x_test, np.ndarray):
            for x in x_test:
                leaf = self.root.sort_example(x)
                prediction.append(leaf.most_frequent())
            return prediction
        else:
            leaf = self.root.sort_example(x_test)
            return leaf.most_frequent()

    def print_tree(self, node):
        if node.is_leaf():
            print('Leaf')
        else:
            print(node.split_feature)
            self.print_tree(node.left_child)
            self.print_tree(node.right_child)


def calc_metrics(y_test, y_pred, row_name):
    accuracy = accuracy_score(y_test, y_pred)
    metrics = list(precision_recall_fscore_support(y_test, y_pred, average='weighted',
                                                   labels=np.unique(y_pred)))
    metrics = pd.DataFrame({'accuracy': accuracy, 'precision': metrics[0],'recall': metrics[1],
                            'f1': metrics[2]}, index=[row_name])
    return metrics


def test_run():
    start_time = time.time()
    # bank.csv whole data size: 4521
    # if more than 4521, it revert back to 4521
    # read_csv has parameter nrows=n that read the first n rows
    '''skiprows=1, index_col=0,'''
    df = pd.read_csv('./dataset/bank.csv', header=0, sep=';')
    # df = pd.read_csv('./dataset/default_of_credit_card_clients.csv', skiprows=1, header=0)
    # df = df.drop(df.columns[0], axis=1)
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle data rows
    title = list(df.columns.values)
    features = title[:-1]
    rows = df.shape[0]
    """# change month string to int
    import calendar
    d = dict((v.lower(),k) for k,v in enumerate(calendar.month_abbr))
    df.month = df.month.map(d)
    """
    # unique values for each feature to use in VFDT
    feature_values = {f: None for f in features}
    for f in features:
        feature_values[f] = df[f].unique()

    # convert df to data examples
    n_training = 4000
    array = df.head(n_training).values

    set1 = array[:1000, :]
    set2 = array[1000:3000, :]
    set3 = array[3000:, :]

    # to simulate continuous training, modify the tree for each training set
    examples = [set1, set2, set3]

    # test set is different from training set
    n_test = 500
    test_set = df.tail(n_test).values
    x_test = test_set[:, :-1]
    y_test = test_set[:, -1]

    # heoffding bound parameter delta: with 1 - delta probability
    # the true mean is at least r - gamma
    # Vfdt parameter nmin: test split if new sample size > nmin

    tree = Vfdt(feature_values, delta=0.01, nmin=300, tau=0.05)
    print('Total data size: ', rows)
    print('Training size size: ', n_training)
    print('Test set (tail): ', n_test)
    n = 0
    for training_set in examples:
        n += len(training_set)
        x_train = training_set[:, :-1]
        y_train = training_set[:, -1]
        for x, y in zip(x_train, y_train):
            tree.update(x, y)
        y_pred = tree.predict(x_test)
        print('Training set:', n, end=', ')
        print('ACCURACY: %.4f' % accuracy_score(y_test, y_pred))

    # tree.print_tree(tree.root)
    print("--- Running time: %.6f seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    test_run()
