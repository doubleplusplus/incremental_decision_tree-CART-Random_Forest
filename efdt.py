# Exetremely Fast Decision Tree i.e. Hoeffding Any Time Tree, described in
# "Extremely Fast Decision Tree" (Manapragada, Webb & Salehi, 2018)
#
# this program contains 2 classes: efdt, efdt_node
# test in command line window: python3 efdt.py
# changed to CART: Gini index
#
# Jamie
# 06/06/2018
# ver 0.01


import numpy as np
import pandas as pd
import time
from itertools import combinations


# EFDT node class
class efdt_node:
    # parameter nijk: statistics of feature i, value j, class k
    def __init__(self, possible_split_features):
        self.parent = None
        self.left_child = None
        self.right_child = None
        self.split_feature = None
        self.split_value = None
        self.split_g = None
        #self.discrete_split_values = {}
        self.new_examples_seen = 0
        self.total_examples_seen = 0
        self.class_frequency = {}
        self.nijk = {i:{} for i in possible_split_features}
        self.possible_split_features = possible_split_features

    def add_children(self, left, right, split_feature, split_value):
        self.left_child = left
        self.right_child = right
        left.parent = self
        right.parent = self

        if isinstance(split_value, list):
            left_value = split_value[0]
            right_value = split_value[1]
            if len(left_value) <= 1:
                new_features = [None if f == split_feature else f for f in left.possible_split_features]
                left.possible_split_features = new_features
            if len(right_value) <= 1:
                new_features = [None if f == split_feature else f for f in right.possible_split_features]
                right.possible_split_features = new_features


    def is_leaf(self):
        return(self.left_child == None and self.right_child == None)

    # upadate node stats in order to calculate Gini
    def update_stats(self, xij, label):
        feats = self.possible_split_features
        iterator = [f for f in feats if f is not None]
        for i in iterator:
            value = xij[feats.index(i)]
            if value not in self.nijk[i]:
                self.nijk[i][value] = {label : 1}
            else:
                try:
                    self.nijk[i][value][label] += 1
                except KeyError:
                    self.nijk[i][value][label] = 1

        self.total_examples_seen += 1
        self.new_examples_seen += 1
        try:
            self.class_frequency[label] += 1
        except KeyError:
            self.class_frequency[label] = 1

    # the most frequent classification
    def most_frequent(self):
        if (self.class_frequency):
            prediction = max(self.class_frequency, key=self.class_frequency.get)
        else:
            # if self.class_frequency dict is empty, go back to parent
            class_frequency = self.parent.class_frequency
            prediction = max(class_frequency, key=class_frequency.get)
        return(prediction)

    def check_not_splitting(self, class_frequency):
        # compute Gini index for not splitting
        g_d1 = 1
        g_d2 = 1
        #most = max(class_frequency, key=class_frequency.get)
        n = sum(class_frequency.values())
        for j, k in class_frequency.items():
            g_d1 -= (k/n)**2

        return g_d1

    def node_split(self, split_feature, split_value):
        self.split_feature = split_feature
        self.split_value = split_value
        features = self.possible_split_features

        left = efdt_node(features)
        right = efdt_node(features)
        self.add_children(left, right, split_feature, split_value)

    # recursively trace down the tree
    def sort_example(self, xij, label, delta, nmin, tau):
        if self.is_leaf():
            self.update_stats(xij, label)
            self.attempt_split(delta, nmin, tau)
            return
        else:
            left = self.left_child
            right = self.right_child

            index = self.possible_split_features.index(self.split_feature)
            value = xij[index]
            try:  # continous value
                if value <= self.split_value:
                    left.update_stats(xij, label)
                    left.re_evaluate_split(delta, nmin, tau)

                    return left.sort_example(xij, label, delta, nmin, tau)
                else:
                    right.update_stats(xij, label)
                    right.re_evaluate_split(delta, nmin, tau)

                    return right.sort_example(xij, label, delta, nmin, tau)
            except TypeError:  # discrete value
                if value in self.split_value[0]:
                    left.update_stats(xij, label)
                    left.re_evaluate_split(delta, nmin, tau)

                    return left.sort_example(xij, label, delta, nmin, tau)
                else:
                    right.update_stats(xij, label)
                    right.re_evaluate_split(delta, nmin, tau)

                    return right.sort_example(xij, label, delta, nmin, tau)

    def sort_to_predict(self, xij, label):
        if self.is_leaf():
            return self
        else:
            index = self.possible_split_features.index(self.split_feature)
            value = xij[index]

            try:  # continous value
                if value <= self.split_value:
                    return self.left_child.sort_to_predict(xij, label)
                else:
                    return self.right_child.sort_to_predict(xij, label)
            except TypeError:  # discrete value
                if value in self.split_value[0]:
                    return self.left_child.sort_to_predict(xij, label)
                else:
                    return self.right_child.sort_to_predict(xij, label)

    # test node split, return the split feature
    def attempt_split(self, delta, nmin, tau):
        if(self.new_examples_seen < nmin):
            return None
        class_frequency = self.class_frequency
        if len(class_frequency) <= 1:
            return None

        self.new_examples_seen = 0  # reset
        nijk = self.nijk
        g_Xa = 1  # minimun g

        Xa = ''
        split_value = None
        for feature in self.possible_split_features:
            if feature is not None:
                njk = nijk[feature]
                if len(njk) == 1:
                    gini = 0
                    Xa = feature
                    split_value = next(iter(njk))
                    if isinstance(split_value, str):
                        split_value = [[split_value],[]]
                else:
                    gini, value = self.Gini(feature, njk, class_frequency)
                    if gini < g_Xa:
                        g_Xa = gini
                        Xa = feature
                        split_value = value

        sigma = self.hoeffding_bound(delta)
        g_X0 = self.check_not_splitting(class_frequency)
        g_Xb = g_X0
        if g_Xb - g_Xa > sigma and g_Xa != g_X0:
            self.split_g = g_Xa  # split on feature Xa
            print('node split')
            self.node_split(Xa, split_value)

    def re_evaluate_split(self, delta, nmin, tau):
        if(self.new_examples_seen < nmin):
            return None
        class_frequency = self.class_frequency
        if len(class_frequency) <= 1:
            return None

        self.new_examples_seen = 0  # reset
        nijk = self.nijk
        g_Xa = 1

        Xa = ''
        split_value = None
        for feature in self.possible_split_features:
            if feature is not None:
                njk = nijk[feature]
                if len(njk) == 1:
                    gini = 0
                    Xa = feature
                    split_value = next(iter(njk))
                    if isinstance(split_value, str):
                        split_value = [[split_value],[]]
                else:
                    gini, value = self.Gini(feature, njk, class_frequency)
                    if (gini < g_Xa):
                        g_Xa = gini
                        Xa = feature
                        split_value = value

        sigma = self.hoeffding_bound(delta)
        g_X0 = self.check_not_splitting(class_frequency)

        split_g = self.split_g  # gini of current split feature
        if split_g - g_Xa > sigma:
            if g_X0 < g_Xa:  # not split
                print('kill subtree')
                self.kill_subtree()
            elif Xa != self.split_feature:
                print('split on new feature')
                self.split_g = g_Xa  # split on feature Xa
                self.node_split(Xa, split_value)

    def kill_subtree(self):
        if self.is_leaf():
            pass
        else:
            self.left_child = None
            self.right_child = None
            self.split_feature = None
            self.split_value = None
            self.split_g = None

    def hoeffding_bound(self, delta):
        n = self.total_examples_seen
        R = np.log(len(self.class_frequency))
        return (R * R * np.log(1/delta) / (2 * n))**0.5

    def Gini(self, feature, njk, class_frequency):
        '''
        Gini(D) = 1 - Sum(pi^2)
        Gini(D, F=f) = |D1|/|D|*Gini(D1) + |D2|/|D|*Gini(D2)
        '''
        D = self.total_examples_seen

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

# very fast decision tree class
class efdt:
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
        self.root = efdt_node(features)
        self.n_examples_processed = 0

    # find the path of example
    def find_path(self, leaf):
        path = []
        node = leaf
        while node:
            path.append(node)
            node = node.parent
        path.reverse()
        return path

    # update the tree by adding training example
    def update(self, example):
        label = example[-1]
        xij = example[:-1]
        self.n_examples_processed += 1
        self.root.sort_example(xij, label, self.delta, self.nmin, self.tau)

    # predict test example's classification
    def predict(self, example):
        label = example[-1]
        xij = example[:-1]
        leaf = self.root.sort_to_predict(xij, label)
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
    rows = 5000
    # n_training = int(0.8 * rows)
    # read_csv has parameter nrows=n that read the first n rows
    '''skiprows=1, index_col=0,'''
    df = pd.read_csv('bank.csv', nrows=rows, header=0, sep=';')
    #df = pd.read_csv('default_of_credit_card_clients.csv', nrows=rows, header=0)
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
    array = df.head(4500).values
    set1 = []
    set2 = []
    set3 = []
    possible_split_features = title[:-1]
    count = 0
    for i in range(len(array)):
        count += 1
        if (count <= 1000):
            set1.append(array[i])
        elif (count >1000 and count <= 3000):
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
    # efdt parameter nmin: test split if new sample size > nmin

    tree = efdt(feature_values, delta=0.01, nmin=500, tau=0.05)
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
