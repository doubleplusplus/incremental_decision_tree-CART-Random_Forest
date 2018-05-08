# vfdt class, vfdt_node class, Example class
#
# Jamie Deng
# 08/05/2018
# ver 0.01

import numpy as np
import pandas as pd
import math
import time


class Example:
    def __init__(self, line):
        self.x = line[:-1]  # attributes values
        self.y = line[-1]  # label


# VFDT node class
class vfdt_node:
    def __init__(self, possible_split_features):
        self.children = None
        self.possible_split_features = possible_split_features
        self.split_feature = None
        # self.n_feature_values = {f:0 for f in possible_split_features}
        self.new_examples_seen = 0
        self.total_examples_seen = 0
        self.class_frequency = {}
        self.nijk = {i:{} for i in possible_split_features}
        self.parent = None

    def __str__(self):
        if (not self.is_leaf()):
            print(len(self.children))

        print('split_feature: ', self.split_feature)
        print('class_frequency:', self.class_frequency)
        print('nijk: ', self.nijk)
        print('examples_seen', self.total_examples_seen)
        return('representation')

    def add_children(self, split_feature, nodes):
        if (not nodes):
            raise Exception('no children')
        self.split_feature = split_feature
        self.children = nodes
        self.nijk.clear()

    # return the leaf node corresponding to the test attribute
    def sort_example(self, example):
        value = 0
        if (self.children != None):
            index = self.possible_split_features.index(self.split_feature)
            value = example.x[index]
            return(self.children[value].sort_example(example))
        else:
            return(self)


    def is_leaf(self):
        return(self.children == None)

    def display_children(self):
        if (self.is_leaf()):
            print('It is leaf')
        else:
            print(list(self.children.keys()))
            for key in self.children:
                print(self.children[key].split_feature, end=' ')
            print()

    def most_frequent(self):
        if (not self.is_leaf()):
            raise Exception('Not a leaf')
        else:
            if (not self.class_frequency and self.parent != None):
                class_frequency = self.parent.class_frequency
                prediction = max(class_frequency, key=class_frequency.get)
            else:
                prediction = max(self.class_frequency, key=self.class_frequency.get)
            return(prediction)

    # need to discretize continous data
    def update_stats(self, example):
        label = example.y
        if(self.is_leaf() == False):
            raise Exception('This is not a leaf')
        feats = self.possible_split_features
        for i in feats:
            if (i != None):
                value = example.x[feats.index(i)]
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


    def splittable(self, delta, nmin):
        if(self.new_examples_seen < nmin):
            return(None)
        else:
            self.new_examples_seen = 0  # reset

        mx = 0
        second_mx = 0
        Xa = 0
        for feature in self.possible_split_features:
            if (feature != None):
                value = self.info_gain(feature)
                if (value > mx):
                    mx = value
                    Xa = feature
                elif (value > second_mx and value < mx):
                    second_mx = value
        R = math.log10(len(self.class_frequency))
        sigma = self.hoeffding_bound(R, delta, self.total_examples_seen)
        if (mx - second_mx > sigma):
            return(Xa)
        #elif (sigma < tau and mx - second_mx < tau):
            #return(Xa)
        else:
            return(None)


    def hoeffding_bound(self, R, delta, n):
        return(math.sqrt((R*R) * math.log(1/delta) / (2 * n)))

    def entropy(self, class_frequencies):
        total_examples = 0
        for k in class_frequencies:
            total_examples += class_frequencies[k]
        if (total_examples == 0):
            return(0)
        # print(total_examples)
        entropy = 0
        for k in class_frequencies:
            if(class_frequencies[k] != 0):
                entropy += -(class_frequencies[k] / float(total_examples)) * math.log2(class_frequencies[k] / float(total_examples))
            else:
                entropy += 0

        return(entropy)

    # nijk: attribute i, value j of class k
    def info_gain(self, featureID):
        njk = self.nijk[featureID]
        class_frequency = self.class_frequency

        total_examples = self.total_examples_seen
        if (total_examples == 0):
            return(0)
        entropy_before = self.entropy(class_frequency)

        # for each value j, class frequency must be counted
        entropy_after = 0
        for j in njk:
            count = 0
            for k in njk[j]:
                count += njk[j][k]
            entropy_after += (count/float(total_examples)) * (self.entropy(njk[j]))

        ig = entropy_before - entropy_after
        return ig

    def get_visualization(self, indent):
        if (self.children == None):
            return(indent + 'Leaf\n')
        else:
            visualization = ''
            for key in self.children:
                visualization += indent + self.split_feature + '=' + str(key) + ':\n'
                visualization += self.children[key].get_visualization(indent + '| ')
            return(visualization)

# very fast decision tree class, i.e. hoeffding tree
class vfdt:
    # parameters
    # feature_values  # number of values of each feature # dict
    # feature_values = {feature:[values_list]}
    # delta   # used for hoeffding bound
    # tau  # used to deal with ties
    # nmin  # used to limit the G computations

    def __init__(self, feature_values, delta, nmin):
        self.feature_values = feature_values
        self.delta = delta
        #self.tau = tau
        self.nmin = nmin

        features = list(feature_values.keys())
        self.root = vfdt_node(features)
        self.n_examples_processed = 0

    # update the tree by adding training example
    def update(self, example):
        self.n_examples_processed += 1
        node = self.root.sort_example(example)
        node.update_stats(example)

        split_feature = node.splittable(self.delta, self.nmin)
        if (split_feature != None):
            children =  self.node_split(node, split_feature)
            node.add_children(split_feature, children)
            for key in children:
                children[key].parent = node

    # try to split node, produce children
    def node_split(self, node, split_feature):
        possible_split_features = node.possible_split_features
        new_features = []
        for f in possible_split_features:
            if (f != split_feature):
                new_features.append(f)
            else:
                new_features.append(None)

        children = {}
        for v in self.feature_values[split_feature]:
            children[v] = vfdt_node(new_features)
        return(children)

    def predict(self, example):
        node = self.root.sort_example(example)
        prediction = node.most_frequent()
        return(prediction)

    def accuracy(self, examples):
        correct = 0
        for ex in examples:
            if (self.predict(ex) == ex.y):
                correct +=1
        return(float(correct) / len(examples))


def main():
    start_time = time.time()
    # read_csv has parameter nrows=n that read the first n rows
    df = pd.read_csv('bank.csv', nrows=500, header=0, sep=";")
    title = list(df.columns.values)
    #print(title)

    bins = 10
    labels = [i for i in range(1, bins+1)]

    feature_values = {} # unique values in a feature
    for c in df.columns:
        if (c != 'y'):
            # check if feature value is continous
            if (df[c].dtype == np.float64 or df[c].dtype == np.int64):
                df[c] = pd.cut(df[c], bins, labels=labels)
            feature_values[c] = df[c].unique()

    # covert df to data examples
    array = df.values
    examples = []
    possible_split_features = title[:-1]
    n = len(df.index)
    for i in range(n):
        ex = Example(array[i])
        examples.append(ex)

    delta = 0.01
    nmin = 10
    tree = vfdt(feature_values, delta, nmin)
    for ex in examples:
        tree.update(ex)

    # print(tree.root.get_visualization('$'))
    print('accuracy: %.4f' % tree.accuracy(examples))

    print("--- Running time: %.6f seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()
