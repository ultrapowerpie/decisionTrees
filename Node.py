import math, numpy as np

class Node:

    def __init__(self, key):
        self.key = key
        self.h = None
        self.leaves = None
        self.split_on = None
        self.left = None
        self.right = None
        self.leaf = None

    def predict(self, example):
        '''
        Args:
            example:    a 1d array of features

        Returns:        0 or 1, a prediction based on input features
        '''
        if self.leaf is not None:
            return self.leaf

        if example[self.split_on[0]] <= self.split_on[1]:
            return self.left.predict(example)
        else:
            return self.right.predict(example)

    def build(self, train, test):
        '''
        Args:
            train:  a 2d training matrix of (examples, features)
            test:   a 1d array of classes of the training examples

        Returns:    N/A, recursively builds the BDT
        '''
        self.h = self.entropy(np.mean(test))
        self.leaves = test.shape[0]
        self.leaf = self.isSame(test)
        if self.leaf is not None:
            return None
        self.split_on = self.chooseSplit(train, test)
        if not self.split_on:
            self.leaf = round(np.mean(test))
            return None
        lx,lt,rx,rt = self.split(self.split_on, train, test)
        self.left = Node(self.key+"->left")
        self.right = Node(self.key+"->right")
        self.left.build(lx, lt)
        self.right.build(rx, rt)

        # print self.key+str(self.split_on)+" "+str(self.h)

    def isSame(self, array):
        '''
        Args:
            array:  a 1d, binary array of example classes

        Returns:    true if all values in a 1d array are the same
        '''
        for i in enumerate(array):
            if i[1] != array[i[0]-1]:
                return None
        return array[0]

    def chooseSplit(self, train, test):
        '''
        Args:
            train:  a 2d array of [examples, features]
            test:   a 1d, binary array of example classes of length # of
                    examples

        Returns:    a (feature index, threshold) tuple of the highest
                    information gain feature to split on
        '''
        ig_max = (0,0,0)
        for i in range(train.shape[1]):
            # sort each (feature, class) tuple-vector
            features = np.asarray(sorted(zip(train[:,i], test),
                                        key=lambda tup: tup[0]))

            for j in range(1,features.shape[0]):
                if features[j-1, 0] == features[j, 0]:
                    continue

                x = float(j)/(features.shape[0])
                y_xt = np.mean(features[:j,1])
                y_xf = np.mean(features[j:,1])
                ig = self.h-self.conEntropy(x, y_xt, y_xf)
                if ig > ig_max[0]:
                    ig_max = (ig, i, features[j-1,0])

        return ig_max[1:]

    def split(self, split_on, train, test):
        '''
        Args:
            split_on:   the tuple (feature index, threshold)
            train:      2d array of (examples, feature vectors)
            test:       1d array of classes

        Returns:    a 4-tuple of split left and right train and test arrays
                    where the left arrays are less than threshold and right
                    arrays are greater than threshold
        '''
        leftTrain = []
        rightTrain = []
        leftTest = []
        rightTest = []
        for i in range(train.shape[0]):
            if train[i, split_on[0]] <= split_on[1]:
                leftTrain.append(train[i,:])
                leftTest.append(test[i])
            else:
                rightTrain.append(train[i,:])
                rightTest.append(test[i])

        return  np.asarray(leftTrain), np.asarray(leftTest), \
                np.asarray(rightTrain), np.asarray(rightTest)

    def entropy(self, x):
        '''
        Args:
            x:      proportion (probability) of true (1) of a bernoulli

        Returns:    entropy of the bernoulli distribution

        Calculates the entropy of a bernoulli x
        x = P(X = true)
        '''
        if x == 1.0 or x == 0.0:
            return 0.0
        q = 1.0 - x
        return -x*math.log(x,2) - q*math.log(q,2)

    def conEntropy(self, x, y_xt, y_xf):
        '''
        Args:
            x:      proportion (probability) of true (1) of a bernoulli
            y_xt:   proportion of true of a bernoulli given x is true
            x_xf:   proportion of true of a bernoulli given x is false

        Returns:    the conditional entropy of the bernoulli distribution

        Calculates the conditional entropy of a bernoulli y on a bernoulli x
        x = P(X = true)
        y_xt = P(Y = true | X = true)
        y_xf = P(Y = true | X = false)
        '''
        q = 1.0 - x
        return x*self.entropy(y_xt) + q*self.entropy(y_xf)
