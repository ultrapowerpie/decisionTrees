from Node import Node
import sys, csv, math, tqdm, numpy as np
import matplotlib.pyplot as plt

class BDT:
    def __init__(self):
        self.root = None

    def build(self, train, test):
        '''
        Args:
            train:  a 2d training matrix of (examples, features)
            test:   a 1d array of classes of the training examples

        Returns:    N/A, builds the BDT
        '''
        if not train.size or not test.size:
            raise ValueError("Data matrices must not be empty")
        self.root = Node("root")
        self.root.build(train, test)

    def predict(self, examples):
        '''
        Args:
            example:    a 1d array of features,
                        or a 2d array of (examples, features)

        Returns:        a numpy array of 0s and 1s corresponding to inputs
        '''
        predictions = []
        if len(examples.shape) == 1:
            predictions.append([self.root.predict(examples)])
        else:
            for i in range(examples.shape[0]):
                predictions.append(self.root.predict(examples[i,:]))

        return np.asarray(predictions)

    def loadArray(self, filename):
        '''
        Args:
            filename:   a string filename of the csv file to load data matrix

        Returns:    a numpy array of input data
        '''
        array = []
        with open(filename, 'rb') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if len(row) > 1:
                    array.append(row)
                else:
                    array.extend(row)
        return np.asarray(array).astype(float)

    def dumpArray(self, filename, array):
        '''
        Args:
            filename:   a string filename of the csv file to dump data matrix
            array:      a 1d array of predictions to write to the csv file

        Returns:    N/A, writes array to csv file
        '''
        with open(filename, 'wb') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows([[int(i)] for i in array])

    def levelTraversal(self, filename=None, level=None):
        '''
        Args:
            filename:   the name of the file to print the level-order
                        traversal to
            level:      the maximum node depth to print

        Returns:    outputs the level-order traversal of the tree as a
                    string to the specified file
        '''
        queue = [(self.root, 0)]
        l = 0
        nodes = 0
        leaves = 0
        outStr = ""
        while queue:
            nodes += 1
            node, l = queue.pop(0)
            outStr += "Node: "+node.key+"\n"
            outStr += "Entropy: "+str(node.h)+"\n"
            outStr += "Leaves: "+str(node.leaves)+"\n"
            if node.split_on:
                outStr += "Feature: "+str(node.split_on[0])+"\n"
                outStr += "Threshold: "+str(node.split_on[1])+"\n"
            else:
                outStr += "Leaf: "+str(node.leaf)+"\n"
                leaves += 1
            outStr += "\n"
            if level and l >= level:
                continue
            if node.left:
                queue.append((node.left, l+1))
            if node.right:
                queue.append((node.right, l+1))
        outStr = "Total leaves up to level "+str(level)+": "+str(leaves) \
                 +"\n\n"+outStr
        outStr = "Total nodes up to level "+str(level)+": "+str(nodes) \
                 +"\n\n"+outStr
        if not filename:
            print outStr
        else:
            with open(filename, 'wb') as f:
                f.write(outStr)

tree = BDT()

trainX = tree.loadArray(sys.argv[1])
trainY = tree.loadArray(sys.argv[2])
testX = tree.loadArray(sys.argv[3])
if len(sys.argv) > 4:
    testY = tree.loadArray(sys.argv[4])

xaxis = np.arange(5, 105, 5)
training = []
testing = []

for percentage in tqdm.tqdm(xaxis):

    limit = trainX.shape[0]*percentage/100

    tree.build(trainX[:limit,:], trainY[:limit])

    training.append(np.mean(np.square(np.subtract(
                    tree.predict(trainX), trainY))))

    if len(sys.argv) > 4:
        testing.append(np.mean(np.square(np.subtract(
                        tree.predict(testX), testY))))

tree.dumpArray("PredictY.csv", tree.predict(testX))
tree.levelTraversal("level_order.txt", 1000)
plt.plot(xaxis, training, 'b', xaxis, testing, 'r')
plt.xlabel('Sample set size (percent of total samples used)')
plt.ylabel('Total loss (error)')
plt.show()
