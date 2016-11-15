COS 402 Binary Decision tree

Jackey Liu
netid: guanghao

BinaryDecisionTree.py

This is the actual decision tree class that has methods for loading
data and building the tree. It also keeps track of the root of its
tree.

This file also contains scripts to build 20 trees on 5% increments of
the TrainX.csv and TestX.csv datasets, and takes and optional 4th
argument TestY.csv to check its accuracy on the test data.

Node.py

This is the node class that actually makes up the tree. Each node has
attributes that allow it to classify and predict a new sample.


This decision tree achieves ~93% accuracy on test data and 100%
accuracy on training data.
