#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: decision_trees.py
@time: 2019-11-26 15:36
"""
from dataclasses import dataclass
import numpy as np
from numpy import ndarray


@dataclass
class Node:
    val: float = None  # 节点的输出值
    left: object = None  # : Node
    right: object = None  # : Node
    feature: int = None  # 选择哪个特征分裂
    split: float = None  # 选择feature的分裂位置
    loss_val: float = None  # loss function的值
    depth: int = None  # 节点所在的层数
    name: str = None  # 节点名称，用于调试和可视化

    def copy(self, node):
        for attr_name in list(Node.__dict__['__dataclass_fields__'].keys()):
            attr = getattr(node, attr_name)
            setattr(self, attr_name, attr)


class _DecisionTreeBase:
    """CART"""
    def __init__(self):
        self.root = Node()
        self.depth = 1
        self._rules = None

    def __str__(self):
        ret = []
        for i, rule in enumerate(self._rules):
            literals, avg = rule

            ret.append("Rule %d: " % i + ' | '.join(
                literals) + ' => y_hat %.4f' % avg)
        return "\n".join(ret)

    @staticmethod
    def _get_split_mse(col: ndarray, label: ndarray, split: float) -> Node:
        """
        Calculate the mse of label when col is splitted into two pieces.
        MSE as Loss fuction:
        y_hat = Sum(y_i) / n, i <- [1, n]
        Loss(y_hat, y) = Sum((y_hat - y_i) ^ 2), i <- [1, n]
        --------------------------------------------------------------------

        Arguments:
            col {ndarray} -- A feature of training data.
            label {ndarray} -- Target values.
            split {float} -- Split point of column.

        Returns:
            Node -- MSE of label and average of splitted x
        """

        # Split label.
        label_left = label[col < split]
        label_right = label[col >= split]

        # Calculate the means of label.
        avg_left = label_left.mean()
        avg_right = label_right.mean()

        # Calculate the mse of label.
        mse = (((label_left - avg_left) ** 2).sum() +
               ((label_right - avg_right) ** 2).sum()) / len(label)

        # Create nodes to store result.
        node = Node(split=split, loss_val=mse)
        node.left = Node(avg_left)
        node.right = Node(avg_right)
        return node

    def _choose_split(self, col: ndarray, label: ndarray) -> Node:
        """Iterate each xi and split x, y into two pieces,
        and the best split point is the xi when we get minimum mse.

        Arguments:
            col {ndarray} -- A feature of training data.
            label {ndarray} -- Target values.

        Returns:
            Node -- The best choice of mse, split point and average.
        """

        # Feature cannot be splitted if there's only one unique element.
        node = Node()
        unique = set(col)
        if len(unique) == 1:
            return node

        # In case of empty split.
        unique.remove(min(unique))

        # Get split point which has min mse.
        ite = map(lambda x: self._get_split_mse(col, label, x), unique)
        node = min(ite, key=lambda x: x.loss_val)

        return node

    def _choose_feature(self, data: ndarray, label: ndarray) -> Node:
        """
        计算feature split point
        对回归树用平方误差(mse, mean square error)最小化准则;
        对分类树用Gini index最小化准则
        Choose the feature which has minimum mse.

        Arguments:
            data {ndarray} -- Training data.
            label {ndarray} -- Target values.

        Returns:
            Node -- feature number, split point, average.
        """

        # Compare the mse of each feature and choose best one.
        _ite = map(lambda x: (self._choose_split(data[:, x], label), x),
                   range(data.shape[1]))
        ite = filter(lambda x: x[0].split is not None, _ite)

        # Return None if no feature can be splitted.
        node, feature = min(
            ite, key=lambda x: x[0].loss_val, default=(Node(), None))
        node.feature = feature

        return node

    def _build_tree(self, que, max_depth, min_samples_split, label):
        # Breadth-First Search.
        while que:
            depth, node, _data, _label = que.pop(0)

            # Terminate loop if tree depth is more than max_depth.
            if depth > max_depth:
                depth -= 1
                break

            # Stop split when number of node samples is less than
            # min_samples_split or Node is 100% pure.
            if len(_label) < min_samples_split or all(_label == label[0]):
                continue

            # Stop split if no feature has more than 2 unique elements.
            _node = self._choose_feature(_data, _label)
            if _node.split is None:
                continue

            # Copy the attributes of _node to node.
            node.copy(_node)
            # Put children of current node in que.
            idx_left = (_data[:, node.feature] < node.split)
            idx_right = (_data[:, node.feature] >= node.split)
            que.append(
                (depth + 1, node.left, _data[idx_left], _label[idx_left]))
            que.append(
                (depth + 1, node.right, _data[idx_right], _label[idx_right]))
        return depth

    def fit(self, data: ndarray, label: ndarray, max_depth=5, min_samples_split=2):
        """Build a regression decision tree.
        Note:
            At least there's one column in data has more than 2 unique elements,
            and label cannot be all the same value.

        Arguments:
            data {ndarray} -- Training data.
            label {ndarray} -- Target values.

        Keyword Arguments:
            max_depth {int} -- The maximum depth of the tree. (default: {5})
            min_samples_split {int} -- The minimum number of samples required
            to split an internal node. (default: {2})
        """

        # Initialize with depth, node, indexes.
        self.root.val = label.mean()
        que = [(self.depth + 1, self.root, data, label)]
        # Update tree depth and rules.
        self.depth = self._build_tree(que, max_depth, min_samples_split, label)

    def predict_one(self, row: ndarray) -> float:
        """Auxiliary function of predict.

        Arguments:
            row {ndarray} -- A sample of testing data.

        Returns:
            float -- Prediction of label.
        """

        node = self.root
        while node.left and node.right:
            if row[node.feature] < node.split:
                node = node.left
            else:
                node = node.right

        return node.val

    def predict(self, data: ndarray) -> ndarray:
        """Get the prediction of label.

        Arguments:
            data {ndarray} -- Testing data.

        Returns:
            ndarray -- Prediction of label.
        """

        return np.apply_along_axis(self.predict_one, 1, data)

    def print_tree(self, tree=None, indent=" "):
        """ Recursively print the decision tree """
        if not tree:
            tree = self.root

        # If we're at leaf => print the label
        if tree.val is not None:
            print(tree.val)
        # Go deeper down the tree
        else:
            # Print test
            print("%s:%s? " % (tree.feature, tree.split))
            # Print the true scenario
            print("%sT->" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            # Print the false scenario
            print("%sF->" % (indent), end="")
            self.print_tree(tree.right, indent + indent)


class RegressionTree(_DecisionTreeBase):
    """后续要把_DecisionTreeBase作为分类树和回归树的基类"""
    def __init__(self):
        super().__init__()
