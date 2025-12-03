##############################################
# Programmer: Aiden Tabrah
# Class: CptS 322-01, Fall 2025
# Programming Assignment #7
# 11/21/25
# 
# Description: A variety of classes that can "classify" input data and predict which label they deserve.
##############################################

import numpy as np
from mysklearn import myutils
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor

class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, x_train, y_train):
        """Fits a simple linear regression line to x_train and y_train.

        Args:
            x_train(list of list of numeric vals): The list of training instances (samples).
                The shape of x_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to x_train)
                The shape of y_train is n_train_samples
        """
        
        self.regressor = MySimpleLinearRegressor(self.discretizer)
        self.regressor.fit(x_train, y_train)

    def predict(self, x_test):
        """Makes predictions for test samples in x_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            x_test(list of list of numeric vals): The list of testing samples
                The shape of x_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to x_test)
        """
        
        if (not isinstance(self.regressor, MySimpleLinearRegressor)):
            raise Exception(TypeError)
        y_hat = self.regressor.predict(x_test)
        
        y_predicted = []
        for y in y_hat:
            y_predicted.append(self.discretizer(y))
        
        return y_predicted

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, x_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            x_test(list of list of numeric vals): The list of testing samples
                The shape of x_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in x_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        
        all_distances = myutils.get_euclidean_distance(x_test, self.X_train)
        
        distances, neighbor_indices = myutils.get_nearest_neighbors(all_distances, self.n_neighbors)
        return distances, neighbor_indices

    def predict(self, x_test):
        """Makes predictions for test instances in x_test.

        Args:
            x_test(list of list of numeric vals): The list of testing samples
                The shape of x_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to x_test)
            
        Raises:
            TypeError when there is no y_train
        """
        
        if not isinstance(self.y_train, list):
            raise TypeError("y_train must be a list")
        
        distances, neighbor_indices = self.kneighbors(x_test) 

        predictions = []
        for neighbor_index_list in neighbor_indices:
            selected_neighbor_classifications = []
            for index in neighbor_index_list:
                selected_neighbor_classifications.append(self.y_train[index])
                
            
            category_frequency = myutils.get_frequency(selected_neighbor_classifications) 
            sorted_frequencies = sorted(category_frequency.items(), key=lambda item: item[1], reverse=True)
            predictions.append(sorted_frequencies[0][0])
        
        return predictions

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        
        frequencies = myutils.get_frequency(y_train)
        label, _ = max(frequencies.items(), key=lambda item: item[1])
        
        self.most_common_label = label

    def predict(self, x_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        
        y_predicted = []
        for x in x_test:
            y_predicted.append(self.most_common_label)
        
        return y_predicted

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        conditionals(YOU CHOOSE THE MOST APPROPRIATE TYPE): The conditional probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = {}
        self.conditionals = {}

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the conditional probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and conditionals.
        """
        
        num_samples = len(y_train)
        num_features = len(X_train[0])
        smoothing_alpha = 1
        
        # class_counts[c] = how many times class c appears in y_train
        class_counts = myutils.get_frequency(y_train)
        
        # priors[c] = P(Y = c)
        self.priors = {}
        # conditionals[c][j][v] = P(X_j = v | Y = c)
        self.conditionals = {}
        
        for class_label, class_count in class_counts.items():
            self.priors[class_label] = class_count / num_samples
            self.conditionals[class_label] = {}

            class_row_indices = [i for i, y in enumerate(y_train) if y == class_label]
        
            for feature_index in range(num_features):
                feature_values_for_class = [
                    X_train[i][feature_index] for i in class_row_indices
                ]
                
                value_counts = {}
                for feature_value in feature_values_for_class:
                    value_counts[feature_value] = value_counts.get(feature_value, 0) + 1
                    
                possible_feature_values = set(row[feature_index] for row in X_train)
                total_for_class_feature = len(feature_values_for_class)

                self.conditionals[class_label][feature_index] = {}
                for feature_value in possible_feature_values:
                    likelihood = (value_counts.get(feature_value, 0) + smoothing_alpha) / (total_for_class_feature + (len(possible_feature_values) * smoothing_alpha))
                    self.conditionals[class_label][feature_index][feature_value] = likelihood


    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []

        for x in X_test:
            best_class = None
            best_score = 0.0

            for class_label, prior in self.priors.items():
                score = prior

                for feature_index, feature_value in enumerate(x):
                    likelihoods_for_feature = self.conditionals[class_label][feature_index]

                    # if a test value never appeared in training, give it a tiny prob
                    likelihood = likelihoods_for_feature.get(feature_value, 1e-9)

                    score *= likelihood

                if score > best_score:
                    best_score = score
                    best_class = class_label

            y_predicted.append(best_class)

        return y_predicted

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        if not X_train:
            raise ValueError("X_train is empty; cannot fit a decision tree.")

        self.X_train = X_train
        self.y_train = y_train

        number_of_features = len(X_train[0])

        training_instances = [
            feature_row + [class_label]
            for feature_row, class_label in zip(X_train, y_train)
        ]

        available_attribute_indexes = list(range(number_of_features))

        attribute_domains = {}
        for attribute_index in available_attribute_indexes:
            values_for_attribute = {
                instance[attribute_index] for instance in training_instances
            }
            attribute_domains[attribute_index] = sorted(values_for_attribute)

        class_label_domain = sorted(set(y_train))
        
        self.tree = myutils.tdidt(
            training_instances,
            available_attribute_indexes,
            attribute_domains,
            class_label_domain,
            parent_instance_count=len(training_instances)
        )

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        if self.tree is None:
            raise ValueError("Decision tree has not been fit yet; call fit() before predict().")

        y_predicted = []
        for instance in X_test:
            predicted_label = myutils.classify_instance_with_tree(instance, self.tree)
            y_predicted.append(predicted_label)

        return y_predicted

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        if self.tree is None:
            raise ValueError("Decision tree has not been fit yet; call fit() before print_decision_rules().")

        # If no attribute names are provided, use "att0", "att1", ...
        if attribute_names is None:
            if self.X_train is None or len(self.X_train) == 0:
                raise ValueError("Cannot infer attribute names because training data is missing.")
            number_of_features = len(self.X_train[0])
            attribute_names = [f"att{index}" for index in range(number_of_features)]

        self._print_decision_rules_recursive(
            current_node=self.tree,
            condition_list=[],
            attribute_names=attribute_names,
            class_name=class_name
        )
    
    def _print_decision_rules_recursive(self, current_node, condition_list, attribute_names, class_name):
        """Recursively walks the tree and prints one IF-THEN rule per leaf.
        
        Parameters:
            current_node (list of obj): The current node in the decision tree.
            condition_list (list of str): The list of conditions leading to the current node.
            attribute_names (list of str): The list of attribute names.
            class_name (str): The name of the class attribute.
        """
        node_type = current_node[0]

        if node_type == "Leaf":
            class_label = current_node[1]

            if condition_list:
                conditions_string = " AND ".join(condition_list)
            else:
                # Tree is just a single leaf
                conditions_string = "True"

            print(f"IF {conditions_string} THEN {class_name} = {class_label}")
            return

        attribute_identifier = current_node[1]
        # Accept either an int index or a string like "att0"
        if isinstance(attribute_identifier, str) and attribute_identifier.startswith("att"):
            attribute_index = int(attribute_identifier[3:])
        else:
            attribute_index = attribute_identifier

        attribute_name = attribute_names[attribute_index]

        for value_branch in current_node[2:]:
            _, attribute_value, subtree = value_branch

            new_condition = f"{attribute_name} == {attribute_value}"
            updated_condition_list = condition_list + [new_condition]

            self._print_decision_rules_recursive(
                current_node=subtree,
                condition_list=updated_condition_list,
                attribute_names=attribute_names,
                class_name=class_name
            )

    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # TODO: (BONUS) fix this

from mysklearn.myevaluation import bootstrap_sample, accuracy_score
class MyRandomForestClassifier:
    """Represents a Random Forest classifier.

    Notes:
        Uses raw TDIDT trees (nested lists) as base learners.
        Bootstraps training data, randomly selects F attributes per split,
        and keeps the M most accurate trees out of N generated.
    """
    

    def __init__(self, N=20, M=7, F=2):
        """Initializer for Random Forest.

        Args:
            N (int): number of trees to generate
            M (int): number of best trees to keep
            F (int): number of random attributes per split
        """
        self.N = N
        self.M = M
        self.F = F
        self.trees = []
        self.attribute_names = None

    def fit(self, X, y):
        """Fits the Random Forest model.

        Steps:
        1. Build attribute names att0, att1, ...
        2. Repeat N times:
           a. Bootstrap sample the data
           b. Build a TDIDT tree using random attribute subsets
           c. Evaluate tree using its OOB samples
        3. Select top M trees
        """
        # Build attribute names
        num_features = len(X[0])
        self.attribute_names = ["att" + str(i) for i in range(num_features)]

        trees_with_scores = []

        for _ in range(self.N):
            # Bootstrap sample
            X_train, X_oob, y_train, y_oob = bootstrap_sample(X, y)

            # Combine training rows w/ labels
            train_instances = [xrow[:] + [y_val] for xrow, y_val in zip(X_train, y_train)]
            available_attributes = list(range(num_features)) # take a subset right now
            subset_attributes = myutils.compute_random_subset(available_attributes, self.F)

            # Build tree with random attribute subsets
            attribute_domains = [
                                     set(instance[j] for instance in train_instances)
                                     for j in range(num_features)
                                 ]
            
            tree = myutils.tdidt(instances= train_instances,
                                 available_attribute_indexes= subset_attributes,
                                 attribute_domains= attribute_domains,
                                 class_label_domain=set(y_train),
                                 parent_instance_count= len(X_train))

            # Evaluate using OOB set
            if len(X_oob) > 0:
                preds = self.predict(X_oob, tree)
                score = accuracy_score(y_oob, preds)
            else:
                score = 0.0  # no OOB samples, rare case

            trees_with_scores.append((tree, score))

        # Sort by descending accuracy
        trees_with_scores.sort(key=lambda t: t[1], reverse=True)

        # Keep top M
        self.trees = [t for t, _ in trees_with_scores[:self.M]]



    def predict(self, X_test, t = None):
        """Predicts labels for X_test using majority vote across M trees."""

        if t is None: # assume for all, majority voting
            predictions = []
            for x in X_test:
                votes = []
                for tree in self.trees:
                    pred = myutils.classify_instance_with_tree(x, tree) # do we need self.attribute_names?
                    votes.append(pred)
                predictions.append(myutils.vote(votes))
            return predictions
        
        else: # tree has value
            predictions = []
            for x in X_test:
                pred = myutils.classify_instance_with_tree(x, t)
                predictions.append(t)
            return predictions
