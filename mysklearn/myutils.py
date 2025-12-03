##############################################
# Programmer: Aiden Tabrah
# Class: CptS 322-01, Fall 2025
# Programming Assignment #7
# 11/21/25
#
# Description: Utility functions to help create different graphs or do common actions
##############################################

from datetime import datetime
from typing import Any, List
import matplotlib.pyplot as plt
import numpy as np
from mysklearn import myevaluation

def convert_zeros_to_mean(col):
  non_zeros = [val for val in col if val != 0]
  non_zeros_mean = sum(non_zeros)/len(non_zeros)
  return [val if val != 0 else non_zeros_mean for val in col]

def high_low_discretizer(col, use_mean = True):
  '''discretizes column of continuous variables based on high or low from median or mean, specified by presence of outliers
    ARGS: col -- column of continuous data
    use_mean -- default is True. if true, uses mean to partition data. if false, uses median

    RETURNS: list of discretized values either "high" or "low"
  '''
  if use_mean:
    crit_val = np.mean(col)
  else: # use median
    crit_val = np.median(col)
  
  return ['high' if col[idx] >= crit_val else 'low' for idx in range(len(col))]



def norm_list(col):
  col_max, col_min = max(col), min(col)
  return [ (val - col_min)/(col_max - col_min) for val in col]

def partition_instances(instances, attribute, header):
    # this is group by attribute domain (not values of attribute in instances)
    # Returns a dictionary: {attribute_value: [instances]}
    att_index = header.index(attribute)
    att_domain = set([instance[att_index] for instance in instances])
    partitions = {}
    for att_value in att_domain: # "Junior" -> "Mid" -> "Senior"
        partitions[att_value] = []
        for instance in instances:
            if instance[att_index] == att_value:
                partitions[att_value].append(instance)

    return partitions

def calc_entropy(x_vals, y_col, idx):
    # if len(x_vals[0]) == 1: # 1 column base separator (stupid)
    #     x_col = x_vals
    # else:
    x_col = [x_val[idx] for x_val in x_vals]
    entropy = sum([(x_col.count(kx)/len(x_col))*
                    sum([sum([-1 for x, y in zip(x_col, y_col) if (x,y) == (kx, ky)])/x_col.count(kx)*np.log2(sum([1 for x, y in zip(x_col, y_col) if (x,y) == (kx, ky)])/x_col.count(kx))# entropy formula
                        if sum([1 for x, y in zip(x_col, y_col) if (x,y) == (kx, ky)]) != 0 else 0 # log0 case
                        for ky in set(y_col) ]) # iterate through ky cases = should result in sum of leading fractions to 1
                   for kx in set(x_col)]) # iterate through kx, should result in leading sum adding up to 1
    return entropy

def select_attribute(instances, attributes):
    # TODO: implement the general Enew algorithm for attribute selection
    # for each available attribute
    #     for each value in the attribute's domain
    #          calculate the entropy for the value's partition
    #     calculate the weighted average for the parition entropies
    # select that attribute with the smallest Enew entropy
    x_vals = [instance[:-1] for instance in instances]
    y_vals = [instance[-1] for instance in instances]
    entropy = {idx: calc_entropy(x_vals=x_vals, y_col =y_vals, idx= idx) for idx in range(len(attributes))}
    choice_idx = min(entropy, key = entropy.get)        
    
    return attributes[choice_idx]

def all_same_class(instances):
    # get the class label of the first instance.
    first_class = instances[0][-1]
    for instance in instances:
        # if any label differs, return False immediately.
        if instance[-1] != first_class:
            return False
        
    # if the loop completes without finding differences, return True.
    return True 

def compute_random_subset(values, num_values):
    # let's use np.random.shuffle()
    values_copy = values.copy()
    np.random.shuffle(values_copy) # inplace
    return values_copy[:num_values]

def vote(vals):
    '''
    args: vals, 1d list
    returns:
    majority vote, if tie, returns sorted first term alphabetically
    '''
    votes = {val: vals.count(val) for val in set(vals)}
    max_votes = max(votes.values())
    return min((k for k, v in votes.items() if v == max_votes))

def DOE_discretizer(mpg):
  """Outputs the label associeated with the given mpg

  Parameters:
    mpg (int): The mpg of the instance
  Returns:
    The associated label
  """
  labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

  if mpg >= 45:
    return labels[9]
  elif mpg >= 37:
    return labels[8]
  elif mpg >= 31:
    return labels[7]
  elif mpg >= 27:
    return labels[6]
  elif mpg >= 24:
    return labels[5]
  elif mpg >= 20:
    return labels[4]
  elif mpg >= 17:
    return labels[3]
  elif mpg >= 15:
    return labels[2]
  elif mpg == 14:
    return labels[1]
  else:
    return labels[0]


def DOE_bucket_counter(mpg, range_label=True):
  """Transforms data into a dictionary that buckets based off of Department of Energy metrics for MPG

  Parameters:
    mpg (list[int]): A 1D list of data to be bucketized

  Returns:
    A dictionary containing the MPG range as the key, and the amount of cars in the buckets
  """
  labels = ["≤ 13", "14", "15-16", "17-19", "20-23",
            "24-26", "27-30", "31-36", "37-44", "≥ 45"]

  if not range_label:
    labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

  mpg_dictionary = {labels[0]: 0, labels[1]: 0, labels[2]: 0, labels[3]: 0, labels[4]
    : 0, labels[5]: 0, labels[6]: 0, labels[7]: 0, labels[8]: 0, labels[9]: 0}

  for item in mpg:
    if item >= 45:
      mpg_dictionary[labels[9]] += 1
    elif item >= 37:
      mpg_dictionary[labels[8]] += 1
    elif item >= 31:
      mpg_dictionary[labels[7]] += 1
    elif item >= 27:
      mpg_dictionary[labels[6]] += 1
    elif item >= 24:
      mpg_dictionary[labels[5]] += 1
    elif item >= 20:
      mpg_dictionary[labels[4]] += 1
    elif item >= 17:
      mpg_dictionary[labels[3]] += 1
    elif item >= 15:
      mpg_dictionary[labels[2]] += 1
    elif item == 14:
      mpg_dictionary[labels[1]] += 1
    else:
      mpg_dictionary[labels[0]] += 1

  return mpg_dictionary


def bucket_equal_width(data, num_buckets):
  """Creates a dictionary of num_buckets amount of buckets that spans the whole data

  Parameters:
    data (list[int]): A list of numbers as the datapoints
    num_buckets (int): The number of buckets to categorize the data into

  Returns:
    A dictionary of the bucket ranges as keys, and the amount of data points in those buckets
  """
  bucket_range = set(data)

  min_val = min(bucket_range)
  max_val = max(bucket_range)
  bucket_width = (max_val - min_val) / num_buckets

  bucket_dictionary = {}
  for k in range(num_buckets):
    lower_bound = min_val + k * bucket_width
    upper_bound = min_val + (k + 1) * bucket_width
    key = f"{lower_bound}-{upper_bound}"
    bucket_dictionary[key] = 0

  for item in data:
    k = min(int((item - min_val) / bucket_width), num_buckets - 1)
    lower_bound = min(bucket_range) + k * bucket_width
    upper_bound = min(bucket_range) + (k + 1) * bucket_width
    key = f"{lower_bound}-{upper_bound}"
    bucket_dictionary[key] += 1

  return bucket_dictionary


def frequency_diagram(data, header, objects):
  """Creates a frequency diagram based off of the 1D list given.

  Parameters:
    data (dictionary): A dictionary of categories and their amount
    header (str): The name of the type of data being analyzed
    objects: The number of ____ <- what the instance actually is - PLURAL

  Notes:
    Shows the generated diagram
  """

  plt.bar(data.keys(), data.values())
  plt.xlabel(header.capitalize())
  plt.ylabel("Count")
  plt.title(
      f"Total Number of {objects.capitalize()} by {header.capitalize()}")
  plt.show()


def histogram_diagram(data, header):
  """Creates a histogram based off of the 1D list provided.

  Args:
    data (list[float]): A list of integer data to turn into a histogram
    header (str): The name of the type of data being analyzed

  Notes:
    Shows the generated diagram
  """

  plt.hist(data, edgecolor='black')
  plt.xlabel(header.capitalize())
  plt.ylabel("Count")
  plt.title(f"Distribution of {header.capitalize()} Values")
  plt.show()


def scatter_plot(x, y, x_label, y_label):
  """Creates a scatter plot based off of provided 1D lists.

  Parameters:
    x (list[float]): The independent variables in a 1D list.
    y (list[float]): The dependent variables in a 1D list
    x_label (str): A description for the data in the x 1D array.
    y_label (str): A description for the data in the y 1D array.

  Notes:
    Shows the generated diagram
  """

  plt.scatter(x, y)
  plt.xlabel(x_label.capitalize())
  plt.ylabel(y_label.capitalize())
  plt.title(f"{y_label.capitalize()} vs. {x_label.capitalize()}")
  plt.show()


def box_and_whisker_plot(data, x_labels, x_header, y_header):
  """Creates a box and whisker plot for each x_label

  Parameters:
    data (list[list[float]]): A 2D list, where each inner list contains the numerical data points for a single box on the plot.
    x_labels (list[str]): A list of labels for each corresponding box in data
    x_label (str): A description for the data in the x 1D array.
    y_label (str): A description for the data in the y 1D array.

  """

  plt.boxplot(data, tick_labels=x_labels)

  plt.xlabel(x_header.capitalize())
  plt.ylabel(y_header.capitalize())
  plt.title(
      f"Distribution of {y_header.capitalize()} by {x_header.capitalize()}")
  plt.grid(True, linestyle='--', alpha=0.7)
  plt.show()


def discretizer_high_low_100(x):
  """Categorizes an input object into either "high" or "low" centered around 100

  Parameters:
    x (int): An numerical value to categorize

  Returns:
    string: A category
  """

  if x >= 100:
    return "high"
  else:
    return "low"


def get_frequency(categorical_list):
  """Creates a dictionary of item to item count given a categorical 1D list.

  Args:
    categorical_list (list(obj)): A list of categorical labels that occur

  Returns:
    dictionary(obj): A dictionary of "category": count
  """
  data_dictionary = {}
  for item in categorical_list:
    data_dictionary[item] = data_dictionary.get(item, 0) + 1

  return data_dictionary


def get_euclidean_distance(test_list, training_list):
  """Gets the Euclidean Distance between two of the instances.

  Parameters:
    test_list (list of list of numeric values): A list of instances with various attributes already normalized
    training_list (list of list of numeric values): A list of instances to find distances with (already normalized)
  Returns:
    distance_list (list of list of floats): A list of all distances from one test instance to each training instance
  """

  X_test = np.asarray(test_list, dtype=float) 
  X_train = np.asarray(training_list, dtype=float) 

  diff = X_test[:, None, :] - X_train[None, :, :]

  dists = np.sqrt(np.sum(diff * diff, axis=2))

  return dists.tolist()


def get_nearest_neighbors(all_distance_list, k):
  """Given all distances, this function sorts through the distances and selects the nearest neighbors

  Parameters:
    all_distance_list (list of list of floats): A list of all distances from one instance to each other instance (including own instance)
    k (int): The number of neighbors to get
  Returns:
      all_neighbor_distances (list of list of floats): A list of the k-nearest distances from one instance to another
      all_neighbor_indices (list of list of ints): A list of the indices of the k-nearest neighbors
  Notes:
    all_neighbor_distances is parallel with all_neighbor_indices
  """

  all_neighbor_distances = []
  all_neighbor_indices = []
  for instance_distances in all_distance_list:
    index_value_dictionary = {
        index: value for index, value in enumerate(instance_distances)}
    sorted_distances = sorted(
        index_value_dictionary.items(), key=lambda item: item[1])

    neighbor_distances = []
    neighbor_indices = []
    for j in range(k):
      nearest_neighbor = sorted_distances[j]

      neighbor_indices.append(nearest_neighbor[0])
      neighbor_distances.append(nearest_neighbor[1])

    all_neighbor_indices.append(neighbor_indices)
    all_neighbor_distances.append(neighbor_distances)

  return all_neighbor_distances, all_neighbor_indices


def linear_regression_classifier_step_1(test_instances, classifier, auto_data, truncate_output=False):
  """Runs step 1 of the linear classifier test via using 'weight' to classify 'mpg'

  Parameters:
    test_instances (list of list of obj): The "data" that is being used to test
    classifier (MySimpleLinearClassifier): The classifier used to test
    auto_data (MyPyTable): The compelte data of the auto_data PyTable.
  """
  classifier.fit(
      [[w] for w in auto_data.get_column('weight')],
      auto_data.get_column('mpg')
  )

  print("===========================================")
  print("STEP 1: Linear Regression MPG Classifier")
  print("===========================================")

  weight_column_index = auto_data.column_names.index('weight')
  mpg_column_index = auto_data.column_names.index('mpg')
  accuracy = 0
  for instance in test_instances:
    predicted = classifier.predict([[instance[weight_column_index]]])[0]
    actual = DOE_discretizer(instance[mpg_column_index])

    if predicted == actual:
      accuracy += 1

    if not truncate_output:
      print(f"Instance: {instance}")
      print(f"Predicted:\t{predicted}")
      print(f"Actual:\t\t{actual}")

  accuracy = accuracy / len(test_instances)
  print(f"\nAccuracy:\t{accuracy}")


def k_neighbors_classifier_step_2(test_data_py_table, classifier, auto_data, truncate_output=False):
  """Runs step 2 of the k neighbors classifier test, using 'cylinders', 'weight', and 'accleration' to classify 'mpg'

  Parameters:
    test_data_py_table (MyPyTable): A table of all the test instances
    classifier (MyKNeighborsClassifier): The classifier used to get k neighbors
    auto_data (MyPyTable): The complete auto_data table.
  """
  neighbors_data = auto_data.select_columns(
      ['cylinders', 'weight', 'acceleration'])
  neighbors_data = neighbors_data.normalize_values(
      ['cylinders', 'weight', 'acceleration'])

  y_train = [DOE_discretizer(v) for v in auto_data.get_column('mpg')]
  classifier.fit(neighbors_data.data, y_train)

  print("===========================================")
  print("STEP 2: k=5 Nearest Neighbor MPG Classifier")
  print("===========================================")

  accuracy = 0
  for i, instance in enumerate(test_data_py_table.data):
    test_data_py_table = test_data_py_table.select_columns(
        ['cylinders', 'weight', 'acceleration', 'mpg'])
    test_data_py_table = test_data_py_table.normalize_values(
        ['cylinders', 'weight', 'acceleration'])

    x_data = [test_data_py_table.data[i][:3]]
    predicted = classifier.predict(x_data)[0]
    if predicted == y_train[i]:
      accuracy += 1

    if not truncate_output:
      print(f"Instance: {instance}")
      print(f"Predicted:\t{predicted}")
      print(f"Actual:\t\t{y_train[i]}")

  accuracy = accuracy / len(test_data_py_table.data)
  print(f"\nAccuracy:\t{accuracy}")


def dummy_classifier_step_3(test_data_py_table, classifier, auto_data, truncate_output=False):
  """Runs step 3 of the dummy classifier test to classify 'mpg'

  Parameters:
    test_data_py_table (MyPyTable): A table of all the test instances
    classifier (MyDummyClassifier): The classifier used to get k neighbors
    auto_data (MyPyTable): The complete auto_data table.
  """

  feature_columns = [c for c in auto_data.column_names if c != 'mpg']
  x_table = auto_data.select_columns(feature_columns)
  y_train = [DOE_discretizer(v) for v in auto_data.get_column('mpg')]
  classifier.fit(x_table.data, y_train)

  print("===========================================")
  print("STEP 3: (Zero-R) Dummy MPG Classifier")
  print("===========================================")

  x_test_table = test_data_py_table.select_columns(feature_columns)
  y_pred = classifier.predict(x_test_table.data)

  y_actual = [DOE_discretizer(v)
              for v in test_data_py_table.get_column('mpg')]

  accuracy = 0
  for i, instance in enumerate(test_data_py_table.data):
    if y_pred[i] == y_actual[i]:
      accuracy += 1

    if not truncate_output:
      print(f"Instance: {instance}")
      print(f"Predicted:\t{y_pred[i]}")
      print(f"Actual:\t\t{y_actual[i]}")

  accuracy = accuracy / len(test_data_py_table.data)
  print(f"\nAccuracy:\t{accuracy}")


def accuracy_error(y_true, y_pred):
  """Calculates the accuracy and error of the predictions.

  Args:
    y_true (list of any): The true values.
    y_pred (list of any): The predicted values.

  Returns:
    tuple of float: The accuracy and error.
  """
  correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
  acc = correct / len(y_true) if y_true else 0.0
  return acc, 1.0 - acc


def select_indexes(list, indexes):
  """Slices a list based on a list of indices.

  Parameters:
      list (list of any): The list to slice.
      indexes (list of int): The indices to slice by.

  Returns:
    list of any: The sliced list.
  """
  return [list[i] for i in indexes]


def random_subsample(classifier, X, y, k=10, test_size=0.33, random_state=None, shuffle=True):
  """Run k repeats of train_test_split -> fit -> predict -> accuracy/error.

  Parameters:
    classifier (MyKNeighborsClassifier or MyDummyClassifier): The classifier to train.
    X (list of num): The attribute data.
    y (list of num): The labeled data.
    k (int): The number of repeats to run.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): The seed for the random number generator.
    shuffle (bool): Whether to shuffle the data before splitting.

  Returns:
    mean_accuracy (float): The mean accuracy across all runs.
    mean_error (float): The mean error across all runs.
    run_accuracies (list of float): The accuracy for each run.
    run_errors (list of float): The error for each run.
  """
  run_accuracies = []
  run_errors = []
  for _ in range(k):
    X_train, X_test, y_train, y_test = myevaluation.train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=shuffle
    )
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy, error = accuracy_error(y_test, y_pred)
    run_accuracies.append(accuracy)
    run_errors.append(error)

  mean_accuracy = sum(run_accuracies) / \
      len(run_accuracies) if run_accuracies else 0.0
  mean_error = sum(run_errors) / len(run_errors) if run_errors else 0.0

  return mean_accuracy, mean_error, run_accuracies, run_errors


def cross_val_predict(classifier, X, y, num_splits=10, random_state=None, shuffle=False):
  """K-fold split -> fit -> predict -> accuracy/error.

  Parameters:
    classifier (MyKNeighborsClassifier or MyDummyClassifier): The classifier to train.
    X (list of num): The attribute data.
    y (list of num): The labeled data.
    num_splits (int): The number of folds to use for cross-validation.
    random_state (int): The seed for the random number generator.
    shuffle (bool): Whether to shuffle the data before splitting.

  Returns:
    mean_accuracy (float): The mean accuracy across all folds.
    mean_error (float): The mean error error all folds.
    fold_predicted (list of float): The predicted values for each fold.
    fold_true (list of float): The true values for each fold.
  """
  folds = myevaluation.kfold_split(X, n_splits=num_splits, random_state=random_state, shuffle=shuffle)

  fold_accuracies = []
  fold_errors = []
  fold_predicted = []
  fold_true = []
  for train_indexes, test_indexes in folds:
    X_train, y_train = select_indexes(X, train_indexes), select_indexes(y, train_indexes)
    X_test, y_test = select_indexes(X, test_indexes), select_indexes(y, test_indexes)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    accuracy, error = accuracy_error(y_test, y_pred)

    fold_accuracies.append(accuracy)
    fold_errors.append(error)
    fold_predicted.extend(y_pred)
    fold_true.extend(y_test)

  mean_accuracy = sum(fold_accuracies) / \
      len(fold_accuracies) if fold_accuracies else 0.0
  mean_error = sum(fold_errors) / len(fold_errors) if fold_errors else 0.0

  return mean_accuracy, mean_error, fold_predicted, fold_true


def bootstrap_method(classifier, X, y, k=10, n_samples=None, random_state=None):
  """Bootstrap + Out of the Bag sampling -> fit -> predict -> accuracy/error.
  Skips runs with empty OOB.

  Parameters:
    classifier (MyKNeighborsClassifier or MyDummyClassifier): The classifier to train.
    X (list of num): The attribute data.
    y (list of num): The labeled data.
    k (int): The number of repeated splits to train with.
    n_samples (int): The number of samples to use for each bootstrap sample.
    random_state (int): The seed for the random number generator.
  Returns:
    mean_accuracy (float): The mean accuracy across all runs.
    mean_error (float): The mean error across all runs.
    run_accuracies (list of float): The accuracy for each run.
    run_errors (list of float): The error for each run.
  """

  run_accuracies = []
  run_errors = []
  for _ in range(k):
    X_samples, X_oob, y_samples, y_oob = myevaluation.bootstrap_sample(
        X, y, n_samples=n_samples, random_state=random_state)

    # If no OOB, skip this split (mirrors sklearn’s NaN OOB behavior)
    if y_oob is None or len(X_oob) == 0:
      continue

    classifier.fit(X_samples, y_samples)
    y_pred = classifier.predict(X_oob)

    accuracy, error = accuracy_error(y_oob, y_pred)
    run_accuracies.append(accuracy)
    run_errors.append(error)

  mean_acc = (sum(run_accuracies) / len(run_accuracies)
              ) if run_accuracies else 0.0
  mean_error = (sum(run_errors) / len(run_errors)) if run_errors else 0.0

  return mean_acc, mean_error, run_accuracies, run_errors


def confusion_matrix_to_table(confusion_matrix, labels):
  """Converts a confusion matrix to a table with recognition rates.

  Parameters:
    confusion_matrix (list of list of int): The confusion matrix.
    labels (list of str): The labels for each class.

  Returns:
    rows (list of list of str): The table rows.
  """
  rows = []
  for i, label in enumerate(labels):
    row = confusion_matrix[i]
    total = sum(row)
    correct = row[i] if i < len(row) else 0
    recognition_rate = 100.0 * (correct / total) if total > 0 else 0.0

    rows.append([label, *row, total, f"{recognition_rate:.0f}"])
  return rows


def compute_entropy_from_frequencies(frequency_dictionary):
  """Compute entropy given a dictionary of value -> count.

  Parameters:
    frequency_dictionary (dict): The frequency dictionary.

  Returns:
    entropy (float): The entropy.
  """
  total_count = sum(frequency_dictionary.values())
  if total_count == 0:
    return 0.0

  entropy = 0.0
  for count in frequency_dictionary.values():
    if count == 0:
      continue
    probability = count / total_count
    entropy -= probability * np.log2(probability)
  return entropy


def compute_majority_class_label(class_labels, class_label_domain):
  """Return the majority class label, breaking ties by class_label_domain ordering.

  Parameters:
    class_labels (list of str): The class labels.
    class_label_domain (list of str): The domain of class labels.

  Returns:
    label (str): The majority class label.
  """
  class_counts = get_frequency(class_labels)
  highest_count = max(class_counts.values())

  candidate_labels = [
    label
    for label, count in class_counts.items()
    if count == highest_count
  ]

  # Break ties using domain ordering
  for label in class_label_domain:
    if label in candidate_labels:
      return label

  # Fallback
  return candidate_labels[0]


def compute_information_gain_for_attribute(instances, attribute_index, attribute_domains, class_label_domain):
  """Compute information gain of splitting on attribute_index.

  Parameters:
    instances (list of list of obj): The dataset.
    attribute_index (int): The index of the attribute to split on.
    attribute_domains (list of list of obj): The domain of each attribute.
    class_label_domain (list of str): The domain of class labels.

  Returns:
    information_gain (float): The information gain.
  """
  class_labels = [instance[-1] for instance in instances]
  parent_frequencies = get_frequency(class_labels)
  parent_entropy = compute_entropy_from_frequencies(parent_frequencies)

  total_instance_count = len(instances)
  if total_instance_count == 0:
    return 0.0

  weighted_child_entropy = 0.0
  for attribute_value in attribute_domains[attribute_index]:
    subset_instances = [
      instance
      for instance in instances
      if instance[attribute_index] == attribute_value
    ]
    if not subset_instances:
      continue

    subset_labels = [instance[-1] for instance in subset_instances]
    subset_frequencies = get_frequency(subset_labels)
    subset_entropy = compute_entropy_from_frequencies(subset_frequencies)

    weight = len(subset_instances) / total_instance_count
    weighted_child_entropy += weight * subset_entropy

  information_gain = parent_entropy - weighted_child_entropy
  return information_gain


def select_best_attribute_index(instances, available_attribute_indexes, attribute_domains, class_label_domain):
  """Select the attribute index with the highest information gain.

  Parameters:
    instances (list of list of obj): The dataset.
    available_attribute_indexes (list of int): The indices of the attributes that can be used for splitting.
    attribute_domains (list of list of obj): The domain of each attribute.
    class_label_domain (list of str): The domain of class labels.

  Returns:
    best_attribute_index (int): The index of the attribute with the highest information gain.
  """
  best_attribute_index = None
  best_information_gain = -1.0

  for attribute_index in available_attribute_indexes:
    information_gain = compute_information_gain_for_attribute(
      instances,
      attribute_index,
      attribute_domains,
      class_label_domain
    )

    if information_gain > best_information_gain:
      best_information_gain = information_gain
      best_attribute_index = attribute_index

  return best_attribute_index


def tdidt(instances, available_attribute_indexes, attribute_domains, class_label_domain, parent_instance_count):
  """Top-Down Induction of Decision Trees (TDIDT).

  Tree representation:
      Internal node: ["Attribute", "att0", ["Value", value1, subtree1], ...]
      Leaf node:     ["Leaf", class_label, majority_count_at_leaf, parent_instance_count]

  Parameters:
    instances (list of list of obj): The dataset.
    available_attribute_indexes (list of int): The indices of the attributes that can be used for splitting.
    attribute_domains (list of list of obj): The domain of each attribute.
    class_label_domain (list of str): The domain of class labels.

  Returns:
    tree (list of obj): The decision tree.
  """
  class_labels = [instance[-1] for instance in instances]

  # All instances have the same class
  if len(set(class_labels)) == 1:
    majority_label = class_labels[0]
    majority_count = len(class_labels)
    return ["Leaf", majority_label, majority_count, parent_instance_count]

  # No attributes left -> majority vote leaf
  if not available_attribute_indexes:
    majority_label = compute_majority_class_label(
        class_labels, class_label_domain)
    class_frequencies = get_frequency(class_labels)
    majority_count = class_frequencies[majority_label]
    return ["Leaf", majority_label, majority_count, parent_instance_count]

  best_attribute_index = select_best_attribute_index(
      instances,
      available_attribute_indexes,
      attribute_domains,
      class_label_domain
  )

  # Use "att#" *string* label to match the unit tests
  best_attribute_label = f"att{best_attribute_index}"
  decision_node = ["Attribute", best_attribute_label]

  decision_node: List[Any] = ["Attribute", best_attribute_label]

  # Only use values that actually appear in this partition
  attribute_values_in_partition: List[Any] = sorted({
      instance[best_attribute_index] for instance in instances
  })
  # For each value of the chosen attribute, grow a branch
  for attribute_value in attribute_values_in_partition:
    subset_instances = [
        instance
        for instance in instances
        if instance[best_attribute_index] == attribute_value
    ]

    new_available_attribute_indexes = [
        index
        for index in available_attribute_indexes
        if index != best_attribute_index
    ]
    subtree = tdidt(
        subset_instances,
        new_available_attribute_indexes,
        attribute_domains,
        class_label_domain,
        parent_instance_count=len(instances)
    )
    decision_node.append(["Value", attribute_value, subtree])

  return decision_node


def get_first_leaf_label(decision_tree_node):
  """Return the class label of the first leaf reachable from this node.

  Parameters:
    decision_tree_node (list of obj): The decision tree node.

  Returns:
    class_label (obj): The class label of the first leaf reachable from this node.
  """
  node_type = decision_tree_node[0]

  if node_type == "Leaf":
    return decision_tree_node[1]

  for value_branch in decision_tree_node[2:]:
    _, _, subtree = value_branch
    leaf_label = get_first_leaf_label(subtree)
    if leaf_label is not None:
      return leaf_label

  return None


def classify_instance_with_tree(instance, decision_tree):
  """Classify a single instance using the learned decision tree.

  Parameters:
    instance (list of obj): The instance to classify.
    decision_tree (list of obj): The learned decision tree.

  Returns:
    class_label (obj): The predicted class label for the instance.
  """
  current_node = decision_tree

  while current_node[0] == "Attribute":
    attribute_identifier = current_node[1]

    if isinstance(attribute_identifier, str) and attribute_identifier.startswith("att"):
      attribute_index = int(attribute_identifier[3:])
    else:
      attribute_index = attribute_identifier

    instance_value = instance[attribute_index]

    matching_subtree = None
    for value_branch in current_node[2:]:
      _, branch_value, subtree = value_branch
      if branch_value == instance_value:
        matching_subtree = subtree
        break

    if matching_subtree is None:
      # Unseen attribute value at prediction time, fall back to the first leaf under this node.
      return get_first_leaf_label(current_node)

    current_node = matching_subtree

  return current_node[1]

def convert_date_to_numeric(dates):
  """Converts date strings to numeric values.

  Args:
    dates (list of str): The list of date strings.
  
  Returns:
    list of float or str: The list of date strings converted to numeric values.
  """
  timestamp_list = []

  for date in dates:
    if isinstance(date, str) and date.strip() not in ("", "NA"):
      dt = datetime.strptime(date, "%Y-%m-%d %H:%M:%S.%f %z")
      timestamp_list.append(dt.timestamp())
    else:
      timestamp_list.append("NA")

  return timestamp_list


# def normalized_discretizer(x):
#   '''Returns discretized value for normalized data
#   args: x value

#   returns: discretized value
  
#   '''
#   return 