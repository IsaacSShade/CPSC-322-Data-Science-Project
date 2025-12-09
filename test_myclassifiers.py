# pylint: skip-file

##############################################
# Programmer: Aiden Tabrah and Valon Haslem
# Class: CptS 322-01, Fall 2025
# Programming Assignment: Project
# 12/7/25
# 
# Description: A variety of unit tests to ensure various sklearn accuracy
##############################################

import numpy as np
from mysklearn.myclassifiers import MyNaiveBayesClassifier
from scipy import stats
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
from mysklearn.myclassifiers import MySimpleLinearRegressionClassifier, MyKNeighborsClassifier, MyDummyClassifier
from mysklearn.myutils import (compute_random_subset, tdidt, classify_instance_with_tree, vote, discretizer_high_low_100)
from mysklearn.myevaluation import bootstrap_sample
import numpy as np
from mysklearn.myclassifiers import MyDecisionTreeClassifier, MyRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline

# note: order is actual/received student value, expected/solution
def test_simple_linear_regression_classifier_fit():    
    test_slope = 2
    test_intercept = 0
    
    x_train = [[val] for val in range(0, 100)]
    y_train = [test_slope * x[0] + np.random.normal(test_intercept, 25) for x in x_train]

    classifier = MySimpleLinearRegressionClassifier(discretizer=discretizer_high_low_100)
    classifier.fit(x_train, y_train)
    
    assert isinstance(classifier.regressor, MySimpleLinearRegressor)
    
    test_linearRegressor = MySimpleLinearRegressor()
    test_linearRegressor.fit(x_train, y_train)
    
    assert (classifier.regressor.slope == test_linearRegressor.slope)
    assert (classifier.regressor.intercept == test_linearRegressor.intercept)

def test_simple_linear_regression_classifier_predict():
    # Test 1
    test_slope = 2
    test_intercept = 0
    
    x_train = [[val] for val in range(0, 100)]
    y_train = [test_slope * x[0] + np.random.normal(test_intercept, 25) for x in x_train]

    classifier = MySimpleLinearRegressionClassifier(discretizer=discretizer_high_low_100)
    classifier.fit(x_train, y_train)
    
    predict_x_list = [[1], [5], [10], [200]]
    predicted_y_list = classifier.predict(predict_x_list)
    
    expected = []
    for x in predict_x_list:
        y_hat = test_slope * x[0] + test_intercept
        expected.append("high" if y_hat >= 100 else "low")

    assert predicted_y_list == expected
    
    # Test 2
    test_slope = 5
    test_intercept = -45
    
    x_train = [[val] for val in range(0, 100)]
    y_train = [test_slope * x[0] + np.random.normal(test_intercept, 25) for x in x_train]

    classifier = MySimpleLinearRegressionClassifier(discretizer=discretizer_high_low_100)
    classifier.fit(x_train, y_train)
    
    predict_x_list = [[1], [5], [10], [200]]
    predicted_y_list = classifier.predict(predict_x_list)
    
    expected = []
    for x in predict_x_list:
        y_hat = test_slope * x[0] + test_intercept
        expected.append("high" if y_hat >= 100 else "low")

    assert predicted_y_list == expected
        

def test_kneighbors_classifier_kneighbors():
    #example #1  (4 instances)
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]
    
    classifier1 = MyKNeighborsClassifier(1)
    classifier2 = MyKNeighborsClassifier(2)
    classifier3 = MyKNeighborsClassifier(3)
    
    for clf in (classifier1, classifier2, classifier3):
        clf.fit(X_train_class_example1, y_train_class_example1)

    # k = 1
    distances1, indices1 = classifier1.kneighbors([[1, 1], [0, 0]])
    assert np.allclose(distances1, [[0.0], [0.0]])
    assert indices1 == [[0], [3]]

    # k = 2
    distances2, indices2 = classifier2.kneighbors([[1, 1], [0, 0]])
    assert np.allclose(distances2, [[0.0, 1.0], [0.0, 0.33]])
    assert indices2 == [[0, 1], [3, 2]]

    # k = 3
    distances3, indices3 = classifier3.kneighbors([[1, 1], [0, 0]])
    assert np.allclose(distances3, [[0.0, 1.0, 1.203702621082],
                                    [0.0, 0.33, 1.0]])
    assert indices3 == [[0, 1, 2], [3, 2, 1]]
    
    # example #2 (8 instances)
    # assume normalized
    X_train_class_example2 = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]]
    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    
    for clf in (classifier1, classifier2, classifier3):
        clf.fit(X_train_class_example2, y_train_class_example2)

    # k = 1
    distances1, indices1 = classifier1.kneighbors([[6, 6], [0, 0]])
    assert np.allclose(distances1, [[0.0], [2.0]])
    assert indices1 == [[1], [5]]

    # k = 2
    distances2, indices2 = classifier2.kneighbors([[6, 6], [0, 0]])
    assert np.allclose(distances2, [[0.0, 2.828427124746],
                                    [2.0, 2.2360679775]])
    assert indices2 == [[1, 3], [5, 4]]

    # k = 3
    distances3, indices3 = classifier3.kneighbors([[6, 6], [0, 0]])
    assert np.allclose(distances3, [[0.0, 2.828427124746, 5.0],
                                    [2.0, 2.2360679775, 3.0]])
    assert indices3 == [[1, 3, 0], [5, 4, 6]]

    # from Bramer
    header_bramer_example = ["Attribute 1", "Attribute 2"]
    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
        "-", "-", "+", "+", "+", "-", "+"]
    
    for clf in (classifier1, classifier2, classifier3):
        clf.fit(X_train_bramer_example, y_train_bramer_example)

    # k = 1
    distances1, indices1 = classifier1.kneighbors([[15, 15], [0, 0]])
    assert np.allclose(distances1, [[2.4738633753705948], [6.350590523722971]])
    assert indices1 == [[13], [0]]

    # k = 2
    distances2, indices2 = classifier2.kneighbors([[15, 15], [0, 0]])
    assert np.allclose(distances2, [[2.4738633753705948, 2.912043955712208],
                                    [6.350590523722971, 7.692203845452876]])
    assert indices2 == [[13, 14], [0, 2]]

    # k = 3
    distances3, indices3 = classifier3.kneighbors([[15, 15], [0, 0]])
    assert np.allclose(distances3, [[2.4738633753705948, 2.912043955712208, 3.5902646142032486],
                                    [6.350590523722971, 7.692203845452876, 8.22009732302483]])
    assert indices3 == [[13, 14, 12], [0, 2, 1]]

def test_kneighbors_classifier_predict():
    # example #1  (4 instances)
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]
    
    classifier = MyKNeighborsClassifier(3)
    classifier.fit(X_train_class_example1, y_train_class_example1)
    
    y_prediction = classifier.predict([[1.0, 1.0], [0.5, 0.5], [0, 0]])
    assert y_prediction == ["bad", "bad", "good"]

    # example #2 (8 instances)
    # assume normalized
    X_train_class_example2 = [
            [3, 2],
            [6, 6],
            [4, 1],
            [4, 4],
            [1, 2],
            [2, 0],
            [0, 3],
            [1, 6]]

    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    
    classifier.fit(X_train_class_example2, y_train_class_example2)
    
    y_prediction = classifier.predict([[6.0, 6.0], [3.0, 3.0], [0, 0]])
    assert y_prediction == ["no", "no", "yes"]

    # from Bramer
    header_bramer_example = ["Attribute 1", "Attribute 2"]
    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
            "-", "-", "+", "+", "+", "-", "+"]
    
    classifier.fit(X_train_bramer_example, y_train_bramer_example)
    
    y_prediction = classifier.predict([[15.0, 15.0], [10.0, 10.0], [5.0, 5.0]])
    assert y_prediction == ["-", "+", "-"]

def test_dummy_classifier_fit():
    # Setup
    x_train = [[val] for val in range(0, 100)]
    classifier = MyDummyClassifier()
    
    # Test 1
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    classifier.fit(x_train, y_train)
    
    assert(classifier.most_common_label == "yes")
    
    # Test 2
    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    classifier.fit(x_train, y_train)
    
    assert(classifier.most_common_label == "no")
    
    # Test 3
    y_train = list(np.random.choice(["yes", "no", "maybe", "unknown", "what"], 100, replace=True, p=[0.2, 0.1, 0.2, 0.1, 0.4]))
    classifier.fit(x_train, y_train)
    
    assert(classifier.most_common_label == "what")

def test_dummy_classifier_predict():
    # Setup
    x_train = [[val] for val in range(0, 100)]
    predict_x_list = [[1], [5], [10], [200]]
    classifier = MyDummyClassifier()
    
    # Test 1
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    classifier.fit(x_train, y_train)
    predicted_y_list = classifier.predict(predict_x_list)
    expected_y = ["yes" for _ in predict_x_list]
    
    assert(expected_y == predicted_y_list)
    
    # Test 2
    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    classifier.fit(x_train, y_train)
    predicted_y_list = classifier.predict(predict_x_list)
    expected_y = ["no" for _ in predict_x_list]
    
    assert(expected_y == predicted_y_list)
    
    # Test 3
    y_train = list(np.random.choice(["yes", "no", "maybe", "unknown", "what"], 100, replace=True, p=[0.2, 0.1, 0.2, 0.1, 0.4]))
    classifier.fit(x_train, y_train)
    predicted_y_list = classifier.predict(predict_x_list)
    expected_y = ["what" for _ in predict_x_list]
    
    assert(expected_y == predicted_y_list)

def test_naive_bayes_classifier_fit():
    # in-class Naive Bayes example (lab task #1)
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]

    # LA7 (fake) iPhone purchases dataset
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

    # Bramer 3.2 train dataset
    X_Brahmer_train = [
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"],
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "none", "slight"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"]
    ]
    Y_Brahmer_train = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                    "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                    "very late", "on time", "on time", "on time", "on time", "on time"]

    # In-class 8-instance example
    classifier = MyNaiveBayesClassifier()
    classifier.fit(X_train_inclass_example, y_train_inclass_example)
    expected_priors_inclass = {
        "yes": 5 / 8,
        "no": 3 / 8,
    }
    assert classifier.priors == expected_priors_inclass

    expected_conditionals_inclass = {
        "yes": {
            0: {1: 5 / 7, 2: 2 / 7},
            1: {5: 3 / 7, 6: 4 / 7},
        },
        "no": {
            0: {1: 3 / 5, 2: 2 / 5},
            1: {5: 3 / 5, 6: 2 / 5},
        },
    }
    assert classifier.conditionals == expected_conditionals_inclass

    # LA7 "iPhone purchases" fake data
    
    classifier.fit(X_train_iphone, y_train_iphone)
    expected_priors_iphone = {
        "no": 1 / 3,   # 5/15
        "yes": 2 / 3,  # 10/15
    }
    assert classifier.priors == expected_priors_iphone

    expected_conditionals_iphone = {
        "no": {
            0: {  # standing
                1: 4 / 7,
                2: 3 / 7,
            },
            1: {  # job_status
                1: 1 / 4,
                2: 3 / 8,
                3: 3 / 8,
            },
            2: {  # credit_rating
                "fair": 3 / 7,
                "excellent": 4 / 7,
            },
        },
        "yes": {
            0: {  # standing
                1: 1 / 4,
                2: 3 / 4,
            },
            1: {  # job_status
                1: 4 / 13,
                2: 5 / 13,
                3: 4 / 13,
            },
            2: {  # credit_rating
                "fair": 2 / 3,
                "excellent": 1 / 3,
            },
        },
    }
    assert classifier.conditionals == expected_conditionals_iphone

    # Bramer 3.2 train dataset
    classifier.fit(X_Brahmer_train, Y_Brahmer_train)

    expected_conditionals_bramer = {
        "on time": {
            0: {  # day
                "holiday": 1 / 6,
                "saturday": 1 / 6,
                "sunday": 1 / 9,
                "weekday": 5 / 9,
            },
            1: {  # season
                "autumn": 1 / 6,
                "spring": 5 / 18,
                "summer": 7 / 18,
                "winter": 1 / 6,
            },
            2: {  # wind
                "high": 5 / 17,
                "none": 6 / 17,
                "normal": 6 / 17,
            },
            3: {  # rain
                "heavy": 2 / 17,
                "none": 6 / 17,
                "slight": 9 / 17,
            },
        },
        "late": {
            0: {  # day
                "holiday": 1 / 6,
                "saturday": 1 / 3,
                "sunday": 1 / 6,
                "weekday": 1 / 3,
            },
            1: {  # season
                "autumn": 1 / 6,
                "spring": 1 / 6,
                "summer": 1 / 6,
                "winter": 1 / 2,
            },
            2: {  # wind
                "high": 2 / 5,
                "none": 1 / 5,
                "normal": 2 / 5,
            },
            3: {  # rain
                "heavy": 2 / 5,
                "none": 2 / 5,
                "slight": 1 / 5,
            },
        },
        "very late": {
            0: {  # day
                "holiday": 1 / 7,
                "saturday": 1 / 7,
                "sunday": 1 / 7,
                "weekday": 4 / 7,
            },
            1: {  # season
                "autumn": 2 / 7,
                "spring": 1 / 7,
                "summer": 1 / 7,
                "winter": 3 / 7,
            },
            2: {  # wind
                "high": 1 / 3,
                "none": 1 / 6,
                "normal": 1 / 2,
            },
            3: {  # rain
                "heavy": 1 / 2,
                "none": 1 / 3,
                "slight": 1 / 6,
            },
        },
        "cancelled": {
            0: {  # day
                "holiday": 1 / 5,
                "saturday": 2 / 5,
                "sunday": 1 / 5,
                "weekday": 1 / 5,
            },
            1: {  # season
                "autumn": 1 / 5,
                "spring": 2 / 5,
                "summer": 1 / 5,
                "winter": 1 / 5,
            },
            2: {  # wind
                "high": 1 / 2,
                "none": 1 / 4,
                "normal": 1 / 4,
            },
            3: {  # rain
                "heavy": 1 / 2,
                "none": 1 / 4,
                "slight": 1 / 4,
            },
        },
    }

    assert classifier.conditionals == expected_conditionals_bramer

def test_naive_bayes_classifier_predict():
    # in-class Naive Bayes example (lab task #1)
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]

    # LA7 (fake) iPhone purchases dataset
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

    # Bramer 3.2 train dataset
    X_Brahmer_train = [
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"],
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "none", "slight"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"]
    ]
    Y_Brahmer_train = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                    "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                    "very late", "on time", "on time", "on time", "on time", "on time"]

    # In-class 8-instance example
    classifier = MyNaiveBayesClassifier()
    classifier.fit(X_train_inclass_example, y_train_inclass_example)
    
    y_pred_inclass = classifier.predict(X_train_inclass_example)
    assert y_pred_inclass == ["yes"] * 8
    
    # LA7 iPhone purchases data
    classifier.fit(X_train_iphone, y_train_iphone)
    expected_y_pred_iphone = [
        "yes",  
        "no",   
        "yes",  
        "yes",  
        "yes", 
        "yes",
        "yes",  
        "yes", 
        "yes", 
        "yes", 
        "no",  
        "yes", 
        "yes",  
        "yes",  
        "yes", 
    ]
    y_pred_iphone = classifier.predict(X_train_iphone)
    assert y_pred_iphone == expected_y_pred_iphone
    
    # Brahmer 3.2 train dataset (Figure 3.1)
    classifier.fit(X_Brahmer_train, Y_Brahmer_train)
    expected_y_pred_bramer = [
        "on time",
        "on time",
        "on time",
        "very late",
        "on time",
        "on time",
        "on time",
        "on time",
        "very late",
        "on time",
        "cancelled",
        "on time",
        "late",
        "on time",
        "very late",
        "on time",
        "on time",
        "on time",
        "on time",
        "on time",
    ]
    y_pred_bramer = classifier.predict(X_Brahmer_train)
    assert y_pred_bramer == expected_y_pred_bramer

# interview dataset
header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
X_train_interview = [
    ["Senior", "Java", "no", "no"],
    ["Senior", "Java", "no", "yes"],
    ["Mid", "Python", "no", "no"],
    ["Junior", "Python", "no", "no"],
    ["Junior", "R", "yes", "no"],
    ["Junior", "R", "yes", "yes"],
    ["Mid", "R", "yes", "yes"],
    ["Senior", "Python", "no", "no"],
    ["Senior", "R", "yes", "no"],
    ["Junior", "Python", "yes", "no"],
    ["Senior", "Python", "yes", "yes"],
    ["Mid", "Python", "no", "yes"],
    ["Mid", "Java", "yes", "no"],
    ["Junior", "Python", "no", "yes"]
]
y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False",
                     "True", "True", "True", "True", "True", "False"]

# note: this tree uses generic "att#" labels and includes counts on leaves
#       ["Leaf", class_label, majority_count_in_leaf, parent_node_instance_count]
# note: attribute values are sorted alphabetically
tree_interview = \
    ["Attribute", "att0",
        ["Value", "Junior",
            ["Attribute", "att3",
                ["Value", "no",
                    ["Leaf", "True", 3, 5]
                ],
                ["Value", "yes",
                    ["Leaf", "False", 2, 5]
                ]
            ]
        ],
        ["Value", "Mid",
            ["Leaf", "True", 4, 14]
        ],
        ["Value", "Senior",
            ["Attribute", "att2",
                ["Value", "no",
                    ["Leaf", "False", 3, 5]
                ],
                ["Value", "yes",
                    ["Leaf", "True", 2, 5]
                ]
            ]
        ]
    ]

# LA7 iPhone purchases dataset for decision tree tests
header_iphone_dt = ["standing", "job_status", "credit_rating"]
X_train_iphone_dt = [
    [1, 3, "fair"],
    [1, 3, "excellent"],
    [2, 3, "fair"],
    [2, 2, "fair"],
    [2, 1, "fair"],
    [2, 1, "excellent"],
    [2, 1, "excellent"],
    [1, 2, "fair"],
    [1, 1, "fair"],
    [2, 2, "fair"],
    [1, 2, "excellent"],
    [2, 2, "excellent"],
    [2, 3, "fair"],
    [2, 2, "excellent"],
    [2, 3, "fair"]
]
y_train_iphone_dt = ["no", "no", "yes", "yes", "yes",
                     "no", "yes", "no", "yes", "yes",
                     "yes", "yes", "yes", "no", "yes"]

# Expected decision tree for the iPhone dataset
tree_iphone_expected = \
    ["Attribute", "att0",
        ["Value", 1,
            ["Attribute", "att1",
                ["Value", 1,
                    ["Leaf", "yes", 1, 5]
                ],
                ["Value", 2,
                    ["Attribute", "att2",
                        ["Value", "excellent",
                            ["Leaf", "yes", 1, 2]
                        ],
                        ["Value", "fair",
                            ["Leaf", "no", 1, 2]
                        ]
                    ]
                ],
                ["Value", 3,
                    ["Leaf", "no", 2, 5]
                ]
            ]
        ],
        ["Value", 2,
            ["Attribute", "att2",
                ["Value", "excellent",
                    ["Attribute", "att1",
                        ["Value", 1,
                            ["Leaf", "no", 1, 4]
                        ],
                        ["Value", 2,
                            ["Leaf", "no", 1, 4]
                        ]
                    ]
                ],
                ["Value", "fair",
                    ["Leaf", "yes", 6, 10]
                ]
            ]
        ]
    ]


def test_decision_tree_classifier_fit():
    # interview dataset
    interview_classifier = MyDecisionTreeClassifier()
    interview_classifier.fit(X_train_interview, y_train_interview)

    assert interview_classifier.tree == tree_interview

    # iPhone dataset
    iphone_classifier = MyDecisionTreeClassifier()
    iphone_classifier.fit(X_train_iphone_dt, y_train_iphone_dt)

    assert iphone_classifier.tree == tree_iphone_expected

def test_decision_tree_classifier_predict():
    # --- interview dataset predictions ---
    interview_classifier = MyDecisionTreeClassifier()
    interview_classifier.fit(X_train_interview, y_train_interview)

    X_test_interview = [
        ["Junior", "Java", "yes", "no"],
        ["Junior", "Java", "yes", "yes"]
    ]
    y_expected_interview = ["True", "False"]
    y_predicted_interview = interview_classifier.predict(X_test_interview)
    assert y_predicted_interview == y_expected_interview

    # --- iPhone dataset predictions ---
    iphone_classifier = MyDecisionTreeClassifier()
    iphone_classifier.fit(X_train_iphone_dt, y_train_iphone_dt)

    # Two unseen instances
    X_test_iphone = [
        [1, 1, "excellent"],
        [2, 3, "excellent"]
    ]
    
    y_expected_iphone = ["yes", "no"]
    y_predicted_iphone = iphone_classifier.predict(X_test_iphone)
    assert y_predicted_iphone == y_expected_iphone


def test_random_forest_classifier_fit():
    """Random forest fit should be deterministic and match replayed TDIDT construction on the interview dataset."""
    # Hyperparameters chosen for a small but non-trivial forest
    N = 5
    M = 3
    F = 2
    seed = 0

    # --- Train the actual forest ---
    rf = MyRandomForestClassifier(N=N, M=M, F=F, random_state=seed)
    rf.fit(X_train_interview, y_train_interview)
    
    rng = np.random.RandomState(seed)
    num_features = len(X_train_interview[0])
    trees_with_scores = []

    for _ in range(N):
        # Expected seed generation pattern
        seed_boot = int(rng.randint(0, 2**31 - 1))
        X_train_bag, X_oob, y_train_bag, y_oob = bootstrap_sample(
            X_train_interview,
            y_train_interview,
            random_state=seed_boot
        )

        # Combine training rows w/ labels
        train_instances = [
            xrow[:] + [y_val]
            for xrow, y_val in zip(X_train_bag, y_train_bag)
        ]

        # Random attribute subset for this tree (same pattern)
        available_attributes = list(range(num_features))
        seed_subset = int(rng.randint(0, 2**31 - 1))
        subset_attributes = compute_random_subset(
            available_attributes,
            F,
            random_state=seed_subset
        )

        # Deterministic attribute domains and class label domain
        attribute_domains = []
        for j in range(num_features):
            values = {inst[j] for inst in train_instances}
            attribute_domains.append(sorted(values))

        class_label_domain = sorted(set(y_train_bag))

        # Build a single tree with TDIDT
        tree = tdidt(
            instances=train_instances,
            available_attribute_indexes=subset_attributes,
            attribute_domains=attribute_domains,
            class_label_domain=class_label_domain,
            parent_instance_count=len(X_train_bag)
        )

        # Evaluate tree on its OOB set (mirror rf.fit: predict(X_oob, t=tree))
        if len(X_oob) > 0:
            oob_preds = [
                classify_instance_with_tree(x, tree)
                for x in X_oob
            ]
            score = accuracy_score(y_oob, oob_preds)
        else:
            score = 0.0

        trees_with_scores.append((tree, score))

    # Sort and keep top M exactly like rf.fit
    trees_with_scores.sort(key=lambda t: t[1], reverse=True)
    expected_trees = [t for t, _ in trees_with_scores[:M]]

    # The trained forest must match the replayed construction
    assert rf.trees == expected_trees

    # Training twice with the same seed should give identical trees
    rf2 = MyRandomForestClassifier(N=N, M=M, F=F, random_state=seed)
    rf2.fit(X_train_interview, y_train_interview)
    assert rf2.trees == rf.trees


def test_random_forest_classifier_predict():
    """Random forest predictions should equal majority vote of individual trees (deterministic)."""
    N = 5
    M = 3
    F = 2
    seed = 0

    rf = MyRandomForestClassifier(N=N, M=M, F=F, random_state=seed)
    rf.fit(X_train_interview, y_train_interview)

    # Use a mix of seen and slightly varied interview-style instances
    X_test = [
        ["Junior", "Python", "no", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "no", "no"],
        ["Mid", "R", "yes", "yes"], 
    ]

    forest_preds = rf.predict(X_test)

    # Manually compute expected predictions via per-tree majority vote
    expected_preds = []
    for x in X_test:
        votes_for_x = [
            classify_instance_with_tree(x, tree)
            for tree in rf.trees
        ]
        expected_preds.append(vote(votes_for_x))

    # Forest.predict must be exactly majority vote across its trees
    assert forest_preds == expected_preds

    # Determinism check: same seed + same data == same predictions
    rf2 = MyRandomForestClassifier(N=N, M=M, F=F, random_state=seed)
    rf2.fit(X_train_interview, y_train_interview)
    forest_preds_2 = rf2.predict(X_test)
    assert forest_preds_2 == forest_preds
