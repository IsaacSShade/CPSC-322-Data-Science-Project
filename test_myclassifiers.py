import numpy as np

from mysklearn.myclassifiers import MyDecisionTreeClassifier

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

