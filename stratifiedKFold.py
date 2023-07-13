from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

iris = load_iris()
features = iris.data
feature_names = iris.feature_names
label = iris.target
label_names = iris.target_names
iris_df = pd.DataFrame(data=features, columns=feature_names)
iris_df['label'] = label
label_sizes = iris_df['label'].value_counts()
print("Iris DataFrame's label counts : \n", label_sizes)

print('\n*****************************************************************************')
print("Create a KFold object that separates the data into 5 Fold sets.")

kfold = KFold(n_splits=5)
n_iter = 0
for train_index, test_index in kfold.split(iris_df):
    n_iter += 1
    label_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    print("\nCross Validation iteration count : {0}".format(n_iter))
    print("Train label size: \n", label_train.value_counts())
    print("Test label size: \n", label_test.value_counts())
# Folds' Train Subsets and Test Subsets are not equally distributed

print('\n*****************************************************************************')
print("Create a StratifiedKFold object that separates the data into 5 Fold sets.")

skfold = StratifiedKFold(n_splits=5)
n_iter = 0
for train_index, test_index in skfold.split(iris_df, iris_df['label']):
    n_iter += 1
    label_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    print("\nCross Validation iteration count : {0}".format(n_iter))
    print("Train label size: \n", label_train.value_counts())
    print("Test label size: \n", label_test.value_counts())

print('\n*****************************************************************************')
print("Run a Stratified K Fold Cross Validation")

dt_clf = DecisionTreeClassifier(random_state=156)
cv_accuracy = []
n_iter = 0
for train_index, test_index in skfold.split(features, label):
    n_iter += 1
    print("\nIteration Count is :", n_iter)

    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]

    print("Training initiated")
    dt_clf.fit(X_train, y_train)
    print("Prediction initiated")
    pred = dt_clf.predict(X_test)
    print("Iteration Count incremented")

    print("Accuracy Evaluation initiated")
    accuracy = np.round(accuracy_score(y_test, pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]

    print('{0}-th Cross Validation Accuracy : {1}, Train Subset size : {2}, Test Subset size : {3}'.format(n_iter,
                                                                                                           accuracy,
                                                                                                           train_size,
                                                                                                           test_size))
    print('{0}-th Validation Test Subset indexes are : {1}'.format(n_iter, test_index))

    cv_accuracy.append(accuracy)

print('\n*****************************************************************************')
print("Accuracies of Cross Validation is :", cv_accuracy)
print("Average Accuracy is : {0:.4f}".format(np.mean(cv_accuracy)))