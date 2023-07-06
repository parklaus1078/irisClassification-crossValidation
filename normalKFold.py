from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np

iris = load_iris()
features = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(random_state=156)

print('\n*****************************************************************************')
print("Create a KFold object that separates the data into 5 Fold sets, and a list object that will contain the accuracy of the each Fold set.")

folds = 5
kfold = KFold(n_splits=folds)
cv_accuracy = []

print('The data set size of iris data is :', features.shape)
print('Train Subset length :', int(features.shape[0] * (folds - 1)/folds))
print('Test Subset length :', int(features.shape[0] * 1/folds))

print('\n*****************************************************************************')
print("Call out the indexes of Train and Test Subsets of KFold object")

n_iter = 1

# The index of Train and Test Subsets of each fold is returned as array when split() method is used
for train_index, test_index in kfold.split(features):
    print("\nIteration Count is :", n_iter)
    # Features of Train Subset are called by selecting features with train_index returned by split and
    # Features of Test Subset are called by selecting features with test_index returned by split
    X_train, X_test = features[train_index], features[test_index]
    # Labels of Train Subset are called by selecting labels with train_index returned by split and
    # Labels of Test Subset are called by selecting labels with test_index returned by split
    y_train, y_test = label[train_index], label[test_index]

    print("Training initiated")
    dt_clf.fit(X_train, y_train)
    print("Prediction initiated")
    pred = dt_clf.predict(X_test)
    print("Iteration Count incremented")

    print("Accuracy Evaluation initiated")
    accuracy = np.round(accuracy_score(y_test, pred), 4)
    train_size = len(train_index)
    test_size = len(test_index)
    print('{0}-th Cross Validation Accuracy : {1}, Train Subset size : {2}, Test Subset size : {3}'.format(n_iter, accuracy, train_size, test_size))
    print('{0}-th Validation Test Subset indexes are : {1}'.format(n_iter, test_index))

    cv_accuracy.append(accuracy)
    n_iter += 1

print('\n*****************************************************************************')
print("Accuracies of Cross Validation is :", cv_accuracy)
print("Average Accuracy is :", np.mean(cv_accuracy))