from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 10)

iris_data = load_iris()
dt_clf = DecisionTreeClassifier(random_state=156)

data = iris_data.data
label = iris_data.target

scores = cross_val_score(dt_clf, data, label, scoring='accuracy', cv = 5)
print("Accuracy per Cross Validation :", np.round(scores, 4))
print("Average Accuracy of Cross Validations :", np.round(np.mean(scores), 4))

print("\n*****************************************************************************")

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=121)

dtree = DecisionTreeClassifier()

parameters = {'max_depth': [1, 2, 3], 'min_samples_split': [2, 3]}
# param_grid의 하이퍼 파라미터들을 3개의 train, test set fold로 나누어서 테스트 수행 설정
### refit=True is default. True면 가장 좋은 파라미터 설정으로 재학습
grid_dtree = GridSearchCV(dtree, param_grid=parameters, cv=5, refit=True, return_train_score=True)

grid_dtree.fit(X_train, y_train)

# GridSearchCV 결과는 cv_results_ 라는 Dictionary 로 저장됨. 보기 편하게 하기 위해 DataFrame으로 변환.
scores_df = pd.DataFrame(grid_dtree.cv_results_)
print(scores_df[['params', 'mean_test_score', 'rank_test_score', 'split0_test_score', 'split1_test_score',
                 'split2_test_score', 'split3_test_score', 'split4_test_score']])

# GridSearchCV의 refit으로 이미 학습된 estimator 반환
esimator = grid_dtree.best_estimator_

print('Optimized parameter from GridSearchCV :', grid_dtree.best_params_)
print('Highest Accuracy from GridSearchCV : {0:.4f}'.format(grid_dtree.best_score_))

pred = grid_dtree.predict(X_test)
print('Test Data accuracy : {0:.4f}'.format(accuracy_score(y_test, pred)))