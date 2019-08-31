import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from sklearn.model_selection import train_test_split
import mglearn

data = pd.read_csv("/Users/lingyu/Desktop/datasets/titanic/train.csv")


# data = data[['Survived', 'Pclass', 'Age',
#              'SibSp', 'Parch', 'Fare']]
#
# data_dummies = pd.get_dummies(data)
#
# print(list(data_dummies.columns))
#
# features = data_dummies.ix[:,'Pclass':'Embarked_S']
# features = features.fillna(features.mean())
# X = features.values
# y = data.ix[:,'Survived'].values

# demo_df = pd.DataFrame({'integer feature': [0, 3, 2, 3], 'str feature': [
#                        'nao', 'zhong', 'xi', 'zhong']})
# display(demo_df)
# demo_df['integer feature'] = demo_df['integer feature'].astype(str)
# data_dummies = pd.get_dummies(demo_df)
#
# print(data_dummies)


# from sklearn.linear_model import LinearRegression
#
# X,y = mglearn.datasets.make_wave(n_samples = 100)
# line = np.linspace(-3,3,1000,endpoint=False).reshape(-1,1)
#
# bins = np.linspace(-3,3,11)
#
# which_bin = np.digitize(X,bins=bins)
#
# from sklearn.preprocessing import OneHotEncoder
# encoder = OneHotEncoder(sparse=False,categories='auto')
# encoder.fit(which_bin)
#
# X_binned = encoder.transform(which_bin)
#
# X_combined = np.hstack([X_binned,X*X_binned])
#
# print(X_combined.shape)
#
# line_binned = encoder.transform(np.digitize(line,bins=bins))
#
# line_combined = np.hstack([line_binned,line*line_binned])
#
# reg = LinearRegression().fit(X_combined,y)
#
# plt.plot(line,reg.predict(line_combined),label = 'linear regression')
# plt.plot(X[:,0],y,'o',c='k')
#
# plt.show()


# X,y = mglearn.datasets.make_wave(n_samples = 100)
#
# line = np.linspace(-3,3,1000,endpoint=False).reshape(-1,1)
#
# from sklearn.preprocessing import PolynomialFeatures
#
# poly = PolynomialFeatures(degree=7,include_bias=False)
#
# poly.fit(X)
#
# X_poly = poly.transform(X)
#
# print(X_poly.shape)
#
# from sklearn.linear_model import LinearRegression
#
# lr = LinearRegression()
# reg = lr.fit(X_poly,y)
#
# line_poly = poly.transform(line)
#
# plt.plot(line, reg.predict(line_poly))
# plt.plot(X[:,0],y,'o',c = 'k')
#
# plt.show()

# from sklearn.datasets import load_breast_cancer
# from sklearn.preprocessing import PolynomialFeatures
#
# data = load_breast_cancer()
#
# poly = PolynomialFeatures(degree=4).fit(data.data)
# data_poly = poly.transform(data.data)
#
# print(data_poly.shape)
# print(data.data.shape)

# from sklearn.datasets import load_breast_cancer
# from sklearn.feature_selection import SelectPercentile
#
# cancer = load_breast_cancer()
#
# X_train, X_test, y_train, y_test = train_test_split(
#     cancer.data, cancer.target, stratify=cancer.target, random_state=7)
#
# select = SelectPercentile(percentile=50)
# select.fit(X_train, y_train)
#
# X_train_select = select.transform(X_train)
#
# print(X_train_select.shape)
# print(X_train.shape)
#
# mask = select.get_support()
# print(mask)
#
# plt.matshow(mask.reshape(-1, 1), cmap='gray_r')
# plt.show()
#
# from sklearn.linear_model import LogisticRegression
#
# X_test_select = select.transform(X_test)
#
# los = LogisticRegression()
#
# los.fit(X_train, y_train)
# print(los.score(X_train, y_train))
# print(los.score(X_test, y_test))
#
# los.fit(X_train_select, y_train)
# print(los.score(X_train_select, y_train))
# print(los.score(X_test_select, y_test))


# from sklearn.datasets import load_breast_cancer
# from sklearn.feature_selection import SelectFromModel
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
#
# cancer = load_breast_cancer()
# select = SelectFromModel(
#     RandomForestClassifier(
#         n_estimators=100,
#         random_state=7),
#     threshold='median')
#
# X_train, X_test, y_train, y_test = train_test_split(
#     cancer.data, cancer.target, stratify=cancer.target, random_state=7)
#
# select.fit(X_train, y_train)
# X_train_select = select.transform(X_train)
# X_test_select = select.transform(X_test)
#
# log = LogisticRegression()
#
# log.fit(X_train_select, y_train)
#
# print(log.score(X_train_select, y_train))
# print(log.score(X_test_select, y_test))


# from sklearn.datasets import load_breast_cancer
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_selection import RFE
#
# cancer = load_breast_cancer()
# X_train, X_test, y_train, y_test = train_test_split(
#     cancer.data, cancer.target, stratify=cancer.target, random_state=7)
# select = RFE(
#     RandomForestClassifier(
#         n_estimators=100,
#         random_state=7),
#     n_features_to_select=20)
#
# select.fit(X_train, y_train)
# X_train_select = select.transform(X_train)
# X_test_select = select.transform(X_test)
#
# log = LogisticRegression()
# log.fit(X_train_select, y_train)
#
# print(log.score(X_train_select, y_train))
# print(log.score(X_test_select, y_test))

# from sklearn.datasets import load_breast_cancer
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score
#
# cancer = load_breast_cancer()
#
# log = LogisticRegression()
# scores = cross_val_score(log,cancer.data,cancer.target,cv=5)
#
# print(scores)
# print(scores.mean())
#
# from sklearn.model_selection import KFold
# kfold = KFold(n_splits=5,shuffle=True,random_state=7)
#
# score = cross_val_score(log,cancer.data,cancer.target,cv=kfold)
#
# print(score)

# from sklearn.model_selection import LeaveOneOut
# loo = LeaveOneOut()
#
# score = cross_val_score(log,cancer.data,cancer.target,cv = loo)
#
# print(score.mean())

# from sklearn.model_selection import ShuffleSplit
# shuff = ShuffleSplit(n_splits=10,test_size=.3,train_size=.3)
#
# score = cross_val_score(log,cancer.data,cancer.target,cv=shuff)
#
# print(score)
#
# print(score.mean())

# from sklearn.model_selection import GroupKFold
# from sklearn.datasets import make_blobs
# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import LogisticRegression
#
# X,y = make_blobs(n_samples = 15,random_state = 7)
# groups = [0,0,1,1,1,1,1,2,2,2,3,3,3,3,3]
#
# log = LogisticRegression()
#
# scores = cross_val_score(log,X,y,groups,cv=GroupKFold(n_splits=4))
#
# print(scores)


# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
#
# cancer = load_breast_cancer()
# param_grid = {
#     'n_estimators': [
#         50, 100, 150, 200, 250], 'max_depth': [
#             3, 4, 5, 6, 7]}
# grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
#
# X_train, X_test, y_train, y_test = train_test_split(
#     cancer.data, cancer.target, stratify=cancer.target, random_state=7)
#
# grid_search.fit(X_train, y_train)
#
# print(grid_search.score(X_test, y_test))
#
# results = pd.DataFrame(grid_search.cv_results_)
#
# scores = np.array(results.mean_test_score).reshape(5, 5)
# mglearn.tools.heatmap(
#     scores,
#     xlabel='n_estimators',
#     xticklabels=param_grid['n_estimators'],
#     ylabel='max_depth',
#     yticklabels=param_grid['max_depth'],
#     cmap='viridis')
#
# plt.show()
#
# scores = cross_val_score(GridSearchCV(RandomForestClassifier(),param_grid,cv = 5),X,y,cv = 5)

# from sklearn.datasets import load_breast_cancer
# from sklearn.linear_model import LogisticRegression
#
# cancer = load_breast_cancer()
# log = LogisticRegression()
# X_train, X_test, y_train, y_test = train_test_split(
#     cancer.data, cancer.target, stratify=cancer.target, random_state=7)
#
# log.fit(X_train, y_train)
# pre_log = log.predict(X_test)

# from sklearn.metrics import confusion_matrix
#
# confusion = confusion_matrix(y_test, pre_log)
#
# print(confusion)

# from sklearn.metrics import classification_report
# print(classification_report(y_test,pre_log,target_names=["good","bad"]))
#
# # mglearn.plots.plot_decision_threshold()
# # plt.show()
#
# y_pred = log.decision_function(X_test) > -.8
# print(classification_report(y_test,y_pred,target_names=["good","bad"]))

# from sklearn.metrics import precision_recall_curve
# precision, recall, thresholds = precision_recall_curve(
#     y_test, log.decision_function(X_test))
#
# close_zero = np.argmin(np.abs(thresholds))
# plt.plot(
#     precision[close_zero],
#     recall[close_zero],
#     'o',
#     markersize=10,
#     label="threshold zero",
#     fillstyle="none",
#     c='k',
#     mew=2)
# plt.plot(precision,recall,label="precision recall curve")
#
# plt.show()

# from sklearn.metrics import average_precision_score
# ap_log = average_precision_score(y_test,log.decision_function(X_test))
# print(ap_log)

# from sklearn.metrics import roc_curve
# fpr,tpr,thresholds = roc_curve(y_test,log.decision_function(X_test))
#
# plt.plot(fpr,tpr)
# close_zero = np.argmin(np.abs(thresholds))
# plt.plot(fpr[close_zero],tpr[close_zero],'o')
# plt.show()

# from sklearn.datasets import load_digits
# from sklearn.metrics import confusion_matrix
# from sklearn.linear_model import LogisticRegression
# digits = load_digits()
# log = LogisticRegression()
# X_train, X_test, y_train, y_test = train_test_split(
#     digits.data, digits.target, random_state=7)
#
# log.fit(X_train, y_train)
# pre_log = log.predict(X_test)
# print(confusion_matrix(y_test, pre_log))
#
# scores_image = mglearn.tools.heatmap(
#     confusion_matrix(
#         y_test,
#         pre_log),
#     xlabel='predicted label',
#     ylabel='true label',
#     xticklabels=digits.target_names,
#     yticklabels=digits.target_names,
#     cmap=plt.cm.gray_r,
#     fmt= "%d")
# plt.gca().invert_yaxis()
# plt.show()

# from sklearn.metrics import classification_report
# print(classification_report(y_test,pre_log))

# from sklearn.metrics import f1_score
# print(f1_score(y_test,pre_log,average="macro"))

# from sklearn.datasets import load_breast_cancer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
#
# cancer = load_breast_cancer()
# X_train, X_test, y_train, y_test = train_test_split(
#     cancer.data, cancer.target, stratify=cancer.target, random_state=7)
#
# # roc_auc = cross_val_score(
# #     RandomForestClassifier(),
# #     X_train,
# #     y_train,
# #     scoring='roc_auc')
# # print(roc_auc)
#
# param_grid = {'n_estimators':[100,200,300,400,500],'max_depth':[2,3,4,5,6]}
#
# grid = GridSearchCV(RandomForestClassifier(),param_grid,scoring='roc_auc')
# grid.fit(X_train,y_train)
#
# print(grid.best_params_)
# print(grid.best_score_)
# print(grid.score(X_test,y_test))

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=7)

# pipe = Pipeline([("scaler", MinMaxScaler()), ("log", LogisticRegression())])
#
# # pipe.fit(X_train, y_train)
# # print(pipe.score(X_test, y_test))
#
# from sklearn.model_selection import GridSearchCV
# param_grid = {'log__C':[0.0001,0.001,0.01,0.1,1,10,100]}
# grid = GridSearchCV(pipe,param_grid,cv=5)
#
# grid.fit(X_train,y_train)
# print(grid.best_score_)
# print(grid.score(X_test,y_test))
# print(grid.best_params_)

from sklearn.pipeline import make_pipeline

pipe = make_pipeline(MinMaxScaler(),LogisticRegression())

print(pipe.steps)

pipe.fit(X_train,y_train)
print(pipe.score(X_test,y_test))

print(pipe.named_steps['logisticregression'].coef_)
