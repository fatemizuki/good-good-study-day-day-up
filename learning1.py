import pandas as pd
import matplotlib.pyplot as plt
import mglearn
import numpy as np
import scipy as sp
# from sklearn.datasets import load_iris
# iris_dataset = load_iris()
#
# print("keys of iris_dataset:\n{}".format(iris_dataset.keys()))
#
# print(iris_dataset['DESCR'])
#
# print(iris_dataset['target_names'])
# print(iris_dataset['target'])
# print(type(iris_dataset['data']))s
# print(iris_dataset['feature_names'])

# print(iris_dataset['data'].shape)

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(
#     iris_dataset['data'], iris_dataset['target'], random_state=0)

# print(X_train.shape)
# print(y_train.shape)

# #绘制散点图矩阵
# iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# grr = pd.plotting.scatter_matrix(
#     iris_dataframe, c=y_train, figsize=(
#         15, 15), marker='.', hist_kwds={
#             'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
#
# plt.show()

# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=1)
#
# knn.fit(X_train, y_train)
# print(knn.score(X_test,y_test))
#
# X, y = mglearn.datasets.make_forge()
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# plt.legend(["Class0", "Class1"], loc=4)
# plt.xlabel("First feature")
# plt.ylabel("Second feature")
# plt.show()
# print(X.shape)

# X,y = mglearn.datasets.make_wave(n_samples = 70)
# plt.plot(X, y, '.')
# plt.ylim(-5,5)
# plt.xlabel("Feature")
# plt.ylabel("Target")
# plt.show()

# from sklearn.datasets import load_breast_cancer
# cancer = load_breast_cancer()
# print(cancer.keys())


# from sklearn.datasets import load_boston
# boston = load_boston()
# print(boston.keys())

# X,y = mglearn.datasets.load_extended_boston()
# print(X.shape)

# from sklearn.model_selection import train_test_split
# X,y = mglearn.datasets.make_forge()
#
# X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=7)
#
# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier(n_neighbors=3)
#
# clf.fit(X_train, y_train)
# print(clf.score(X_test,y_test))
#
#
# mglearn.plots.plot_2d_separator(clf,X,fill=True,eps=0.5,alpha=0.4)
#
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# plt.legend()
# plt.show()

# from sklearn.datasets import load_breast_cancer
# from sklearn.neighbors import KNeighborsClassifier
#
# cancer = load_breast_cancer()
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(
#     cancer.data, cancer.target, stratify=cancer.target, random_state=7)
#
# training_accuracy = []
# test_accuracy = []
#
# neighbours_settings = range(1, 11)
#
# for n_neighbours in neighbours_settings:
#     clf = KNeighborsClassifier(n_neighbors=n_neighbours)
#
#     clf.fit(X_train, y_train)
#
#     training_accuracy.append(clf.score(X_train, y_train))
#     test_accuracy.append(clf.score(X_test, y_test))
#
# plt.plot(neighbours_settings, training_accuracy, label='training accuracy')
# plt.plot(neighbours_settings, test_accuracy, label='test accuracy')
#
# plt.legend()
# plt.show()

# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.model_selection import train_test_split
#
# X, y = mglearn.datasets.make_wave(n_samples=40)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#
# training_accuracy = []
# test_accuracy = []
# n_neighbours = range(1, 10)
#
# for n_neighbours in n_neighbours:
#
#     reg = KNeighborsRegressor(n_neighbors=n_neighbours)
#
#     reg.fit(X_train, y_train)
#     training_accuracy.append(reg.score(X_train, y_train))
#     test_accuracy.append(reg.score(X_test, y_test))
#
# print(training_accuracy)
# print(test_accuracy)
#
# plt.plot(training_accuracy, test_accuracy)
# plt.show()
#
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# X, y = mglearn.datasets.make_wave(n_samples=70)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)
#
# lr = LinearRegression().fit(X_train, y_train)
#
# print(lr.coef_)
# print(lr.intercept_)
#
# print(lr.score(X_train, y_train))
# print(lr.score(X_test, y_test))

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
#
# X, y = mglearn.datasets.load_extended_boston()
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)
#
# lr = LinearRegression().fit(X_train, y_train)
#
# print(lr.score(X_train, y_train))
# print(lr.score(X_test, y_test))

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import Ridge
# X, y = mglearn.datasets.load_extended_boston()
#
# alpha1 = [0.001, 0.01, 0.1, 1, 10, 100]
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)
#
# for alpha in alpha1:
#     lr = Ridge(alpha=alpha).fit(X_train, y_train)
#
#     print(
#         "alpha and train_score: {}|| {:.2f}".format(
#             alpha,
#             lr.score(
#                 X_train,
#                 y_train)))
#     print(
#         "alpha and test_score: {}|| {:.2f}".format(
#             alpha,
#             lr.score(
#                 X_test,
#                 y_test)))

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import Lasso
# X, y = mglearn.datasets.load_extended_boston()
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)
#
# alpha1 = 0.001
#
#
# lasso = Lasso(alpha=alpha1, max_iter=100000).fit(X_train, y_train)
#
# print(
#     "alpha {} || train_score {}".format(
#         alpha1,
#         lasso.score(
#             X_train,
#             y_train)))
# print(
#     "alpha {} || test_score {}".format(
#         alpha1,
#         lasso.score(
#             X_test,
#             y_test)))
# print(np.sum(lasso.coef_ != 0))

# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import LinearSVC
#
# X, y = mglearn.datasets.make_forge()
#
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#
# for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
#     clf = model.fit(X, y)
#     mglearn.plots.plot_2d_separator(
#         clf, X, fill=False, eps=0.5, ax=ax, alpha=0.7)
#     mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
#     ax.set_title("{}".format(clf.__class__.__name__))
#     ax.set_xlabel("feature0")
#     ax.set_ylabel("feature1")
# axes[0].legend()
# plt.show()

# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_breast_cancer
# cancer = load_breast_cancer()
#
# X_train, X_test, y_train, y_test = train_test_split(
#     cancer.data, cancer.target, stratify=cancer.target, random_state=7)
#
#
# logreg = LogisticRegression(C=0.001).fit(X_train, y_train)
# print(logreg.score(X_train, y_train))
# print(logreg.score(X_test, y_test))

# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_breast_cancer
#
# cancer = load_breast_cancer()
#
# X_train, X_test, y_train, y_test = train_test_split(
#     cancer.data, cancer.target, stratify=cancer.target, random_state=7)
#
# for C in [0.001, 1, 100]:
#     lr_l1 = LogisticRegression(
#         C=C,
#         penalty='l1',
#         max_iter=100000,
#         solver='liblinear').fit(
#         X_train,
#         y_train)
#     print(lr_l1.score(X_train, y_train))
#     print(lr_l1.score(X_test, y_test))

# from sklearn.datasets import make_blobs
# from sklearn.svm import LinearSVC
# from sklearn.model_selection import train_test_split
#
# X, y = make_blobs(random_state=42)
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)
#
# linear_svm = LinearSVC().fit(X_train, y_train)
# line = np.linspace(-15, 15)
# for coef, intercept, color in zip(
#     linear_svm.coef_, linear_svm.intercept_, [
#         'b', 'r', 'g']):
#     plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
#
# plt.show()

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.datasets import load_breast_cancer
#
# cancer = load_breast_cancer()
#
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(
#     cancer.data, cancer.target, stratify=cancer.target, random_state=7)
#
# tree = DecisionTreeClassifier(max_depth=5,random_state=7)
#
# tree.fit(X_train, y_train)
#
# print(tree.score(X_train, y_train))
# print(tree.score(X_test, y_test))

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_breast_cancer
#
# cancer = load_breast_cancer()
#
# X_train, X_test, y_train, y_test = train_test_split(
#     cancer.data, cancer.target, stratify=cancer.target, random_state=7)
#
# tree = DecisionTreeClassifier(max_depth=5, random_state=7)
#
# tree.fit(X_train, y_train)
#
# from sklearn.tree import export_graphviz
#
# export_graphviz(
#     tree,
#     out_file="tree.dot",
#     class_names=[
#         "malignant",
#         "benign"],
#     feature_names=cancer.feature_names,
#     impurity=False,
#     filled=True)

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_breast_cancer
#
# cancer = load_breast_cancer()
#
# X_train, X_test, y_train, y_test = train_test_split(
#     cancer.data, cancer.target, stratify=cancer.target, random_state=7)
#
# tree = DecisionTreeClassifier(max_depth=5, random_state=7)
#
# tree.fit(X_train, y_train)
#
# print(tree.feature_importances_)
# print(cancer.data.shape)
# print(range(cancer.data.shape[1]))
# def plot_teature_importances_cancer(model):
#     n_features = cancer.data.shape[1]
#     plt.barh(range(n_features),model.feature_importances_,align='center')
#     plt.yticks(np.arange(n_features),cancer.feature_names)
#     plt.xlabel("Feature importance")
#     plt.ylabel("Feature")
#
# plot_teature_importances_cancer(tree)
# plt.show()

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.datasets import make_moons
# from sklearn.model_selection import train_test_split
#
# X,y = make_moons(n_samples=100, noise=0.25,random_state=7)
#
# X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=7)
#
# forest = RandomForestClassifier(n_estimators=5,random_state=7)
#
# forest.fit(X_train,y_train)
#
# print(forest.score(X_train,y_train))
# print(forest.score(X_test,y_test))

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
#
# cancer = load_breast_cancer()
# X_train, X_test, y_train, y_test = train_test_split(
#     cancer.data, cancer.target, stratify=cancer.target, random_state=7)
#
# forest = RandomForestClassifier(n_estimators=100, max_depth =5,random_state=7)
#
# forest.fit(X_train, y_train)
#
# print(forest.score(X_train, y_train))
# print(forest.score(X_test, y_test))
#
# def plot_teature_importances_cancer(model):
#     n_features = cancer.data.shape[1]
#     plt.barh(range(n_features),model.feature_importances_,align='center')
#     plt.yticks(np.arange(n_features),cancer.feature_names)
#     plt.xlabel("Feature importance")
#     plt.ylabel("Feature")
#
# plot_teature_importances_cancer(forest)
# plt.show()
#
# from sklearn.datasets import load_breast_cancer
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.model_selection import train_test_split
#
# cancer = load_breast_cancer()
# X_train, X_test, y_train, y_test = train_test_split(
#     cancer.data, cancer.target, stratify=cancer.target, random_state=7)
#
# gbc = GradientBoostingClassifier(
#     learning_rate=0.01,
#     n_estimators=1000,
#     max_depth=2,
#     random_state=7)
#
# gbc.fit(X_train, y_train)
#
# print(gbc.score(X_train, y_train))
# print(gbc.score(X_test, y_test))
#
#
# def plot_teature_importances_cancer(model):
#     n_features = cancer.data.shape[1]
#     plt.barh(range(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), cancer.feature_names)
#     plt.xlabel("Feature importance")
#     plt.ylabel("Feature")
#
# plot_teature_importances_cancer(gbc)
# plt.show()

# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_breast_cancer
# from sklearn.svm import SVC
#
# cancer = load_breast_cancer()
# X_train, X_test, y_train, y_test = train_test_split(
#     cancer.data, cancer.target, stratify=cancer.target, random_state=17)
#
# svc = SVC()
#
# svc.fit(X_train, y_train)
#
# print(svc.score(X_train, y_train))
# print(svc.score(X_test, y_test))

# from sklearn.neural_network import MLPClassifier
# from sklearn.datasets import make_moons
# from sklearn.model_selection import train_test_split
#
# X,y = make_moons(n_samples=100,noise=0.25,random_state=7)
#
# X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=7)
#
# mlp = MLPClassifier(solver='lbfgs',random_state=7,hidden_layer_sizes=[20,10],activation='tanh')
#
# mlp.fit(X_train,y_train)
#
# mglearn.plots.plot_2d_separator(mlp,X_train,fill=True,alpha=0.5)
#
# mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
#
# plt.show()

# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_breast_cancer
# from sklearn.neural_network import MLPClassifier
#
# cancer = load_breast_cancer()
#
# X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=7)
#
# mlp = MLPClassifier()
#
# mlp.fit(X_train,y_train)
#
# print(mlp.score(X_train,y_train))
# print(mlp.score(X_test,y_test))
# #
# # plt.figure(figsize=(20,5))
# # plt.imshow(mlp.coefs_[0],interpolation='none')
# # plt.yticks(range(30),cancer.feature_names)
# # plt.colorbar()
# # plt.show()
# print(mlp.predict_proba(X_test))

# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import MinMaxScaler
#
# cancer = load_breast_cancer()
#
# X_train, X_test, y_train, y_test = train_test_split(
#     cancer.data, cancer.target, stratify=cancer.target, random_state=7)
#
# scaler = MinMaxScaler()
#
# scaler.fit(X_train)
# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# mlp = MLPClassifier(solver='lbfgs', random_state=7)
# mlp.fit(X_train, y_train)
#
# print(mlp.score(X_train, y_train))
# print(mlp.score(X_test, y_test))
#
#
# mlp.fit(X_train_scaled, y_train)
#
# print(mlp.score(X_train_scaled, y_train))
# print(mlp.score(X_test_scaled, y_test))

# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_breast_cancer
#
# cancer = load_breast_cancer()
# fig, axes = plt.subplots(8, 2, figsize=(10, 8))
# malignant = cancer.data[cancer.target == 0]
# benign = cancer.data[cancer.target == 1]
#
# ax = axes.ravel()
#
# for i in range(16):
#     _, bins = np.histogram(cancer.data[:, i], bins=50)
#     ax[i].hist(malignant[:, i], bins=bins, color=mglearn.cm3(0), alpha=0.5)
#     ax[i].hist(benign[:, i], bins=bins, color=mglearn.cm3(2), alpha=0.5)
#     ax[i].set_title(cancer.feature_names[i])
#     ax[i].set_yticks(())
# ax[0].set_xlabel("feature magnitude")
# ax[0].set_ylabel("frequency")
# ax[0].legend(["malignant", "benign"], loc="best")
# fig.tight_layout()
# plt.show()

# from sklearn.datasets import load_breast_cancer
# from sklearn.preprocessing import StandardScaler
#
# cancer = load_breast_cancer()
# scaler = StandardScaler()
# scaler.fit(cancer.data)
# X_scaled = scaler.transform(cancer.data)
#
# from sklearn.decomposition import PCA
#
# pca = PCA(n_components=2)
# pca.fit(X_scaled)
#
# X_pca = pca.transform(X_scaled)
#
# # mglearn.discrete_scatter(X_pca[:,0],X_pca[:,1],cancer.target)
# # plt.gca().set_aspect("equal")
# # plt.legend(cancer.target_names)
# # plt.show()
#
# print(pca.components_)
#
# plt.matshow(pca.components_)
# plt.colorbar()
# plt.xticks(range(30),cancer.feature_names,rotation=90,ha='left')
# plt.show()

# from sklearn.datasets import load_digits
# from sklearn.decomposition import PCA
#
# digits = load_digits()
#
# # fig,axes=plt.subplots(2,5,figsize=(10,5),subplot_kw={'xticks':(),'yticks':()})
# #
# # for ax,img in zip(axes.ravel(),digits.images):
# #     ax.imshow(img)
# #
# # plt.show()
#
# # pca = PCA(n_components=2)
# # pca.fit(digits.data)
# # digits_pca = pca.transform(digits.data)
# #
# # plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max())
# # plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max())
# # for i in range(len(digits.data)):
# #     plt.text(digits_pca[i, 0], digits_pca[i, 1], str(
# #         digits.target[i]), fontdict={'weight':'bold','size':9})
# #
# # plt.show()
#
# from sklearn.manifold import TSNE
# tsne = TSNE(random_state=7)
# digits_tsne = tsne.fit_transform(digits.data)
#
# plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max())
# plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max())
# for i in range(len(digits.data)):
#     plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(
#         digits.target[i]), fontdict={'weight':'bold','size':9})
#
# plt.show()

# from sklearn.datasets import load_breast_cancer
# from sklearn.decomposition import PCA
#
# cancer = load_breast_cancer()
#
# pca = PCA(n_components=2)
#
# pca.fit(cancer.data)
# cancer_pca = pca.transform(cancer.data)
#
# from sklearn.cluster import KMeans
#
# kmeans = KMeans(n_clusters=2)
#
# kmeans.fit(cancer_pca)
#
# print(kmeans.labels_)
#
# y_pred = kmeans.predict(cancer_pca)
# mglearn.discrete_scatter(cancer_pca[:, 0], cancer_pca[:, 1], y_pred)
#
# plt.show()

# from sklearn.datasets import make_moons
# from sklearn.cluster import KMeans
#
# X,y = make_moons(n_samples=200,noise=0.05,random_state=7)
#
# kmeans = KMeans(n_clusters=10,random_state=7)
# kmeans.fit(X)
#
# y_pred = kmeans.predict(X)
#
# mglearn.discrete_scatter(X[:,0],X[:,1],y_pred)
#
# plt.show()
