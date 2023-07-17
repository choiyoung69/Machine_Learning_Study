import mglearn
import matplotlib.pyplot as plt

#훈련 데이터셋을 저장하는 것이 모델을 만드는 전부
X, y = mglearn.datasets.make_forge()

#mglearn.plots에 있는 plot_knn_classfication 적용
mglearn.plots.plot_knn_classification(n_neighbors=1)
plt.show()

mglearn.plots.plot_knn_classification(n_neighbors=3)
plt.show()

from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_forge()
#train, test set으로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors= 3)

clf.fit(X_train, y_train)

print("테스트 세트 예측: ", clf.predict(X_test))

#모델이 얼마나 잘 일반화되었는지 평가
print("테스트 세트 정확도: {:.2f}" .format(clf.score(X_test, y_test)))

