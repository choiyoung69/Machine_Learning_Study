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


##fig = figure 데이터가 담기는 프레임(크기 모양을 변형할 수는 있지만 실제로 프레임위에 글씨를 쓸수는 없음), 
##################################axes = axes 실제 데이터가 그려지는 캔버스
fig, axes = plt.subplots(1, 3, figsize=(10, 3))

for n_neighbors, ax in zip ([1, 3, 9], axes):
    #fit 메소드는 self 오브젝트를 리턴한다
    #그래서 객체생성과 fit 메소드를 한 줄에 쓸 수 있다.
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    #결정경계를 표시한다
    #model 객체를 넣고, train 데이터, 평면 칠하기, 캔버스 설정, 투명도 설정을 해줄 수 있다
    mglearn.plots.plot_2d_classification(clf, X, fill = True, eps = 0.5, ax = ax, alpha=.4)
    #산점도 그리기
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} 이웃" .format(n_neighbors))
    ax.set_xlabel("특성 0")
    ax.set_ylabel("특성 1")
axes[0].legend(loc = 3)
plt.rc('font', family = 'Malgun Gothic')
plt.show()



###################################모델의 복잡도와 일반화 사이의 관계##########################
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
#stratify는 기존 데이터를 나눌 뿐만 아니라 클래스 분포 비율까지 맞춰준다
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state = 66
)

train_accurancy = []
test_accurancy = []
#1에서 10까지 이웃의 수를 결정
neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
    #모델 생성
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    #훈련 세트 정확도 예측
    train_accurancy.append(clf.score(X_train, y_train))
    #일반화 정확도 저장
    test_accurancy.append(clf.score(X_test, y_test))
plt.plot(neighbors_settings, train_accurancy, label="훈련 정확도")
plt.plot(neighbors_settings, test_accurancy, label="테스트 정확도")
plt.ylabel("정확도")
plt.xlabel("n_neighbors")
#그래프 범례 표시
plt.legend()
plt.show()




###############################k-최근접 이웃 회귀#####################################
#plot_knn_regression은 mgleran에 있는 wave 데이터셋을 이용하여
#k-NN 회귀 그래프를 그려준다.
mglearn.plots.plot_knn_regression(n_neighbors = 1)
plt.show()

#n-NN 최근접 이웃 평균
mglearn.plots.plot_knn_regression(n_neighbors=3)
plt.show()

from sklearn.neighbors import KNeighborsRegressor

X, y = mglearn.datasets.make_wave(n_samples = 40)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

#이웃의 개수를 3으로 하여 모델의 객체를 만든다.
reg = KNeighborsRegressor(n_neighbors=3)
#훈련데이터를 이용하여 모델을 학습시킨다
reg.fit(X_train, y_train)

print("테스트 세트 예측: \n", reg.predict(X_test))
print("테스트 세트 R^2: {:.2f}" .format(reg.score(X_test, y_test)))


from sklearn.neighbors import KNeighborsRegressor

X, y = mglearn.datasets.make_wave(n_samples = 40)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

#이웃의 개수를 3으로 하여 모델의 객체를 만든다.
reg = KNeighborsRegressor(n_neighbors=3)
#훈련데이터를 이용하여 모델을 학습시킨다
reg.fit(X_train, y_train)

print("테스트 세트 예측: \n", reg.predict(X_test))
print("테스트 세트 R^2: {:.2f}" .format(reg.score(X_test, y_test)))