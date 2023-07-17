###Machine Learning - book: Introduction to Machine Learning with Python
import mglearn
import matplotlib.pyplot as plt

#make datasets, mglearn이라는 모듈에서 forge라는 데이터셋을 불러온다는 의미
X, y = mglearn.datasets.make_forge()
#x의 형태와 y의 형태를 확인한다 
print(X, '\n', type(X), '\n', X.shape)
print(y, '\n', type(y), '\n', y.shape)

#산점도를 그린다. x[:, 0], x[: 1]이 각각 x축, y축이 되고 (x, x')쌍마다 y에 있는 정답값이 1:1로 할당된다
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
#한국어가 깨지는 것을 막기 위한 폰트 설정
plt.rc('font', family = 'Malgun Gothic')
#legend 함수의 loc 파라미터를 이용해서 범례가 표시될 위치를 설정
plt.legend(["클래스 0", "클래스 1"], loc = 4)
plt.xlabel("첫 번째 특성")
plt.ylabel("두 번째 특성")
plt.show()

#인위적으로 만든 wave 데이터셋 사용, 표본은 40개
X, y = mglearn.datasets.make_wave(n_samples = 40)
#X, y 값 확인
print(X, '\n', type(X), '\n', X.shape)
print(y, '\n', type(y), '\n', y.shape)
#plot으로 데이터셋 그리기
plt.plot(X, y, 'o')
plt.rc('font', family = 'Malgun Gothic')
#마이너스 표현이 안될 떄 설정
plt.rcParams['axes.unicode_minus'] = False
plt.ylim(-3, 3)
plt.xlabel("특성")
plt.ylabel('타깃')
plt.show()

###############################위스콘신 유방암 데이터셋 표현#####################
from sklearn.datasets import load_breast_cancer
import numpy as np

cancer = load_breast_cancer()
#keys()는 딕셔너리에서 key값만 빼올 수 있는 함수
print('cancer.keys() : \n', cancer.keys())
print("유방암 데이터의 형태: ", cancer.data.shape)
print(cancer.target_names)
print(cancer.target)
#zip() 함수는 여러 개의 순회 가능한 객체를 인자로 받고, 각 객체를 튜플형태로 차례로 접근할 수 있는 반복자를 반환한다
#bincount는 0부터 객체의 최대값까지 각 원소의 빈도수를 계산한다
#212개는 악성, 357개는 양성
print("클래스별 샘플 개수: \n", {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))})
print("특성 이름: \n", cancer.feature_names)
print("자세한 특성 설명: \n", cancer.DESCR)

##########################보스턴 주택가격 데이터셋###############################
from sklearn.datasets import load_boston
boston = load_boston()
print("데이터의 형태: ", boston.data.shape)
#이때 13개의 입력 특성뿐만 아니라 특성끼리 곱하여(또는 상호작용이라 부름)
#의도적으로 확장하겠다. 이처럼 특성을 유도해내는 것을 특성 공학이라고 한다

X, y = mglearn.datasets.load_extended_boston()
print("X.shpae: ", X.shape)